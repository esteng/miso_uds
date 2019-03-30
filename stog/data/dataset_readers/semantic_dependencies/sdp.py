import re
import json
from collections import defaultdict, Counter
from functools import reduce

import numpy as np
import penman
import networkx as nx
from penman import Triple

from stog.data.vocabulary import DEFAULT_PADDING_TOKEN, DEFAULT_OOV_TOKEN
from stog.data.dataset_readers.amr_parsing.graph_repair import GraphRepair
from stog.utils.string import find_similar_token, is_abstract_token, is_english_punct
from stog.utils import logging


logger = logging.init_logger()


# Disable inverting ':mod' relation.
penman.AMRCodec._inversions.pop('domain')
penman.AMRCodec._deinversions.pop('mod')
from penman import Triple

amr_codec = penman.AMRCodec(indent=6)

WORDSENSE_RE = re.compile(r'-\d\d$')
QUOTED_RE = re.compile(r'^".*"$')
RELATION_ORDER = defaultdict(lambda: 10000)
RELATION_ORDER = [
    "BV_reversed",
    "compound_reversed",
    "measure_reversed",
    "ARG1",
    "ARG2",
    "ARG3",
    "ARG4",
    "ARG1_reversed",
    "ARG2_reversed",
    "ARG3_reversed",
    "ARG4_reversed",
    "loc_reversed"
]
relation_dist = defaultdict(list)


class SDPGraph:

    def __init__(self,
                 annotated_sentence,
                 arc_indices,
                 arc_tags,
                 evaluation=False
                 ):

        self.evaluation = evaluation
        self.annotated_sentence = annotated_sentence
        self.arc_indices = arc_indices
        self.arc_tags = arc_tags
        self.top_node = None
        if evaluation or len(arc_indices) == 0 or len(arc_tags) == 0:
            self.build_node_info(annotated_sentence)
        else:
            self.build_graph(annotated_sentence)

    def build_node_info(self, annotated_sentence):
        self.sentence = []
        self.pos_tags = []
        self.lemmas = []
        self.predicate_indices = []
        self.top_index = []
        self.aux_top_indices = []
        
        if len(annotated_sentence) == 0:
            return

        for index, item in enumerate(annotated_sentence):
            self.sentence.append(item["form"])
            self.pos_tags.append(item["pos"])
            #if "+" in item["lemma"]:
            #    self.lemmas.append(item["form"].lower())
            #else:
            #    self.lemmas.append(item["lemma"])
            self.lemmas.append(item["lemma"].lower())
            if item.get("top", None) == '+':
                self.top_index.append(index)
            if item.get('pred', None) == '+':
                self.predicate_indices.append(index)

        if not self.evaluation:
            if len(self.top_index) > 1:
                raise NotImplementedError
            elif len(self.top_index) == 0:
                # There are cases that not top node in Graph.
                # We would set the first predicate as top node.
                self.top_index = self.predicate_indices[0]
            else:
                self.top_index = self.top_index[0]

            if self.top_index not in self.active_token_indices() and len(self.predicate_indices) > 0:
                # There are cases that top node is not a pedicate, or even not a part of a graph
                # We would set the first predicate as top node.
                self.top_index = self.predicate_indices[0]


    def build_graph(self, annotated_sentence):
        self.build_node_info(annotated_sentence)
        self.node_list = []
        self.src_index_to_node = {}
        self._G = nx.Graph()
    
        for index in self.active_token_indices():
            self.node_list.append(
                SDPNode(
                    index=index,
                    token=self.get_src_tokens()[index],
                    lemma=self.lemmas[index],
                    pos_tag=self.pos_tags[index]
                )
            )
            self.src_index_to_node[index] = self.node_list[-1]

        for (child_idx, parent_idx), edge_label in zip(self.arc_indices, self.arc_tags):
            #if child_idx == self.top_index:
                # The top node are allowed to be the child node of some non-top node.
                # Here we revert the relations so that make sure the top node is the root of the tree
            #    parent_idx, child_idx = child_idx, parent_idx
            #    edge_label = self.reverse_relation(edge_label)

            self.add_edge(
                self.src_index_to_node[parent_idx],
                self.src_index_to_node[child_idx],
                edge_label,
            )
            self._G.add_edge(
                self.src_index_to_node[child_idx],
                self.src_index_to_node[parent_idx],
                label=edge_label
            )


        self.top_node = self.src_index_to_node[self.top_index]
        
        self.dummy_top = SDPNode(
            index=None,
            token="dummy",
            lemma="dummy",
            pos_tag="dummy"
        )

        self.add_edge(
            self.dummy_top,
            self.top_node,
            "root"
        )
        # If there is disconnected subgraph
        self.aux_top_nodes = []
        def find_aux_top_indices(isolated_edges):
            # 1. find aux graph top
            sub_graph_node_edges = Counter()
            for edge, value in isolated_edges.items():
                sub_graph_node_edges[edge[0]] += 1
                sub_graph_node_edges[edge[1]] += 1
                isolated_edges[edge] = 1
                
            return [sub_graph_node_edges.most_common(1)[0]]
        subgraph = list(nx.connected_component_subgraphs(self._G))
        if len(subgraph) > 1:
            #import pdb;pdb.set_trace()
            for graph in subgraph:
                if self.top_node not in graph._node.keys():
                    nodes_num_edges = Counter()
                    for node in sorted(
                            list(graph._node.keys()), key=lambda x: x.src_index
                    ):
                        nodes_num_edges[node] += len(node.children)

                    self.aux_top_nodes.append(
                        nodes_num_edges.most_common(1)[0][0]
                    )

                    self.add_edge(
                        self.dummy_top,
                        nodes_num_edges.most_common(1)[0][0],
                        "root"
                    )


    def active_token_indices(self):
        if len(self.arc_indices) == 0:
            return []
        else:
            return list(set(reduce(lambda x, y: x + y, self.arc_indices)))

    def get_src_tokens(self):
        return self.lemmas

    def get_list_data(self, bos=None, eos=None, bert_tokenizer=None, max_tgt_length=None):
        if self.evaluation:
            return self.get_test_data(bert_tokenizer, max_tgt_length)
        else:
            return self.get_train_data(bos, eos, bert_tokenizer, max_tgt_length)
    
    def get_test_data(self, bert_tokenizer=None, max_tgt_length=None):
        tgt_tokens = [DEFAULT_PADDING_TOKEN, DEFAULT_PADDING_TOKEN]
        tgt_pos_tags = [DEFAULT_PADDING_TOKEN, DEFAULT_PADDING_TOKEN]
        tgt_copy_map = [(0, 0), (1, 1)]
        tgt_copy_mask = [0, 0]
        tgt_copy_indices = [1, 2]

        head_tags = [DEFAULT_PADDING_TOKEN]
        head_indices = [0]
        
        src_tokens = self.get_src_tokens()
        src_token_ids = None
        src_token_subword_index = None
        src_pos_tags = self.pos_tags

        src_copy_vocab = RedundentSourceCopyVocabulary(src_tokens)
        src_copy_indices = [0, 0]
        src_copy_map = src_copy_vocab.get_copy_map()

        pos_tag_lut = RedundentSourceCopyVocabulary(src_pos_tags)
        if bert_tokenizer is not None:
            src_token_ids, src_token_subword_index = bert_tokenizer.tokenize(src_tokens, True)

        src_must_copy_tags = [1 if is_abstract_token(t) else 0 for t in src_tokens]
        src_copy_invalid_ids = set(
                [idx + 1 for idx, t in enumerate(src_tokens) if is_english_punct(t)]
        )

        return {
            "tgt_tokens": tgt_tokens,
            "tgt_pos_tags": tgt_pos_tags,
            "tgt_copy_indices": tgt_copy_indices,
            "tgt_copy_map": tgt_copy_map,
            "tgt_copy_mask": tgt_copy_mask,
            "src_tokens": src_tokens,
            "src_token_ids": src_token_ids,
            "src_token_subword_index": src_token_subword_index,
            "src_must_copy_tags": src_must_copy_tags,
            "src_pos_tags": src_pos_tags,
            "src_copy_vocab": src_copy_vocab,
            "src_copy_indices": src_copy_indices,
            "src_copy_map": src_copy_map,
            "pos_tag_lut": pos_tag_lut,
            "head_tags": head_tags,
            "head_indices": head_indices,
            "src_copy_invalid_ids": src_copy_invalid_ids,
            "isolated_edges": {}
        }

    def get_train_data(
            self,
            bos=None,
            eos=None,
            bert_tokenizer=None,
            max_tgt_length=None
    ):
        tgt_tokens = []
        tgt_pos_tags = []
        tgt_copy_map = []
        tgt_copy_mask = []
        tgt_copy_indices = []

        tgt_index_from_src = {}
        src_index_from_tgt = {}

        head_tags = []
        head_indices = []

        isolated_edges = []

        if bos:
            tgt_tokens.append(bos)
            tgt_pos_tags.append(bos)
            tgt_copy_mask.append(0)
            tgt_copy_indices.append(0)
            tgt_copy_map.append((len(tgt_tokens) - 1, len(tgt_tokens) - 1))
            tgt_index_from_src[0] = 0
            src_index_from_tgt[0] = 0

        edge_visited = {(child_idx, parent_idx) : 0 for child_idx, parent_idx in self.arc_indices}
        node_visited = defaultdict(int)

        def depth_first_search(node, antecedent_nodes=[]):
            node_visited[node.src_index] = 1
            for child_tag, child_node in node.children:

                # don't make the last node visited as children
                if child_node.src_index in antecedent_nodes[-1:]:
                    continue

                # add the node to tree
                if child_node.src_index not in node.tree_children: 
                    node.tree_children[child_node.src_index] = child_tag
                
                if "_reversed" not in child_tag:
                    edge_visited[(child_node.src_index, node.src_index)] = 1
                else:
                    edge_visited[(node.src_index, child_node.src_index)] = 1

                if child_node.src_index not in antecedent_nodes and node_visited[child_node.src_index] == 0:
                    depth_first_search(child_node, antecedent_nodes + [node.src_index])
            
            if node.is_frontier(edge_visited):
                for parent_tag, parent_node in node.parents:
                    if parent_tag == "root":
                        continue
                    if edge_visited[(node.src_index, parent_node.src_index)] == 0:
                        node.children.append((self.reverse_relation(parent_tag), parent_node))


        # Convert tree to graph
        num_edge_visited = 0
        #print(self.sentence)
        while len(edge_visited) > 0 and \
                sum(edge_visited.values()) < len(edge_visited.values()):
            edge_visited = {(child_idx, parent_idx) : 0 for child_idx, parent_idx in self.arc_indices}
            node_visited = defaultdict(int)
            depth_first_search(self.dummy_top)
            if num_edge_visited == sum(edge_visited.values()):
                # print("{}".format(" ".join(self.sentence)))
                # print("Num of isolated nodes : {}".format(sum([1 for v in edge_visited.values() if v == 0])))
                isolated_edges = {edge: visited for edge, visited in edge_visited.items() if visited == 0}
                import pdb;pdb.set_trace()
                #        "root"
                #    )
                break
            num_edge_visited = sum(edge_visited.values())

        node_visited = defaultdict(int)

        def travel_converted_graph(node, parent_node=None, tag_with_parent=None):
            # 1. if node is dummy root
            if parent_node is None:
                for child_node_src_index, child_tag in node.tree_children.items():
                    travel_converted_graph(
                        self.src_index_to_node[child_node_src_index],
                        node,
                        child_tag
                    )
                return
            # 2. normal nodes
            tgt_tokens.append(node.token)
            tgt_pos_tags.append(node.pos_tag)

            node_index_in_target = len(tgt_tokens) - 1
            node_index_in_source = node.src_index

            # record this every time we visit the node.
            src_index_from_tgt[node_index_in_target] = node_index_in_source

            if tag_with_parent is "root":
                head_tags.append("root")
                head_indices.append(0)
            else:
                head_tags.append(tag_with_parent)
                head_indices.append(tgt_index_from_src[parent_node.src_index])
            
            if node_visited[node.src_index] == 0:
                node_visited[node.src_index] = 1
                # Record the 1st time this token appeared.
                tgt_index_from_src[node_index_in_source] = node_index_in_target

                tgt_copy_mask.append(0)
                tgt_copy_map.append(
                    (
                        node_index_in_target, 
                        node_index_in_target
                    )
                )
                tgt_copy_indices.append(0)

                for child_node_src_index, child_tag in node.tree_children.items():
                    travel_converted_graph(
                        self.src_index_to_node[child_node_src_index],
                        node,
                        child_tag
                    )
            else:
                tgt_copy_mask.append(1)
                tgt_copy_map.append(
                    (
                        node_index_in_target,
                        tgt_index_from_src[node_index_in_source]
                    )
                )
                tgt_copy_indices.append(
                    tgt_index_from_src[node_index_in_source]
                )

        if len(self.annotated_sentence) > 0 and self.top_node is not None:
            travel_converted_graph(self.dummy_top)

        if eos:
            tgt_tokens.append(eos)
            tgt_pos_tags.append(eos)
            tgt_copy_mask.append(0)
            tgt_copy_indices.append(0)
            tgt_copy_map.append((len(tgt_tokens) - 1, len(tgt_tokens) - 1))
            src_index_from_tgt[len(tgt_tokens) - 1] = 0
        
        src_tokens = self.get_src_tokens()
        src_token_ids = None
        src_token_subword_index = None
        src_pos_tags = self.pos_tags

        src_copy_vocab = RedundentSourceCopyVocabulary(src_tokens)
        src_copy_indices = src_copy_vocab.index_sequence(tgt_tokens, src_index_from_tgt, bos, eos)
        src_copy_map = src_copy_vocab.get_copy_map()

        pos_tag_lut = RedundentSourceCopyVocabulary(src_pos_tags)
        if bert_tokenizer is not None:
            src_token_ids, src_token_subword_index = bert_tokenizer.tokenize(src_tokens, True)

        src_must_copy_tags = [1 if is_abstract_token(t) else 0 for t in src_tokens]
        src_copy_invalid_ids = set(
                [idx + 1 for idx, t in enumerate(src_tokens) if is_english_punct(t)]
        )

        #if len(self.aux_top_nodes) > 0:
        #    import pdb; pdb.set_trace()
        return {
            "tgt_tokens": tgt_tokens,
            "tgt_pos_tags": tgt_pos_tags,
            "tgt_copy_indices": tgt_copy_indices,
            "tgt_copy_map": tgt_copy_map,
            "tgt_copy_mask": tgt_copy_mask,
            "src_tokens": src_tokens,
            "src_token_ids": src_token_ids,
            "src_token_subword_index": src_token_subword_index,
            "src_must_copy_tags": src_must_copy_tags,
            "src_pos_tags": src_pos_tags,
            "src_copy_vocab": src_copy_vocab,
            "src_copy_indices": src_copy_indices,
            "src_copy_map": src_copy_map,
            "pos_tag_lut": pos_tag_lut,
            "head_tags": head_tags,
            "head_indices": head_indices,
            "src_copy_invalid_ids": src_copy_invalid_ids,
            "isolated_edges": isolated_edges
        }
    
    @staticmethod
    def add_edge(parent_node, child_node, label):
        parent_node.children.append((label, child_node))
        child_node.parents.append((label, parent_node))

    @staticmethod
    def reverse_relation(label):
        return label + "_reversed"



class SDPNode:

    def __init__(
            self,
            index,
            token,
            lemma,
            pos_tag
    ):
        self.src_index = index
        self.token = token
        self.lemma = lemma
        self.pos_tag = pos_tag
        self.parents = []
        self.children = []
        self.tree_children = {}

    def is_frontier(self, edge_visited):
        if self.src_index is None:
            return False

        num_unvisited_incoming_edge = 0
        for parent_tag, parent_node in self.parents:
            if parent_tag != "root":
                num_unvisited_incoming_edge += 1 - edge_visited[(self.src_index, parent_node.src_index)]
        return num_unvisited_incoming_edge != 0

    def __repr__(self):
        string = "Id: {}\tToken : {}\tLemma : {}\tPOS : {}\n".format(self.src_index, self.token, self.lemma, self.pos_tag)
        string += "\nParent :\n"
        string += "\n".join(["{} <-- {}, {}".format(label, parent_node.src_index, parent_node.token) for label, parent_node in self.parents])
        string += "\nChildren :\n"
        string += "\n".join(["{} --> {}, {}".format(label, child_node.src_index, child_node.token) for label, child_node in self.children])
        return string

#TODO: leave OOV tokens now since we now every token on target side is a copy from source
#However in some case we might need a dummy node to deal with multi-top graph

class SourceCopyVocabulary:
    def __init__(self, sentence, pad_token=DEFAULT_PADDING_TOKEN, unk_token=DEFAULT_OOV_TOKEN):
        if type(sentence) is not list:
            sentence = sentence.split(" ")

        self.src_tokens = sentence
        self.pad_token = pad_token
        self.unk_token = unk_token

        self.token_to_idx = {self.pad_token : 0, self.unk_token : 1}
        self.idx_to_token = {0 : self.pad_token, 1 : self.unk_token}

        self.vocab_size = 2

        for token in sentence:
            if token not in self.token_to_idx:
                self.token_to_idx[token] = self.vocab_size
                self.idx_to_token[self.vocab_size] = token
                self.vocab_size += 1

    def get_token_from_idx(self, idx):
        return self.idx_to_token[idx]

    def get_token_idx(self, token):
        return self.token_to_idx.get(token, self.token_to_idx[self.unk_token])

    def index_sequence(self, list_tokens):
        return [self.get_token_idx(token) for token in list_tokens]

    def get_copy_map(self, list_tokens):
        src_indices = [self.get_token_idx(self.unk_token)] + self.index_sequence(list_tokens)
        return [
            (src_idx, src_token_idx) for src_idx, src_token_idx in enumerate(src_indices)
        ]

    def get_special_tok_list(self):
        list_to_return = [self.pad_token]
        if self.unk_token:
            list_to_return.append(self.unk_token)
        return list_to_return

    def __repr__(self):
        return json.dumps(self.idx_to_token)

class RedundentSourceCopyVocabulary(SourceCopyVocabulary):
    def __init__(self, sentence, pad_token=DEFAULT_PADDING_TOKEN, unk_token=None):
        #TODO: leave OOV tokens now since we now every token on target side is a copy from source
        #However in some case we might need a dummy node to deal with multi-top graph
        if type(sentence) is not list:
            sentence = sentence.split(" ")

        self.src_tokens = sentence
        self.pad_token = pad_token
        self.num_special = 1

        self.idx_to_token = {0 : self.pad_token}

        if unk_token:
            self.unk_token = unk_token
            self.idx_to_token[1] = self.unk_token
            self.num_special += 1
        else:
            self.unk_token = None

        self.vocab_size = 1

        for token in sentence:
            self.idx_to_token[self.vocab_size] = token
            self.vocab_size += 1
    
    def get_copy_map(self, *args):
        map_to_return = []
        for i in range(self.vocab_size):
            map_to_return.append((i, i))
        return map_to_return

    def index_sequence(self, tgt_tokens, src_index_from_tgt, bos=None, eos=None):
        tokens_not_from_source = [token for token in tgt_tokens if token not in self.src_tokens and token not in [bos, eos]]
        if len(tokens_not_from_source):
            raise ConfigurationError(
                "Only complete copy is support right now, [{}] there tokens are not in source tgt_tokens".format()
            )
        else:
            indices = [self.num_special + src_index_from_tgt[tgt_index] for tgt_index in range(len(tgt_tokens))]
            if bos:
                indices[0] = 0
            if eos:
                indices[-1] = 0 

            return indices
    
    def get_special_tok_list(self):
        #TODO: a hack here, need to be fix later
        list_to_return = [self.pad_token, self.pad_token]
        return list_to_return


        
