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


class SDPGraph:

    def __init__(self,
                 annotated_sentence,
                 arc_indices,
                 arc_tags,
                 ):

        self.arc_indices = arc_indices
        self.arc_tags = arc_tags
        self.sentence = []
        self.pos_tags = []
        self.lemmas = []
        self.predicate_indices = []
        self.top_index = []

        for index, item in enumerate(annotated_sentence):
            self.sentence.append(item["form"])
            self.pos_tags.append(item["pos"])
            self.lemmas.append(item["lemma"])
            if item["top"] == '+':
                self.top_index.append(index)
            if item['pred'] == '+':
                self.predicate_indices.append(index)

        if len(self.top_index) > 1:
            raise NotImplementedError
        elif len(self.top_index) == 0:
            # There are cases that not top node in Graph.
            # We would set the first predicate as top node.
            self.top_index = self.predicate_indices[0]
        else:
            self.top_index = self.top_index[0]

        if self.top_index not in self.active_token_indices():
            # There are cases that top node is not a pedicate, or even not a part of a graph
            # We would set the first predicate as top node.
            self.top_index = self.predicate_indices[0]

        self.node_list = []
        self.index_to_node = {}
        self._G = nx.DiGraph()
    
        for index in self.active_token_indices():
            self.node_list.append(
                SDPNode(
                    index=index,
                    token=self.sentence[index],
                    lemma=self.lemmas[index],
                    pos_tag=self.pos_tags[index]
                )
            )
            self.index_to_node[index] = self.node_list[-1]

        for (child_idx, parent_idx), edge_label in zip(self.arc_indices, self.arc_tags):
            #if child_idx == self.top_index:
                # The top node are allowed to be the child node of some non-top node.
                # Here we revert the relations so that make sure the top node is the root of the tree
            #    parent_idx, child_idx = child_idx, parent_idx
            #    edge_label = self.reverse_relation(edge_label)

            self.add_edge(
                self.index_to_node[parent_idx],
                self.index_to_node[child_idx],
                edge_label,
            )
            #self._G.add_edge(
            #    self.index_to_node(child_idx),
            #    self.index_to_node(parent_idx),
            #    label=edge_label
            #)


        self.top_node = self.index_to_node[self.top_index]

    def active_token_indices(self):
        return list(set(reduce(lambda x, y: x + y, self.arc_indices)))

    def get_src_tokens(self):
        return self.sentence

    def get_list_data(self, bos=None, eos=None, bert_tokenizer=None, max_tgt_length=None):
        tgt_tokens = []
        tgt_pos_tags = []
        tgt_copy_map = []
        tgt_copy_mask = []
        tgt_copy_indices = []
        pos_tag_lut ={
            DEFAULT_OOV_TOKEN: DEFAULT_OOV_TOKEN,
            DEFAULT_PADDING_TOKEN: DEFAULT_OOV_TOKEN
        }
        tgt_index_from_src = {}
        src_index_from_tgt = {}

        head_tags = []
        head_indices = []

        if bos:
            tgt_tokens.append(bos)
            tgt_pos_tags.append(bos)
            tgt_copy_mask.append(0)
            tgt_copy_indices.append(0)
            tgt_copy_map.append((len(tgt_tokens) - 1, len(tgt_tokens) - 1))

        visited = defaultdict(int)

        def travel_graph(node, parent_node=None, tag_with_parent=None, prohibit_nodes_indices=[]):
            tgt_tokens.append(node.token)
            tgt_pos_tags.append(node.pos_tag)
            pos_tag_lut[node.token] = node.pos_tag

            node_index_in_target = len(tgt_tokens) - 1
            node_index_in_source = node.index

            if parent_node is None:
                head_tags.append("root")
                head_indices.append(0)
            else:
                head_tags.append(tag_with_parent)
                head_indices.append(tgt_index_from_src[parent_node.index])
            
            if visited[node.index] == 0:
                visited[node.index] = 1
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

                for child_tag, child_node in node.children:
                    if child_node.index not in prohibit_nodes_indices:
                        travel_graph(
                            child_node,
                            node,
                            child_tag
                        )
                for parent_tag, parent_node in node.parents:
                    if visited[parent_node.index] == 0 and parent_node.index not in prohibit_nodes_indices:
                        travel_graph(
                            parent_node,
                            node,
                            self.reverse_relation(parent_tag),
                            [node.index]
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

        travel_graph(self.top_node)

        if eos:
            tgt_tokens.append(eos)
            tgt_pos_tags.append(eos)
            tgt_copy_mask.append(0)
            tgt_copy_indices.append(0)
            tgt_copy_map.append((len(tgt_tokens) - 1, len(tgt_tokens) - 1))
        
        src_tokens = self.get_src_tokens()
        src_token_ids = None
        src_token_subword_index = None
        src_pos_tags = self.pos_tags
        src_copy_vocab = SourceCopyVocabulary(src_tokens)
        src_copy_indices = src_copy_vocab.index_sequence(tgt_tokens)
        src_copy_map = src_copy_vocab.get_copy_map(src_tokens)
        if bert_tokenizer is not None:
            src_token_ids, src_token_subword_index = bert_tokenizer.tokenize(src_tokens, True)

        src_must_copy_tags = [1 if is_abstract_token(t) else 0 for t in src_tokens]
        src_copy_invalid_ids = set(
            src_copy_vocab.index_sequence(
                [t for t in src_tokens if is_english_punct(t)]
            )
        )

        return {
            "tgt_tokens" : tgt_tokens,
            "tgt_pos_tags": tgt_pos_tags,
            "tgt_copy_indices" : tgt_copy_indices,
            "tgt_copy_map" : tgt_copy_map,
            "tgt_copy_mask" : tgt_copy_mask,
            "src_tokens" : src_tokens,
            "src_token_ids" : src_token_ids,
            "src_token_subword_index" : src_token_subword_index,
            "src_must_copy_tags" : src_must_copy_tags,
            "src_pos_tags": src_pos_tags,
            "src_copy_vocab" : src_copy_vocab,
            "src_copy_indices" : src_copy_indices,
            "src_copy_map" : src_copy_map,
            "pos_tag_lut": pos_tag_lut,
            "head_tags" : head_tags,
            "head_indices" : head_indices,
            "src_copy_invalid_ids" : src_copy_invalid_ids
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
        self.index = index
        self.token = token
        self.lemma = lemma
        self.pos_tag = pos_tag
        self.parents = []
        self.children = []

    def __repr__(self):
        string = "Id: {}\tToken : {}\tLemma : {}\tPOS : {}\n".format(self.index, self.token, self.lemma, self.pos_tag)
        string += "\nParent :\n"
        string += "\n".join(["{} <-- {}, {}".format(label, parent_node.index, parent_node.token) for label, parent_node in self.parents])
        string += "\nChildren :\n"
        string += "\n".join(["{} --> {}, {}".format(label, child_node.index, child_node.token) for label, child_node in self.children])
        return string


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
        return [self.pad_token, self.unk_token]

    def __repr__(self):
        return json.dumps(self.idx_to_token)
