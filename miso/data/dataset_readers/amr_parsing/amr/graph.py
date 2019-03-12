import re
import json
from collections import defaultdict, Counter

import numpy as np
import penman
import networkx as nx
from penman import Triple

from miso.data.dataset_readers.amr_parsing.graph_repair import GraphRepair
from miso.utils import logging

from .node import AMRNode
from .src_copy_vocab import SourceCopyVocabulary
from .utils import recover_triples_from_prediction, prepare_stog_instance


logger = logging.init_logger()

# Disable inverting ':mod' relation.
penman.AMRCodec._inversions.pop('domain')
penman.AMRCodec._deinversions.pop('mod')
from penman import Triple

amr_codec = penman.AMRCodec(indent=6)


class AMRGraph(penman.Graph):

    edge_label_priority = (
        'mod name time location degree poss domain quant manner unit purpose topic condition part-of compared-to '
        'duration source ord beneficiary concession direction frequency consist-of example medium location-of '
        'manner-of quant-of time-of instrument prep-in destination accompanier prep-with extent instrument-of age '
        'path concession-of subevent-of prep-as prep-to prep-against prep-on prep-for degree-of prep-under part '
        'condition-of prep-without topic-of season duration-of poss-of prep-from prep-at range purpose-of source-of '
        'subevent example-of value path-of scale conj-as-if prep-into prep-by prep-on-behalf-of medium-of prep-among '
        'calendar beneficiary-of prep-along-with extent-of age-of frequency-of dayperiod accompanier-of '
        'destination-of prep-amid prep-toward prep-in-addition-to ord-of name-of weekday direction-of prep-out-of '
        'timezone subset-of'.split())

    def __init__(self, penman_graph):
        super(AMRGraph, self).__init__()
        self._triples = penman_graph._triples
        self._top = penman_graph._top
        self._build_extras()
        self._src_tokens = []

    def __str__(self):
        self._triples = penman.alphanum_order(self._triples)
        return amr_codec.encode(self)

    def _build_extras(self):
        G = nx.DiGraph()

        self.variable_to_node = {}
        for v in self.variables():
            if type(v) is not str:
                continue
            attributes = [(t.relation, t.target) for t in self.attributes(source=v)]
            node = AMRNode(v, attributes)
            G.add_node(node)
            self.variable_to_node[v] = node

        edge_set = set()
        for edge in self.edges():
            if type(edge.source) is not str:
                logger.warn("A source is not string : {} (type : {})".format(edge, type(edge.source)))
                continue
            source = self.variable_to_node[edge.source]
            target = self.variable_to_node[edge.target]
            relation = edge.relation

            if relation == 'instance':
                continue

            if source == target:
                continue

            if edge.inverted:
                source, target, relation = target, source, amr_codec.invert_relation(edge.relation)

            if (source, target) in edge_set:
                target = target.copy()

            edge_set.add((source, target))
            G.add_edge(source, target, label=relation)

        self._G = G

    def attributes(self, source=None, relation=None, target=None):
        # Refine attributes because there's a bug in penman.attributes()
        # See https://github.com/goodmami/penman/issues/29
        attrmatch = lambda a: (
                (source is None or source == a.source) and
                (relation is None or relation == a.relation) and
                (target is None or target == a.target)
        )
        variables = self.variables()
        attrs = [t for t in self.triples() if t.target not in variables or t.relation == 'instance']
        return list(filter(attrmatch, attrs))

    def _update_penman_graph(self, triples):
        self._triples = triples
        if self._top not in self.variables():
            self._top = None

    def is_name_node(self, node):
        edges = list(self._G.in_edges(node))
        return any(self._G[source][target].get('label', None) == 'name' for source, target in edges)

    def get_name_node_type(self, node):
        edges = list(self._G.in_edges(node))
        for source, target in edges:
            if self._G[source][target].get('label', None) == 'name':
                return source.instance
        raise KeyError

    def get_name_node_wiki(self, node):
        edges = list(self._G.in_edges(node))
        for source, target in edges:
            if self._G[source][target].get('label', None) == 'name':
                for attr, value in source.attributes:
                    if attr == 'wiki':
                        if value != '-':
                            value = value[1:-1]  # remove quotes
                        return value
        return None

    def set_name_node_wiki(self, node, wiki):
        edges = list(self._G.in_edges(node))
        parent = None
        for source, target in edges:
            if self._G[source][target].get('label', None) == 'name':
                parent = source
                break
        if parent:
            if not all(attr != 'wiki' for attr, _ in parent.attributes):
                logger.warn('Multiple wikis are been set.')
            if wiki != '-':
                wiki = '"{}"'.format(wiki)
            self.add_node_attribute(parent, 'wiki', wiki)

    def is_date_node(self, node):
        return node.instance == 'date-entity'

    def add_edge(self, source, target, label):
        self._G.add_edge(source, target, label=label)
        t = penman.Triple(source=source.identifier, relation=label, target=target.identifier)
        triples = self._triples + [t]
        triples = penman.alphanum_order(triples)
        self._update_penman_graph(triples)

    def remove_edge(self, x, y):
        if isinstance(x, AMRNode) and isinstance(y, AMRNode):
            self._G.remove_edge(x, y)
        if isinstance(x, AMRNode):
            x = x.identifier
        if isinstance(y, AMRNode):
            y = y.identifier
        triples = [t for t in self._triples if not (t.source == x and t.target == y)]
        self._update_penman_graph(triples)

    def update_edge_label(self, x, y, old, new):
        self._G[x][y]['label'] = new
        triples = []
        for t in self._triples:
            if t.source == x.identifier and t.target == y.identifier and t.relation == old:
                t = Triple(x.identifier, new, y.identifier)
            triples.append(t)
        self._update_penman_graph(triples)

    def add_node(self, instance):
        identifier = instance[0]
        assert identifier.isalpha()
        if identifier in self.variables():
            i = 2
            while identifier + str(i) in self.variables():
                i += 1
            identifier += str(i)
        triples = self._triples + [Triple(identifier, 'instance', instance)]
        self._triples = penman.alphanum_order(triples)

        node = AMRNode(identifier, [('instance', instance)])
        self._G.add_node(node)
        return node

    def remove_node(self, node):
        self._G.remove_node(node)
        triples = [t for t in self._triples if t.source != node.identifier]
        self._update_penman_graph(triples)

    def replace_node_attribute(self, node, attr, old, new):
        node.replace_attribute(attr, old, new)
        triples = []
        found = False
        for t in self._triples:
            if t.source == node.identifier and t.relation == attr and t.target == old:
                found = True
                t = penman.Triple(source=node.identifier, relation=attr, target=new)
            triples.append(t)
        if not found:
            raise KeyError
        self._triples = penman.alphanum_order(triples)

    def remove_node_attribute(self, node, attr, value):
        node.remove_attribute(attr, value)
        triples = [t for t in self._triples if not (t.source == node.identifier and t.relation == attr and t.target == value)]
        self._update_penman_graph(triples)

    def add_node_attribute(self, node, attr, value):
        node.add_attribute(attr, value)
        t = penman.Triple(source=node.identifier, relation=attr, target=value)
        self._triples = penman.alphanum_order(self._triples + [t])

    def remove_node_ops(self, node):
        ops = []
        for attr, value in node.attributes:
            if re.search(r'^op\d+$', attr):
                ops.append((attr, value))
        for attr, value in ops:
            self.remove_node_attribute(node, attr, value)

    def remove_subtree(self, root):
        children = []
        removed_nodes = set()
        for _, child in list(self._G.edges(root)):
            self.remove_edge(root, child)
            children.append(child)
        for child in children:
            if len(list(self._G.in_edges(child))) == 0:
                removed_nodes.update(self.remove_subtree(child))
        if len(list(self._G.in_edges(root))) == 0:
            self.remove_node(root)
            removed_nodes.add(root)
        return removed_nodes

    def get_subtree(self, root, max_depth):
        if max_depth == 0:
            return []
        nodes = [root]
        children = [child for _, child in self._G.edges(root)]
        nodes += children
        for child in children:
            if len(list(self._G.in_edges(child))) == 1:
                nodes = nodes + self.get_subtree(child, max_depth - 1)
        return nodes

    def get_nodes(self):
        return self._G.nodes

    def get_edges(self):
        return self._G.edges

    def set_src_tokens(self, sentence):
        if type(sentence) is not list:
            sentence = sentence.split(" ")
        self._src_tokens = sentence

    def get_src_tokens(self):
        return self._src_tokens

    def get_list_node(self, replace_copy=True):
        visited = defaultdict(int)
        node_list = []

        def dfs(node, relation, parent):

            node_list.append((
                node if node.copy_of is None or not replace_copy else node.copy_of,
                relation,
                parent if parent.copy_of is None or not replace_copy else parent.copy_of))

            if len(self._G[node]) > 0 and visited[node] == 0:
                visited[node] = 1
                for child_node, child_relation in self.sort_edges(self._G[node].items()):
                    dfs(child_node, child_relation["label"], node)

        dfs(
            self.variable_to_node[self._top],
            'root',
            self.variable_to_node[self._top]
        )

        return node_list

    def sort_edges(self, edges):
        return edges

    def get_tgt_tokens(self):
        node_list = self.get_list_node()

        tgt_token = []
        visited = defaultdict(int)

        for node, relation, parent_node in node_list:
            instance = [attr[1] for attr in node.attributes if attr[0] == "instance"]
            assert len(instance) == 1
            tgt_token.append(str(instance[0]))

            if len(node.attributes) > 1 and visited[node] == 0:
                for attr in node.attributes:
                    if attr[0] != "instance":
                        tgt_token.append(str(attr[1]))

            visited[node] = 1

        return tgt_token

    def get_meta_data(self):
        node_list = self.get_list_node()

        tgt_tokens = []
        head_tags = []
        head_indices = []

        node_to_idx = defaultdict(list)
        visited = defaultdict(int)

        def update_info(node, relation, parent, token):
            head_indices.append(1 + node_to_idx[parent][-1])
            head_tags.append(relation)
            tgt_tokens.append(str(token))

        for node, relation, parent_node in node_list:

            node_to_idx[node].append(len(tgt_tokens))

            instance = [attr[1] for attr in node.attributes if attr[0] == "instance"]
            assert len(instance) == 1
            instance = instance[0]

            update_info(node, relation, parent_node, instance)

            if len(node.attributes) > 1 and visited[node] == 0:
                for attr in node.attributes:
                    if attr[0] != "instance":
                        update_info(node, attr[0], node, attr[1])

            visited[node] = 1

        head_indices[node_to_idx[self.variable_to_node[self.top]][0]] = 0

        return tgt_tokens, head_tags, head_indices, node_to_idx

    def get_stog_data(self, amr, bos=None, eos=None, bert_tokenizer=None, max_tgt_length=None):
        src_tokens = self.get_src_tokens()
        src_pos_tags = amr.pos_tags
        tgt_tokens, head_tags, head_indices, node_to_idx = self.get_meta_data()
        return prepare_stog_instance(
            src_tokens, src_pos_tags, tgt_tokens, head_tags, head_indices, node_to_idx,
            bos, eos, bert_tokenizer, max_tgt_length
        )

    @classmethod
    def decode(cls, raw_graph_string):
        _graph = amr_codec.decode(raw_graph_string)
        return cls(_graph)

    @classmethod
    def from_lists(cls, all_list):
        head_tags = all_list['head_tags']
        head_indices = all_list['head_indices']
        tgt_tokens = all_list['tokens']

        tgt_copy_indices = all_list['coref']
        variables = []
        variables_count = defaultdict(int)
        for i, token in enumerate(tgt_tokens):
            if tgt_copy_indices[i] != i:
                variables.append(variables[tgt_copy_indices[i]])
            else:
                if token[0] in variables_count:
                    variables.append(token[0] + str(variables_count[token[0]]))
                else:
                    variables.append(token[0])

                variables_count[token[0]] += 1

        Triples = []
        for variable, token in zip(variables, tgt_tokens):
            Triples.append(Triple(variable, "instance", token))
            Triples.append(
                Triple(
                    head_indices[variable],
                    head_tags[variable],
                    variable
                )
            )

    @classmethod
    def from_prediction(cls, prediction):
        top, nodes, triples = recover_triples_from_prediction(prediction)
        graph = penman.Graph()
        graph._top = top
        graph._triples = [penman.Triple(*t) for t in triples]
        graph = cls(graph)
        try:
            GraphRepair.do(graph, nodes)
            amr_codec.encode(graph)
        except Exception as e:
            logger.warn('Graph repairing failed.')
            # logger.error(e, exc_info=True)
            # import pdb; pdb.set_trace()
            graph._top = top
            graph._triples = [penman.Triple(*t) for t in triples]
            graph = cls(graph)
        return graph
