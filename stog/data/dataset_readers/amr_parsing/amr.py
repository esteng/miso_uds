import re
import json

import penman
import networkx as nx

from penman import Triple
from collections import defaultdict


# Disable inverting ':mod' relation.
penman.AMRCodec._inversions.pop('domain')
penman.AMRCodec._deinversions.pop('mod')

amr_codec = penman.AMRCodec(indent=6)

WORDSENSE_RE = re.compile(r'-\d\d$')
QUOTED_RE = re.compile(r'^".*"$')


class AMR:

    def __init__(self,
                 id=None,
                 sentence=None,
                 graph=None,
                 tokens=None,
                 lemmas=None,
                 pos_tags=None,
                 ner_tags=None,
                 misc=None):
        self.id = id
        self.sentence = sentence
        self.graph = graph
        self.tokens = tokens
        self.lemmas = lemmas
        self.pos_tags = pos_tags
        self.ner_tags = ner_tags
        self.misc = misc

    def __repr__(self):
        fields = []
        for k, v in dict(
            id=self.id,
            snt=self.sentence,
            tokens=self.tokens,
            lemmas=self.lemmas,
            pos_tags=self.pos_tags,
            ner_tags=self.ner_tags,
            misc=self.misc,
            graph=self.graph
        ).items():
            if v is None:
                continue
            if k == 'misc':
                fields += v
            elif k == 'graph':
                fields.append(amr_codec.encode(v))
            else:
                if not isinstance(v, str):
                    v = json.dumps(v)
                fields.append('# ::{} {}'.format(k, v))
        return '\n'.join(fields)

    def get_src_tokens(self):
        return self.lemmas if self.lemmas else self.sentence.split()


class AMRNode:

    def __init__(self, identifier, attributes=None):
        self.identifier = identifier
        if attributes is None:
            self.attributes = []
        else:
            self.attributes = attributes

    def __hash__(self):
        return hash(self.identifier)

    def __eq__(self, other):
        if not isinstance(other, AMRNode):
            return False
        return self.identifier == other.identifier

    def __repr__(self):
        ret = str(self.identifier)
        for k, v in self.attributes:
            if k == 'instance':
                ret += ' / ' + v
                break
        return ret

    def __str__(self):
        ret = repr(self)
        for key, value in self.attributes:
            if key == 'instance':
                continue
            ret += '\n\t:{} {}'.format(key, value)
        return ret

    @property
    def instance(self):
        for key, value in self.attributes:
            if key == 'instance':
                return value
        else:
            return None

    def remove_attribute(self, attr, value):
        self.attributes.remove((attr, value))

    def add_attribute(self, attr, value):
        self.attributes.append((attr, value))

    def replace_attribute(self, attr, old, new):
        index = self.attributes.index((attr, old))
        self.attributes[index] = (attr, new)

    def get_frame_attributes(self):
        for k, v in self.attributes:
            if isinstance(v, str) and re.search(r'-\d\d$', v):
                yield k, v

    def get_senseless_attributes(self):
        for k, v in self.attributes:
            if isinstance(v, str) and not re.search(r'-\d\d$', v):
                yield k, v


class AMRGraph(penman.Graph):

    def __init__(self, penman_graph):
        super(AMRGraph, self).__init__()
        self._triples = penman_graph._triples
        self._top = penman_graph._top
        self._build_extras()

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

        for edge in self.edges():
            if type(edge.source) is not str:
                print("A source is not string : {} (type : {})".format(edge, type(edge.source)))
                continue
            source = self.variable_to_node[edge.source]
            target = self.variable_to_node[edge.target]
            relation = edge.relation


            if source == target:
                continue

            if edge.inverted:
                source, target, relation = target, source, amr_codec.invert_relation(edge.relation)

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
        if len(edges) != 1:
            return False
        source, target = edges[0]
        return self._G[source][target]['label'] == 'name'

    def remove_edge(self, x, y):
        if isinstance(x, AMRNode) and isinstance(y, AMRNode):
            self._G.remove_edge(x, y)
        if isinstance(x, AMRNode):
            x = x.identifier
        if isinstance(y, AMRNode):
            y = y.identifier
        triples = [t for t in self._triples if t.source != x and t.target != y]
        self._update_penman_graph(triples)

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
        self._triples = triples

    def remove_node_attribute(self, node, attr, value):
        node.remove_attribute(attr, value)
        triples = [t for t in self._triples if t.source != node.identifier and t.relation != attr and t.target != value]
        self._update_penman_graph(triples)

    def add_node_attribute(self, node, attr, value):
        node.add_attribute(attr, value)
        t = penman.Triple(source=node.identifier, relation=attr, target=value)
        self._triples = penman.alphanum_order(self._triples + [t])

    def get_nodes(self):
        return self._G.nodes

    def get_edges(self):
        return self._G.edges

    def get_list_node(self):
        visited = defaultdict(int)
        node_list = []

        def dfs(node, relation, parent):

            node_list.append((node, relation, parent))

            if len(self._G[node]) > 0 and visited[node] == 0:
                visited[node] = 1
                for child_node, child_relation in self._G[node].items():
                    dfs(child_node, child_relation["label"], node)

        dfs(
            self.variable_to_node[self._top],
            'root',
            self.variable_to_node[self._top]
        )

        return node_list


    def get_list_data(self, bos=None, eos=None):
        node_list = self.get_list_node()

        tokens = []
        head_tags = []
        head_index = []

        node_to_idx = defaultdict(list)


        def update_info(node, relation, parent, token):
            head_index.append(node_to_idx[parent][-1])
            head_tags.append(relation)
            tokens.append(str(token))

        for node, relation, parent_node in node_list:

            node_to_idx[node].append(len(tokens))

            instance = [attr[1] for attr in node.attributes if attr[0] =="instance"]
            assert len(instance) == 1
            instance = instance[0]

            update_info(node, relation, parent_node, instance)

            if len(node.attributes) > 1:
                for attr in node.attributes:
                    if attr[0] != "instance":
                        update_info(node, attr[0], parent_node, attr[1])


        # Corefenrence
        offset = 1 if bos else 0
        pad_eos = 1 if eos else 0
        coref_index = [i for i in range(offset + len(tokens) + pad_eos)]

        for node, indices in node_to_idx.items():
            if len(indices) > 1:
                copy_idx = indices[0] + offset
                for token_idx in indices[1:]:
                    coref_index[token_idx + offset] = copy_idx

        coref_map = [(token_idx, copy_idx) for token_idx, copy_idx in enumerate(coref_index)]

        return {
            "amr_tokens" : tokens,
            "coref_index" : coref_index,
            "coref_map" : coref_map,
            "head_tags" : head_tags,
            "head_indices" : head_index
        }

    @classmethod
    def decode(cls, raw_graph_string):
        _graph = amr_codec.decode(raw_graph_string)
        return cls(_graph)
