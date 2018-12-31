import re
import json

import penman
import networkx as nx

from penman import Triple
from collections import defaultdict

import json


# Disable inverting ':mod' relation.
penman.AMRCodec._inversions.pop('domain')
penman.AMRCodec._deinversions.pop('mod')
from penman import Triple

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
                 abstract_map=None,
                 misc=None):
        self.id = id
        self.sentence = sentence
        self.graph = graph
        self.tokens = tokens
        self.lemmas = lemmas
        self.pos_tags = pos_tags
        self.ner_tags = ner_tags
        self.abstract_map = abstract_map
        self.misc = misc

    def is_named_entity(self, index):
        return self.ner_tags[index] not in ('0', 'O')

    def get_named_entity_span(self, index):
        if self.ner_tags is None or not self.is_named_entity(index):
            return []
        span = [index]
        tag = self.ner_tags[index]
        prev = index - 1
        while prev > 0 and self.ner_tags[prev] == tag:
            span.append(prev)
            prev -= 1
        next = index + 1
        while next < len(self.ner_tags) and self.ner_tags[next] == tag:
            span.append(next)
            next += 1
        return span

    def find_span_indexes(self, span):
        for i, token in enumerate(self.tokens):
            if token == span[0]:
                _span = self.tokens[i: i + len(span)]
                if len(_span) == len(span) and all(x == y for x, y in zip(span, _span)):
                    return list(range(i, i + len(span)))
        return None

    def replace_span(self, indexes, new, pos=None, ner=None):
        self.tokens = self.tokens[:indexes[0]] + new + self.tokens[indexes[-1] + 1:]
        self.lemmas = self.lemmas[:indexes[0]] + new + self.lemmas[indexes[-1] + 1:]
        if pos is None:
            pos = [self.pos_tags[indexes[0]]]
        self.pos_tags = self.pos_tags[:indexes[0]] + pos + self.pos_tags[indexes[-1] + 1:]
        if ner is None:
            ner = [self.ner_tags[indexes[0]]]
        self.ner_tags = self.ner_tags[:indexes[0]] + ner + self.ner_tags[indexes[-1] + 1:]

    def __repr__(self):
        fields = []
        for k, v in dict(
            id=self.id,
            snt=self.sentence,
            tokens=self.tokens,
            lemmas=self.lemmas,
            pos_tags=self.pos_tags,
            ner_tags=self.ner_tags,
            abstract_map=self.abstract_map,
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

    @property
    def ops(self):
        ops = []
        for key, value in self.attributes:
            if re.search(r'op\d+', key):
                ops.append((int(key[2:]), value))
        if len(ops):
            ops.sort(key=lambda x: x[0])
        return [v for k, v in ops]

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
        self._src_tokens = []

    def __str__(self):
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

        for edge in self.edges():
            if type(edge.source) is not str:
                print("A source is not string : {} (type : {})".format(edge, type(edge.source)))
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

    def is_date_node(self, node):
        return node.instance == 'date-entity'

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

            instance = [attr[1] for attr in node.attributes if attr[0] =="instance"]
            assert len(instance) == 1
            instance = instance[0]

            update_info(node, relation, parent_node, instance)

            if len(node.attributes) > 1 and visited[node] == 0:
                for attr in node.attributes:
                    if attr[0] != "instance":
                        update_info(node, attr[0], parent_node, attr[1])
            visited[node] = 1

        copy_offset = 0
        if bos:
            tgt_tokens = [bos] + tgt_tokens
            copy_offset += 1
        if eos:
            tgt_tokens = tgt_tokens + [eos]

        head_indices[node_to_idx[self.variable_to_node[self.top]][0]] = 0

        # Target side Coreference
        tgt_copy_indices = [i for i in range(len(tgt_tokens))]

        for node, indices in node_to_idx.items():
            if len(indices) > 1:
                copy_idx = indices[0] + copy_offset
                for token_idx in indices[1:]:
                    tgt_copy_indices[token_idx + copy_offset] = copy_idx

        tgt_copy_map = [(token_idx, copy_idx) for token_idx, copy_idx in enumerate(tgt_copy_indices)]

        for i, copy_index in enumerate(tgt_copy_indices):
            # Set the coreferred target to 0 if no coref is available.
            if i == copy_index:
                tgt_copy_indices[i] = 0

        # Source Copy
        src_tokens = self.get_src_tokens()
        src_copy_vocab = SourceCopyVocabulary(src_tokens)
        src_copy_indices = src_copy_vocab.index_sequence(tgt_tokens)
        src_copy_map = src_copy_vocab.get_copy_map(src_tokens)

        return {
            "tgt_tokens" : tgt_tokens,
            "tgt_copy_indices" : tgt_copy_indices,
            "tgt_copy_map" : tgt_copy_map,
            "src_tokens" : src_tokens,
            "src_copy_vocab" : src_copy_vocab,
            "src_copy_indices" : src_copy_indices,
            "src_copy_map" : src_copy_map,
            "head_tags" : head_tags,
            "head_indices" : head_indices,
        }

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

        def is_attribute_value(value):
            return re.search(r'(^".*"$|^[^a-zA-Z]+$)', value) is not None

        nodes = prediction['nodes']
        heads = prediction['heads']
        corefs = prediction['corefs']
        head_labels = prediction['head_labels']

        triples = []
        top = None
        # Build the variable map from variable to instance.
        variable_map = {}
        for coref_index in corefs:
            node = nodes[coref_index - 1]
            if is_attribute_value(node):
                continue
            variable_map['vv{}'.format(coref_index)] = node
        for head_index in heads:
            if head_index == 0:
                continue
            node = nodes[head_index - 1]
            coref_index = corefs[head_index - 1]
            variable_map['vv{}'.format(coref_index)] = node
        # Build edge triples and other attribute triples.
        for i, head_index in enumerate(heads):
            if head_index == 0:
                top_variable = 'vv{}'.format(corefs[i])
                if top_variable not in variable_map:
                    variable_map[top_variable] = nodes[i]
                top = top_variable
                continue
            head_variable = 'vv{}'.format(corefs[head_index - 1])
            modifier = nodes[i]
            modifier_variable = 'vv{}'.format(corefs[i])
            label = head_labels[i]
            assert head_variable in variable_map
            if modifier_variable in variable_map:
                triples.append((head_variable, label, modifier_variable))
            else:
                # Add quotes if there's a backslash.
                if '/' in modifier and not re.search(r'^".*"$', modifier):
                    modifier = '"{}"'.format(modifier)
                triples.append((head_variable, label, modifier))

        for var, node in variable_map.items():
            if '/' in node and not re.search(r'^".*"$', node):
                node = '"{}"'.format(node)
            triples.append((var, 'instance', node))

        triples.sort()
        graph = penman.Graph()
        graph._top = top
        graph._triples = [penman.Triple(*t) for t in triples]
        # import pdb; pdb.set_trace()
        return cls(graph)


class SourceCopyVocabulary:
    def __init__(self, sentence, pad_token="@@PAD@@", unk_token="@@UNKNOWN@@"):
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
