import re
from collections import namedtuple, defaultdict

import nltk

from stog.data.dataset_readers.amr_parsing.io import AMRIO


Entity = namedtuple('Entity', ('span', 'type', 'node', 'confidence'))

entity_map = {
    'Netherlands': 'Dutch',
    'Shichuan': 'Sichuan',
    'France': 'Frence',
    'al-Qaida': 'Al-Qaida',
    'Gorazde': 'Gerlaridy',
    'Sun': 'Solar',
    'China': 'Sino',
    'America': 'US',
    'America': 'U.S.',
    'U.S.': 'US',
    'Georgia': 'GA'
}

tokens_to_join = [
    (['Al', '-', 'Faleh'], 'Al-Faleh', 'NNP', 'PERSON'),
]


def strip_lemma(lemma):
    # Remove twitter '@'.
    if lemma[0] == '@':
        lemma = lemma[1:]
    # Remove '-$' suffix.
    if lemma[-1] == '$':
        lemma = lemma[:-1]
    return lemma


def resolve_conflict_entities(entities):
    index_entity_map = {}
    for entity in entities:
        for index in entity.span:
            if index in index_entity_map:
                _entity = index_entity_map[index]
                if _entity.confidence < entity.confidence:
                    index_entity_map[index] = entity
            else:
                index_entity_map[index] = entity
    node_entity_map = {}
    for entity in index_entity_map.values():
        node_entity_map[entity.node] = entity
    return list(node_entity_map.values())


def group_indexes_to_spans(indexes):
    indexes = list(indexes)
    indexes.sort()
    spans = []
    last_index = None
    for idx in indexes:
        if last_index is None or idx - last_index != 1:
            spans.append([])
        last_index = idx
        spans[-1].append(idx)
    return spans


def add_more_ops(ops):
    more_ops = []
    joined_ops = ' '.join(map(str, ops))
    if joined_ops == '"United" "States"':
        more_ops.append('"America"')
    elif joined_ops == '"World" "War" "II"':
        more_ops.append('"WWII"')
    return ops + more_ops


def reorganize_tokens(amr):
    # for tokens, joined_tokens, pos, ner in tokens_to_join:
    #     span = amr.find_span_indexes(tokens)
    #     if span:
    #         amr.replace_span(span, [joined_tokens], [pos], [ner])

    # Joint the words starting with a cap letter which is followed by '^-\d+$'
    while True:
        span = None
        if len(amr.tokens) < 2:
            break
        for i in range(len(amr.tokens) - 1):
            x, y = amr.tokens[i: i + 2]
            if x.isupper() and re.search(r'^-\d+$', y):
                span = list(range(i, i + 2))
                joined_tokens = ''.join([x, y])
                break
        else:
            break
        amr.replace_span(span, [joined_tokens], ['NNP'], ['ENTITY'])

    # Join the words starting with ['Al', '-'] or ['al', '-'].
    while True:
        span = None
        if len(amr.tokens) < 3:
            break
        for i in range(len(amr.tokens) - 2):
            x, y, z = amr.tokens[i:i + 3]
            if x in ('al', 'Al') and y == '-' and z[0].isupper():
                span = list(range(i, i + 3))
                joined_tokens = ''.join(['Al', y, z])
                break
        else:
            break
        amr.replace_span(span, [joined_tokens], ['NNP'], ['PERSON'])

    # Split word with 'pro-' prefix.
    while True:
        index = None
        for i, lemma in enumerate(amr.lemmas):
            if lemma.lower().startswith('pro-'):
                index = i
                break
        else:
            break
        pos = amr.pos_tags[index]
        ner = amr.ner_tags[index]
        _, lemma = amr.lemmas[index].split('-', 1)
        amr.replace_span([index], ['pro', lemma], ['JJ', pos], ['O', ner])

    # Split word with 'anti-' prefix.
    while True:
        index = None
        for i, lemma in enumerate(amr.lemmas):
            if lemma.lower().startswith('anti-'):
                index = i
                break
        else:
            break
        pos = amr.pos_tags[index]
        ner = amr.ner_tags[index]
        _, lemma = amr.lemmas[index].split('-', 1)
        amr.replace_span([index], ['anti', lemma], ['JJ', pos], ['O', ner])

    # Replace 'NT' in front of '$' with 'Taiwan'.
    for i, token in enumerate(amr.tokens):
        if token == 'NT' and len(amr.tokens) > i + 1 and amr.tokens[i + 1] in ('$', 'dollars', 'dollar'):
            amr.replace_span([i], ['Taiwan'], ['NNP'], ['COUNTRY'])


class Recategorizer:

    def __init__(self, node_utils):
        self.node_utils = node_utils
        self.stemmer = nltk.stem.SnowballStemmer('english').stem

    def recategorize_file(self, file_path):
        for amr in AMRIO.read(file_path):
            self.recategorize_graph(amr)
            yield amr

    def recategorize_graph(self, amr):
        reorganize_tokens(amr)
        amr.stems = [self.stemmer(l) for l in amr.lemmas]
        self.recategorize_name_nodes(amr)

    def recategorize_name_nodes(self, amr):
        graph = amr.graph
        entities = []
        for node in graph.get_nodes():
            if graph.is_name_node(node):
                entity = self._get_aligned_entity(node, amr)
                if entity is not None:
                    entities.append(entity)
        entities = resolve_conflict_entities(entities)
        self._collapse_name_nodes(entities, amr)

    def _get_aligned_entity(self, node, amr):
        ops = node.ops
        if len(ops) == 0:
            return None
        ops = add_more_ops(ops)
        alignment = self._get_alignment_for_ops(ops, amr)
        entity = self._get_entity_for_ops(alignment, node, amr)
        if entity is None:
            print(node)
            import pdb; pdb.set_trace()
        else:
            print(' '.join(amr.tokens[i] for i in entity.span), end='')
            print(' --> ' + ' '.join(map(str, entity.node.ops)))
        return entity

    def _get_alignment_for_ops(self, ops, amr):
        alignment = {}
        for i, op in enumerate(ops):
            for j, token in enumerate(amr.tokens):
                confidence = self._maybe_align_op_to(op, j, amr)
                if confidence > 0:
                    if j not in alignment or (j in alignment and alignment[j][1] < confidence):
                        alignment[j] = (i, confidence)
                    break
        return alignment

    def _maybe_align_op_to(self, op, index, amr):
        if not isinstance(op, str):
            op = str(op)
        if re.search(r'^".*"$', op):
            op = op[1:-1]
        op_lower = op.lower()
        token_lower = amr.tokens[index].lower()
        lemma_lower = amr.lemmas[index].lower()
        stripped_lemma_lower = strip_lemma(lemma_lower)
        if op_lower == token_lower or op_lower == lemma_lower or op_lower == stripped_lemma_lower:
            return 10
        elif self.stemmer(op) == amr.stems[index]:
            return 8
        elif amr.is_named_entity(index) and (
                op_lower[:3] == token_lower[:3] or
                op_lower[:3] == lemma_lower[:3] or
                op_lower[:3] == stripped_lemma_lower[:3]
        ):
            return 5
        elif (op_lower[:3] == token_lower[:3] or
              op_lower[:3] == lemma_lower[:3] or
              op_lower[:3] == stripped_lemma_lower[:3]
        ):
            return 1
        elif op in entity_map:
            return self._maybe_align_op_to(entity_map[op], index, amr)
        else:
            return 0

    def _get_entity_for_ops(self, alignment, node, amr):
        spans = group_indexes_to_spans(alignment.keys())
        spans.sort(key=lambda span: len(span), reverse=True)
        candidate_entities = []
        for span in spans:
            entity = None
            for index in []: # span:
                if amr.is_named_entity(index):
                    entity_span = amr.get_named_entity_span(index)
                    entity_type = amr.ner_tags[index]
                    confidence = sum(alignment[j][1] for j in entity_span if j in alignment)
                    entity = Entity(entity_span, entity_type, node, confidence)
                    break
            if entity is None:
                confidence = sum(alignment[j][1] for j in span)
                entity_type = amr.ner_tags[span[0]]
                if entity_type in ('0', 'O'):
                    entity_type = 'ENTITY'
                entity = Entity(span, entity_type, node, confidence)
            candidate_entities.append(entity)

        if len(candidate_entities):
            return max(candidate_entities, key=lambda entity: entity.confidence)
        return None

    def _collapse_name_nodes(self, entities, amr):
        if len(entities) == 0:
            return
        type_counter = defaultdict(int)
        entities.sort(key=lambda entity: entity.span[-1])
        offset = 0
        for entity in entities:
            type_counter[entity.type] += 1
            abstract = '{}_{}'.format(entity.type, type_counter[entity.type])
            span_with_offset = [index - offset for index in entity.span]
            amr.replace_span(span_with_offset, [abstract], ['NNP'], [entity.type])
            amr.graph.remove_node_ops(entity.node)
            amr.graph.add_node_attribute(entity.node, 'ABSTRACTops', abstract)
            offset += len(entity.span) - 1



if __name__ == '__main__':
    import argparse

    from stog.data.dataset_readers.amr_parsing.node_utils import NodeUtilities as NU

    parser = argparse.ArgumentParser('recategorizer.py')
    parser.add_argument('--amr_train_files', nargs='+', default=[])
    parser.add_argument('--amr_dev_files', nargs='+', required=True)
    parser.add_argument('--json_dir', default='./temp')

    args = parser.parse_args()

    node_utils = NU.from_json(args.json_dir)

    recategorizer = Recategorizer(node_utils)

    for file_path in args.amr_dev_files:
        with open(file_path + '.recategorized', 'w', encoding='utf-8') as f:
            for amr in recategorizer.recategorize_file(file_path):
                f.write(str(amr) + '\n\n')

