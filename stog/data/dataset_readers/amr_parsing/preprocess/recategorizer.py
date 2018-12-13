import os
import re
import json
from collections import defaultdict

import nltk

from stog.data.dataset_readers.amr_parsing.io import AMRIO
from stog.data.dataset_readers.amr_parsing.preprocess import input_cleaner
from stog.utils import logging


logger = logging.init_logger()


class Entity:

    # Sometimes there are different ways to say the same thing.
    entity_map = {
        'Netherlands': ['Dutch'],
        'Shichuan': ['Sichuan'],
        'France': ['Frence'],
        'al-Qaida': ['Al-Qaida'],
        'Gorazde': ['Gerlaridy'],
        'Sun': ['Solar'],
        'China': ['Sino'],
        'America': ['US', 'U.S.'],
        'U.S.': ['US'],
        'Georgia': ['GA']
    }

    unknown_entity_type = 'ENTITY'

    def __init__(self, span=None, node=None, ner_type=None, amr_type=None, confidence=0):
        self.span = span
        self.node = node
        self.ner_type = ner_type
        self.amr_type = amr_type
        self.confidence = confidence


def strip_lemma(lemma):
    # Remove twitter '@'.
    if lemma[0] == '@':
        lemma = lemma[1:]
    # Remove '-$' suffix.
    if lemma[-1] == '$':
        lemma = lemma[:-1]
    return lemma


def resolve_conflict_entities(entities):
    # If there's overlap between any two entities,
    # remove the one that has lower confidence.
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


def rephrase_ops(ops):
    ret = []
    joined_ops = ' '.join(map(str, ops))
    if joined_ops == '"United" "States"':
        ret.append('"America"')
    elif joined_ops == '"World" "War" "II"':
        ret.append('"WWII"')
    return ret


class Recategorizer:
    # TODO:
    #   1. Decide how to choose type for abstract named entities.
    #   2. Decide whether to further collapse the name node.
    #   3. Check the mismatch between aligned entities and ops.

    def __init__(self, train_data=None, build_map=False, map_dir=None):
        self.stemmer = nltk.stem.SnowballStemmer('english').stem
        self.train_data = train_data
        self.build_map = build_map
        if build_map:
            # Build two maps from the training data.
            #   name_node_type_map counts the number of times that a type of AMR name node
            #       (e.g., "country", "person") is aligned to a named entity span in the
            #       input sentence that has been tagged with a NER type, i.e.,
            #           count = self.name_node_type_map[name_node_type][ner_type]
            #
            #   entity_map maps a named entity span to a NER type.
            self.name_node_type_map_done = False
            self.name_node_type_map = defaultdict(lambda: defaultdict(int))
            self.entity_map = {}
            self._build_map()
            self._dump_map(map_dir)
        else:
            self._load_map(map_dir)

    def _build_map(self):
        logger.info('Building name_node_type_map...')
        for _ in self.recategorize_file(self.train_data):
            pass
        self.name_node_type_map_done = True
        logger.info('Done.')
        logger.info('Building entity_map...')
        for _ in self.recategorize_file(self.train_data):
            pass
        logger.info('Done.')

    def _dump_map(self, directory):
        with open(os.path.join(directory, 'name_node_type_map.json'), 'w', encoding='utf-8') as f:
            json.dump(self.name_node_type_map, f, indent=4)
        with open(os.path.join(directory, 'entity_map.json'), 'w', encoding='utf-8') as f:
            json.dump(self.entity_map, f, indent=4)

    def _load_map(self, directory):
        with open(os.path.join(directory, 'name_node_type_map.json'), encoding='utf-8') as f:
            self.name_node_type_map = json.load(f)
        with open(os.path.join(directory, 'entity_map.json'), 'w', encoding='utf-8') as f:
            self.entity_map = json.load(f)

    def _map_name_node_type(self, name_node_type):
        if not self.build_map and name_node_type in self.name_node_type_map:
            return max(self.name_node_type_map[name_node_type].keys(),
                       key=lambda ner_type: self.name_node_type_map[name_node_type][ner_type])
        else:
            return Entity.unknown_entity_type

    def recategorize_file(self, file_path):
        for amr in AMRIO.read(file_path):
            self.recategorize_graph(amr)
            yield amr

    def recategorize_graph(self, amr):
        input_cleaner.clean(amr)
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
        if not self.build_map:
            self._collapse_name_nodes(entities, amr)
        else:
            self._update_map(entities, amr)

    def _get_aligned_entity(self, node, amr):
        ops = node.ops
        if len(ops) == 0:
            return None
        alignment = self._get_alignment_for_ops(ops, amr)
        if len(alignment) == 0:
            ops = rephrase_ops(ops)
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
        # Exact match.
        if op_lower == token_lower or op_lower == lemma_lower or op_lower == stripped_lemma_lower:
            return 10
        # Stem exact match.
        elif self.stemmer(op) == amr.stems[index]:
            return 8
        # Tagged as named entity and match the first 3 chars.
        elif amr.is_named_entity(index) and (
                op_lower[:3] == token_lower[:3] or
                op_lower[:3] == lemma_lower[:3] or
                op_lower[:3] == stripped_lemma_lower[:3]
        ):
            return 5
        # Match the first 3 chars.
        elif (op_lower[:3] == token_lower[:3] or
              op_lower[:3] == lemma_lower[:3] or
              op_lower[:3] == stripped_lemma_lower[:3]
        ):
            return 1
        # Match after mapping.
        elif op in Entity.entity_map:
            return max(self._maybe_align_op_to(mapped_op, index, amr) for mapped_op in Entity.entity_map[op])
        else:
            return 0

    def _get_entity_for_ops(self, alignment, node, amr):
        spans = group_indexes_to_spans(alignment.keys())
        spans.sort(key=lambda span: len(span), reverse=True)
        amr_type = amr.graph.get_name_node_type(node)
        backup_ner_type = self._map_name_node_type(amr_type)
        candidate_entities = []
        for span in spans:
            confidence = sum(alignment[j][1] for j in span)
            ner_type = backup_ner_type
            for index in span:
                if amr.is_named_entity(index):
                    ner_type = amr.ner_tags[index]
                    break
            entity = Entity(span, node, ner_type, amr_type, confidence)
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
            amr.graph.add_node_attribute(entity.node, 'collapsed-ops', abstract)
            offset += len(entity.span) - 1

    def _update_map(self, entities, amr):
        if self.name_node_type_map_done:
            for entity in entities:
                entity_text = ' '.join(amr.tokens[index] for index in entity.span)
                self.entity_map[entity_text] = entity.ner_type
        else:
            for entity in entities:
                if entity.ner_type != Entity.unknown_entity_type:
                    self.name_node_type_map[entity.amr_type][entity.ner_type] += 1


if __name__ == '__main__':
    import argparse

    from stog.data.dataset_readers.amr_parsing.node_utils import NodeUtilities as NU

    parser = argparse.ArgumentParser('recategorizer.py')
    parser.add_argument('--amr_train', default='data/all_amr/train_amr.txt.features.align')
    parser.add_argument('--amr_dev_files', nargs='+', default=[])
    parser.add_argument('--dump_dir', default='./temp')

    args = parser.parse_args()

    recategorizer = Recategorizer(train_data=args.amr_train, build_map=True, map_dir=args.dump_dir)


    for file_path in args.amr_dev_files:
        with open(file_path + '.recategorized', 'w', encoding='utf-8') as f:
            for amr in recategorizer.recategorize_file(file_path):
                f.write(str(amr) + '\n\n')

