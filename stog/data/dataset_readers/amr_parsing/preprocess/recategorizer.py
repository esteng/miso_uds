import os
import re
import json
from collections import defaultdict

import nltk
from nltk.tokenize import word_tokenize
from word2number import w2n

from stog.data.dataset_readers.amr_parsing.io import AMRIO
from stog.data.dataset_readers.amr_parsing.entity import Entity
from stog.data.dataset_readers.amr_parsing.date import DATE
from stog.data.dataset_readers.amr_parsing.preprocess import input_cleaner
from stog.utils import logging


logger = logging.init_logger()


def resolve_conflict_entities(entities):
    # If there's overlap between any two entities,
    # remove the one that has lower confidence.
    index_entity_map = {}
    empty_entities = []
    for entity in entities:
        if not entity.span:
            empty_entities.append(entity)
            continue

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

    removed_entities = []
    for entity in entities:
        if not entity.span:
            continue
        if entity.node in node_entity_map:
            continue
        removed_entities.append(entity)

    return list(node_entity_map.values()) + empty_entities, removed_entities


class Recategorizer:
    # TODO:
    #   1. Recategorize date-entity and other '*-entity'.
    #   2. Check the mismatch between aligned entities and ops.

    def __init__(self, train_data=None, build_map=False, map_dir=None, debug=False):
        self.stemmer = nltk.stem.SnowballStemmer('english').stem
        self.train_data = train_data
        self.build_map = build_map
        self.debug = debug
        self.named_entity_count = 0
        self.recat_named_entity_count = 0
        self.date_entity_count = 0
        self.recat_date_entity_count = 0
        self.removed_wiki_count = 0
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

    def _print_statistics(self):
        if self.named_entity_count != 0:
            logger.info('Named entity collapse rate: {} ({}/{})'.format(
                self.recat_named_entity_count / self.named_entity_count,
                self.recat_named_entity_count, self.named_entity_count))
        if self.date_entity_count != 0:
            logger.info('Dated entity collapse rate: {} ({}/{})'.format(
                self.recat_date_entity_count / self.date_entity_count,
                self.recat_date_entity_count, self.date_entity_count))
        logger.info('Removed {} wikis.'.format(self.removed_wiki_count))

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
        with open(os.path.join(directory, 'entity_map.json'), encoding='utf-8') as f:
            self.entity_map = json.load(f)

    def _map_name_node_type(self, name_node_type):
        if not self.build_map and name_node_type in self.name_node_type_map:
            return max(self.name_node_type_map[name_node_type].keys(),
                       key=lambda ner_type: self.name_node_type_map[name_node_type][ner_type])
        else:
            return Entity.unknown_entity_type

    def recategorize_file(self, file_path):
        for i, amr in enumerate(AMRIO.read(file_path), 1):
            self.recategorize_graph(amr)
            yield amr
            if i % 1000 == 0:
                logger.info('Processed {} examples.'.format(i))
                self._print_statistics()
        logger.info('Done.')
        self._print_statistics()

    def recategorize_graph(self, amr):
        input_cleaner.clean(amr)
        amr.stems = [self.stemmer(l) for l in amr.lemmas]
        self.recategorize_name_nodes(amr)
        if self.build_map:
            return
        self.remove_wiki(amr)
        self.recategorize_date_nodes(amr)

    def remove_wiki(self, amr):
        graph = amr.graph
        for node in graph.get_nodes():
            for attr, value in node.attributes.copy():
                if attr == 'wiki':
                    self.removed_wiki_count += 1
                    graph.remove_node_attribute(node, attr, value)

    def recategorize_name_nodes(self, amr):
        graph = amr.graph
        entities = []
        for node in graph.get_nodes():
            if graph.is_name_node(node):
                self.named_entity_count += 1
                amr_type = amr.graph.get_name_node_type(node)
                backup_ner_type = self._map_name_node_type(amr_type)
                entity = Entity.get_aligned_entity(node, amr, backup_ner_type)
                if len(entity.span):
                    self.recat_named_entity_count += 1
                entities.append(entity)
        entities, removed_entities = resolve_conflict_entities(entities)
        if not self.build_map:
            type_counter = Entity.collapse_name_nodes(entities, amr)
            for entity in removed_entities:
                amr_type = amr.graph.get_name_node_type(entity.node)
                backup_ner_type = self._map_name_node_type(amr_type)
                entity = Entity.get_aligned_entity(entity.node, amr, backup_ner_type)
                Entity.collapse_name_nodes([entity], amr, type_counter)
        else:
            self._update_map(entities, amr)

    def recategorize_date_nodes(self, amr):
        graph = amr.graph
        dates = []
        for node in graph.get_nodes():
            if graph.is_date_node(node) and len(node.attributes) > 1:
                self.date_entity_count += 1
                date = self._get_aligned_date(node, amr)
                if date.span is not None:
                    self.recat_date_entity_count += 1
                dates.append(date)
        dates, removed_dates = resolve_conflict_entities(dates)
        DATE.collapse_date_nodes(dates, amr)

    def _get_aligned_date(self, node, amr):
        date = DATE(node)
        if len(date.attributes) == 0:
            return date
        alignment = date._get_alignment(amr)
        date._get_span(alignment, amr)
        if self.debug:
            if date.span is None:
                print(date.node)
                import pdb; pdb.set_trace()
            else:
                print(' '.join([amr.tokens[index] for index in date.span]))
                print(' --> ' + str(node) + '\n')
        return date

    def _update_map(self, entities, amr):
        if self.name_node_type_map_done:
            for entity in entities:
                if len(entity.span) == 0:
                    continue
                entity_text = ' '.join(amr.tokens[index] for index in entity.span)
                self.entity_map[entity_text] = entity.ner_type
        else:
            for entity in entities:
                if len(entity.span) == 0:
                    continue
                if entity.ner_type != Entity.unknown_entity_type:
                    self.name_node_type_map[entity.amr_type][entity.ner_type] += 1


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser('recategorizer.py')
    parser.add_argument('--amr_train_file', default='data/all_amr/train_amr.txt.features.align')
    parser.add_argument('--amr_files', nargs='+', default=[])
    parser.add_argument('--dump_dir', default='./temp')
    parser.add_argument('--build_map', action='store_true')

    args = parser.parse_args()

    recategorizer = Recategorizer(train_data=args.amr_train_file, build_map=args.build_map, map_dir=args.dump_dir)

    for file_path in args.amr_files:
        with open(file_path + '.recategorize', 'w', encoding='utf-8') as f:
            for amr in recategorizer.recategorize_file(file_path):
                f.write(str(amr) + '\n\n')

