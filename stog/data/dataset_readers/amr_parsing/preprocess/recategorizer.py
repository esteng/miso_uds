import os
import re
import json
from collections import defaultdict

import nltk
from nltk.tokenize import word_tokenize
from word2number import w2n

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
        'Georgia': ['GA'],
        'Pennsylvania': ['PA', 'PA.'],
        'Missouri': ['MO', 'MO.'],
        'WWII': ['WW2'],
        'WWI': ['WW1'],
        'Iran': ['Ian'],
        'Jew': ['Semitism', 'Semites'],
        'Islam': ['Muslim'],
        'influenza': ['flu'],
    }

    unknown_entity_type = 'ENTITY'

    def __init__(self, span=None, node=None, ner_type=None, amr_type=None, confidence=0):
        self.span = span
        self.node = node
        self.ner_type = ner_type
        self.amr_type = amr_type
        self.confidence = confidence


class DATE:

    attribute_list = ['year', 'month', 'day', 'decade', 'time', 'century', 'era', 'timezone',
                  'quant', 'value', 'quarter', 'year2']
    month_map = [
        ('January', 'Jan.', 'Jan'),
        ('February', 'Feb.', 'Feb', 'Febuary'),
        ('March', 'Mar.', 'Mar'),
        ('April', 'Apr.', 'Apr', 'Aril'),
        ('May',),
        ('June', 'Jun.', 'Jun'),
        ('July', 'Jul.', 'Jul'),
        ('August', 'Aug.', 'Aug'),
        ('September', 'Sep.', 'Sep', 'Sept.'),
        ('October', 'Oct.', 'Oct'),
        ('November', 'Nov.', 'Nov', 'Novmber'),
        ('December', 'Dec.', 'Dec')
    ]

    def __init__(self, node):
        self.node = node
        self.attributes = {attr: value for attr, value in node.attributes if attr in ('year', 'month', 'day', 'decade', 'time')}
        self.span = None

    def _is_covered(self, alignment):
        attributes = self.attributes.copy()
        for index in self.span:
            if index in alignment:
                for item, _ in alignment[index]:
                    if item[0] in attributes:
                        attributes.pop(item[0])
        return len(attributes) == 0

    def _get_alignment(self, amr):
        alignment = defaultdict(list)
        for item in self.attributes.items():
            attr, value = item
            for i in range(len(amr.tokens)):
                confidence = self._maybe_align(attr, value, i, amr)
                if confidence != 0:
                    alignment[i].append((item, confidence))
        return alignment

    def _get_span(self, alignment, amr):
        spans = group_indexes_to_spans(alignment.keys(), amr)
        span_scores = []
        for span in spans:
            attr_set = set()
            for index in span:
                if index in alignment:
                    for item, _ in alignment[index]:
                        attr_set.add(item[0])
            span_scores.append((span, len(span), len(attr_set)))
        span_scores.sort(key=lambda x: (x[1], x[2]), reverse=True)
        spans = [span for span, _, _ in span_scores]
        # spans = self._clean_span(spans, alignment, amr)
        if len(spans):
            self.span = max(spans, key=lambda span: sum(
                max(alignment[i], key=lambda x: x[1])[1] for i in span if i in alignment))
        if len(spans) == 0 or not self._is_covered(alignment) and 'time' in self.attributes:
            print(self.node)
            import pdb; pdb.set_trace()

    def _clean_span(self, spans, alignment, amr):
        # Make sure each op only appears once in a span.
        clean_spans = []
        for span in spans:
            _align = {}
            trivial_indexes = []
            for index in span:
                if index not in alignment:
                    trivial_indexes.append(index)
                    continue
                for item, confidence in alignment[index]:
                    if item not in _align or _align[item][1] < confidence:
                        _align[item] = (index, confidence)
            indexes = [i for i, _ in _align.values()] + trivial_indexes
            _spans = group_indexes_to_spans(indexes, amr)
            clean_spans.append(max(_spans, key=lambda s: len(s)))
        return clean_spans

    def _maybe_align(self, attr, value, index, amr):
        if attr == 'year':
            return self._maybe_align_year(value, index, amr)
        elif attr == 'month':
            return self._maybe_align_month(value, index, amr)
        elif attr == 'day':
            return self._maybe_align_day(value, index, amr)
        elif attr == 'decade':
            return self._maybe_align_decade(value, index, amr)
        elif attr == 'time':
            return self._maybe_align_basic(value, index, amr)
        else:
            return self._maybe_align_basic(value, index, amr)

    def _maybe_align_basic(self, value, index, amr):
        value = str(value)
        if amr.tokens[index] == value or amr.lemmas[index] == value:
            return 10
        if self._strip_date_lemma(amr.lemmas[index]) == value:
            return 10
        return 0

    def _strip_date_lemma(self, lemma):
        # Remove '-'.
        if len(lemma) and lemma[0] == '-':
            lemma = lemma[1:]
        if len(lemma) and lemma[-1] == '-':
            lemma = lemma[:-1]
        return lemma

    def _maybe_align_year(self, value, index, amr):
        basic_confidence = self._maybe_align_basic(value, index, amr)
        if basic_confidence != 0:
            return basic_confidence
        lemma = amr.lemmas[index]
        stripped_lemma = self._strip_date_lemma(lemma)
        year_short = str(value)[-2:]
        if lemma.startswith(year_short) or stripped_lemma.startswith(year_short):
            return 10
        year_with_s = str(value) + 's'
        if year_with_s == lemma:
            return 10
        year_with_stroke = "'" + year_short
        if year_with_stroke == lemma:
            return 10
        return 0

    def _maybe_align_month(self, value, index, amr):
        lemma = amr.lemmas[index]
        if 0 < value < 13:
            for month in self.month_map[value - 1]:
                if month.lower() == lemma.lower():
                    return 15
        basic_confidence = self._maybe_align_basic(value, index, amr)
        if basic_confidence != 0:
            return basic_confidence
        month_fixed_length = '{:02d}'.format(value)
        if month_fixed_length == lemma:
            return 10
        return 0

    def _maybe_align_day(self, value, index, amr):
        basic_confidence = self._maybe_align_basic(value, index, amr)
        if basic_confidence != 0:
            return basic_confidence
        lemma = amr.lemmas[index]
        day_fixed_length = '{:02d}'.format(value)
        if day_fixed_length == lemma:
            return 10
        day = str(value)
        if (day + 'th' == lemma or
            day + 'st' == lemma or
            day + 'nd' == lemma or
            day + 'rd' == lemma or
            day + 'sr' == lemma
        ):
            return 10
        if value == 1 and lemma == 'first':
            return 8
        if value == 2 and lemma == 'second':
            return 8
        if value == 3 and lemma == 'third':
            return 8
        return 0

    def _maybe_align_decade(self, value, index, amr):
        basic_confidence = self._maybe_align_basic(value, index, amr)
        if basic_confidence != 0:
            return basic_confidence
        year_confidence = self._maybe_align_year(value, index, amr)
        if year_confidence != 0:
            return year_confidence
        try:
            w2n_number = str(w2n.word_to_num(amr.lemmas[index]))
        except:
            w2n_number = None
        decade_short = str(value)[-2:]
        lemma = amr.lemmas[index]
        if lemma.endswith(decade_short + 's'):
            return 10
        if decade_short == w2n_number:
            return 10
        return 0


def strip_lemma(lemma):
    # Remove twitter '@'.
    if len(lemma) and lemma[0] == '@':
        lemma = lemma[1:]
    # Remove '-$' suffix.
    if len(lemma) and lemma[-1] == '$':
        lemma = lemma[:-1]
    return lemma


def resolve_conflict_entities(entities):
    # If there's overlap between any two entities,
    # remove the one that has lower confidence.
    index_entity_map = {}
    empty_entities = []
    for entity in entities:
        if len(entity.span) == 0:
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
    return list(node_entity_map.values()) + empty_entities


def group_indexes_to_spans(indexes, amr):
    indexes = list(indexes)
    indexes.sort()
    spans = []
    last_index = None
    for idx in indexes:
        if last_index is None or idx - last_index > 2:
            spans.append([])
        elif idx - last_index == 2:
            if re.search(r"(,|'s|of|'|-|in)", amr.tokens[idx - 1]):
                spans[-1].append(idx - 1)
            else:
                spans.append([])
        last_index = idx
        spans[-1].append(idx)
    return spans


def tokenize_ops(ops):
    ret = []
    for op in ops:
        if not isinstance(op, str):
            ret += [op]
            continue
        if re.search(r'^".*"$', op):
            op = op[1:-1]
        ret += re.split(r"(-|'s|n't|')", op)
    return ret


def rephrase_ops(ops):
    ret = []
    joined_ops = ' '.join(map(str, ops))
    if joined_ops == '"United" "States"':
        ret.append('"America"')
    elif joined_ops == '"World" "War" "II"':
        ret.append('"WWII"')
    elif joined_ops == '"Republican" "National" "Convention"':
        ret.append('"RNC"')
    elif joined_ops == '"Grand" "Old" "Party"':
        ret.append('"GOP"')
    elif joined_ops == '"United" "Nations"':
        ret.append('"U.N."')

    return ret


class Recategorizer:
    # TODO:
    #   1. Recategorize date-entity and other '*-entity'.
    #   2. Check the mismatch between aligned entities and ops.

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
        self.named_entity_count = 0
        self.recat_named_entity_count = 0
        self.date_entity_count = 0
        self.recat_date_entity_count = 0
        self.removed_wiki_count = 0

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
        self.remove_wiki(amr)
        # self.recategorize_name_nodes(amr)
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
                entity = self._get_aligned_entity(node, amr)
                if len(entity.span):
                    self.recat_named_entity_count += 1
                entities.append(entity)
        entities = resolve_conflict_entities(entities)
        if not self.build_map:
            self._collapse_name_nodes(entities, amr)
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
        # dates = resolve_conflict_entities(dates)
        # self._collapse_date_nodes(dates, amr)

    def _get_aligned_entity(self, node, amr):
        ops = node.ops
        if len(ops) == 0:
            return None
        rephrased_ops = rephrase_ops(ops)
        alignment = self._get_alignment_for_ops(rephrased_ops, amr)
        if len(alignment) == 0:
            alignment = self._get_alignment_for_ops(ops, amr)
        entity = self._get_entity_for_ops(alignment, node, amr)
        # Try the tokenized version.
        ops = tokenize_ops(ops)
        alignment = self._get_alignment_for_ops(ops, amr)
        _entity = self._get_entity_for_ops(alignment, node, amr)
        if _entity.confidence > entity.confidence:
            entity = _entity
        if entity.confidence == 0:
            if len(entity.span):
                print(' '.join(amr.tokens[i] for i in entity.span), end='')
                print(' --> ' + ' '.join(map(str, entity.node.ops)))
            # print(node)
            # import pdb; pdb.set_trace()
        # else:
        #     print(' '.join(amr.tokens[i] for i in entity.span), end='')
        #     print(' --> ' + ' '.join(map(str, entity.node.ops)))
        return entity

    def _get_aligned_date(self, node, amr):
        date = DATE(node)
        if len(date.attributes) == 0:
            return date
        alignment = date._get_alignment(amr)
        date._get_span(alignment, amr)
        if date.span is None:
            print(date.node)
            import pdb; pdb.set_trace()
        else:
            print(' '.join([amr.tokens[index] for index in date.span]))
            print(' --> ' + str(node) + '\n')
        return date

    def _get_alignment_for_ops(self, ops, amr):
        alignment = {}
        for i, op in enumerate(ops):
            for j, token in enumerate(amr.tokens):
                confidence = self._maybe_align_op_to(op, j, amr)
                if confidence > 0:
                    if j not in alignment or (j in alignment and alignment[j][1] < confidence):
                        alignment[j] = (i, confidence)
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
        if amr.tokens[index] == op or amr.lemmas[index] == op:
            return 15
        elif op_lower == token_lower or op_lower == lemma_lower or op_lower == stripped_lemma_lower:
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
        spans = group_indexes_to_spans(alignment.keys(), amr)
        spans.sort(key=lambda span: len(span), reverse=True)
        spans = self._clean_span(spans, alignment, amr)
        amr_type = amr.graph.get_name_node_type(node)
        backup_ner_type = self._map_name_node_type(amr_type)
        candidate_entities = []
        for span in spans:
            confidence = sum(alignment[j][1] for j in span if j in alignment)
            ner_type = backup_ner_type
            for index in span:
                if amr.is_named_entity(index):
                    ner_type = amr.ner_tags[index]
                    break
            entity = Entity(span, node, ner_type, amr_type, confidence)
            candidate_entities.append(entity)

        if len(candidate_entities):
            return max(candidate_entities, key=lambda entity: entity.confidence)
        return Entity([], node, backup_ner_type, amr_type, 0)

    def _clean_span(self, spans, alignment, amr):
        # Make sure each op only appears once in a span.
        clean_spans = []
        for span in spans:
            _align = {}
            trivial_indexes = []
            for index in span:
                if index not in alignment:
                    trivial_indexes.append(index)
                    continue
                op, confidence = alignment[index]
                if op not in _align or (op in _align and _align[op][1] < confidence):
                    _align[op] = (index, confidence)
            indexes = [i for i, _ in _align.values()] + trivial_indexes
            _spans = group_indexes_to_spans(indexes, amr)
            clean_spans.append(max(_spans, key=lambda s: len(s)))
        return clean_spans

    def _collapse_name_nodes(self, entities, amr):
        amr.abstract_map = {}
        if len(entities) == 0:
            return
        type_counter = defaultdict(int)
        entities.sort(key=lambda entity: entity.span[-1] if len(entity.span) else float('inf'))
        offset = 0
        for entity in entities:
            if len(entity.span) > 0:
                type_counter[entity.ner_type] += 1
                abstract = '{}_{}'.format(
                    entity.ner_type, type_counter[entity.ner_type])
                span_with_offset = [index - offset for index in entity.span]
                amr.abstract_map[abstract] = ' '.join(map(amr.tokens.__getitem__, span_with_offset))
                amr.replace_span(span_with_offset, [abstract], ['NNP'], [entity.ner_type])
                amr.graph.remove_node_ops(entity.node)
                amr.graph.replace_node_attribute(
                    entity.node, 'instance', entity.node.instance, abstract)
                offset += len(entity.span) - 1
            else:
                amr.graph.remove_node(entity.node)

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

    from stog.data.dataset_readers.amr_parsing.node_utils import NodeUtilities as NU

    parser = argparse.ArgumentParser('recategorizer.py')
    parser.add_argument('--amr_train', default='data/all_amr/train_amr.txt.features.align')
    parser.add_argument('--amr_dev_files', nargs='+', default=[])
    parser.add_argument('--dump_dir', default='./temp')
    parser.add_argument('--build_map', action='store_true')

    args = parser.parse_args()

    recategorizer = Recategorizer(train_data=args.amr_train, build_map=args.build_map, map_dir=args.dump_dir)


    for file_path in args.amr_dev_files:
        with open(file_path + '.recategorized', 'w', encoding='utf-8') as f:
            for amr in recategorizer.recategorize_file(file_path):
                f.write(str(amr) + '\n\n')

