import os
import re
from collections import Counter

from stog.data.dataset_readers.amr_parsing.io import AMRIO
from stog.data.dataset_readers.amr_parsing.propbank_reader import PropbankReader
from stog.utils import logging


logger = logging.init_logger()
WORDSENSE_RE = re.compile(r'-\d\d$')
QUOTED_RE = re.compile(r'^".*"$')


def is_sense_string(s):
    return isinstance(s, str) and WORDSENSE_RE.search(s)


def is_quoted_string(s):
    return isinstance(s, str) and QUOTED_RE.search(s)


class NodeUtilities:

    def __init__(self, amr_train_files, propbank_dir, verbalize_file, amr_dev_files):
        self.amr_train_files = amr_train_files
        self.amr_dev_files = amr_dev_files
        self.propbank_reader = PropbankReader(propbank_dir)
        self.verbalize_file = verbalize_file

        self.senseless_nodes = set()
        # counter[lemma][frame]: the co-occurrence of (lemma, frame)
        self.lemma_frame_counter = dict()
        # counter[frame][lemma]: the co-occurrence of (frame, lemma)
        self.frame_lemma_counter = dict()

        self._update_counter_from_train_files()
        self._update_counter_from_propbank()
        self._update_counter_from_verbalization()
        self._update_senseless_nodes()

    def get_lemma(self, frame):
        if frame not in self.frame_lemma_counter:
            return re.sub(WORDSENSE_RE, '', frame)
        else:
            return max(self.frame_lemma_counter[frame].keys(), key=lambda lemma: self.frame_lemma_counter[frame][lemma])

    def get_frame(self, lemma):
        if lemma in self.senseless_nodes or lemma not in self.lemma_frame_counter:
            return lemma
        else:
            return max(self.lemma_frame_counter[lemma].keys(), key=lambda frame: self.lemma_frame_counter[lemma][frame])

    def _update_senseless_nodes(self):
        logger.info('Updating senseless nodes.')
        sense_less_nodes = []
        for amr_file in self.amr_train_files:
            for amr in AMRIO.read(amr_file):
                for node in amr.graph.get_nodes():
                    for attr, value in node.get_senseless_attributes():
                        sense_less_nodes.append(value)
        sense_less_node_counter = Counter(sense_less_nodes)
        num_frequence_sense_nodes = int(len(sense_less_node_counter) * 0.8)
        for node, count in sense_less_node_counter.most_common(num_frequence_sense_nodes):
            if count >= 5:  # hard threshold
                self.senseless_nodes.add(node)

    def _update_counter_from_train_files(self):
        logger.info('Updating counter from train files.')
        for file_path in self.amr_train_files:
            for amr in AMRIO.read(file_path):
                for node in amr.graph.get_nodes():
                    for _, frame in node.get_frame_attributes():
                        frame_lemma = re.sub(WORDSENSE_RE, '', frame)
                        self._update_counter(self.lemma_frame_counter, frame_lemma, frame, 1)
                        self._update_counter(self.frame_lemma_counter, frame, frame_lemma, 1)

    def _update_counter_from_propbank(self, base_freq=1):
        logger.info('Updating counter from Propbank.')
        for lemma, frames in self.propbank_reader.lemma_map.items():
            for frame in frames:
                freq = base_freq
                if lemma == frame.lemma:  # bonus the frame lemma.
                    freq *= 10
                self._update_counter(self.lemma_frame_counter, lemma, frame.frame, freq)
                self._update_counter(self.frame_lemma_counter, frame.frame, lemma, freq)

    def _update_counter_from_verbalization(self):
        logger.info('Updating counter from Verbalization.')
        with open(self.verbalize_file, encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split(' ')
                if len(parts) == 4 and parts[0] in ('VERBALIZE', 'MAYBE-VERBALIZE'):
                    lemma = parts[1]
                    frame = parts[3]
                    frame_lemma = re.sub(WORDSENSE_RE, '', frame)
                    freq = 100 if parts[0] == 'VERBALIZE' else 1  # bonus 'VERBALIZE'
                    if lemma == frame_lemma:  # bonus frame lemma
                        freq *= 10
                    self._update_counter(self.lemma_frame_counter, lemma, frame, freq)
                    self._update_counter(self.frame_lemma_counter, frame, lemma, freq)

    @staticmethod
    def _update_counter(obj, key1, key2, value):
        if key1 not in obj:
            obj[key1] = dict()
        if key2 not in obj[key1]:
            obj[key1][key2] = 0
        obj[key1][key2] += value

    def dump_counter(self, directory):
        with open(os.path.join(directory, 'lemma_map'), 'w', encoding='utf-8') as f:
            for lemma in self.lemma_frame_counter:
                f.write(lemma + ':\n')
                for frame, freq in sorted(self.lemma_frame_counter[lemma].items(), key=lambda x: -x[1]):
                    f.write('\t{}\t{}\n'.format(frame, freq))
                f.write('\n')
        with open(os.path.join(directory, 'node_map'), 'w', encoding='utf-8') as f:
            for frame in self.frame_lemma_counter:
                f.write(frame + ':\n')
                for lemma, freq in sorted(self.frame_lemma_counter[frame].items(), key=lambda x: -x[1]):
                    f.write('\t{}\t{}\n'.format(lemma, freq))
                f.write('\n')


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser('node_utils.py')
    parser.add_argument('--amr_files', nargs='+', required=True)
    parser.add_argument('--amr_dev_files', nargs='+', required=True)
    parser.add_argument('--propbank_dir', required=True)
    parser.add_argument('--verbalize_file', required=True)
    parser.add_argument('--output_dir', required=True)

    args = parser.parse_args()

    nu = NodeUtilities(args.amr_files, args.propbank_dir, args.verbalize_file, args.amr_dev_files)
