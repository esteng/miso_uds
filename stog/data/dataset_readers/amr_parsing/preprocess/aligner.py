import re

import nltk

from stog.data.dataset_readers.amr_parsing.io import AMRIO
from stog.utils import logging


logger = logging.init_logger()


class Aligner:
    """
    This aligner **only** aligns instances of AMR nodes to lemmas of the input sentence.
    Other attributes' alignment will be done in the recategorization stage.
    """

    def __init__(self, node_utils):
        self.node_utils = node_utils
        self.stemmer = nltk.stem.SnowballStemmer('english').stem

        self.aligned_instance_count = 0
        self.amr_instance_count = 0
        self.restore_count = 0
        self.no_aligned_instances = set()

    def align_file(self, file_path):
        for i, amr in enumerate(AMRIO.read(file_path), 1):
            if i % 1000 == 0:
                logger.info('Processed {} examples.'.format(i))
                self.print_statistics()
            self.align_graph(amr)
            yield amr
        self.print_statistics()

    def align_graph(self, amr):
        graph = amr.graph
        for node in graph.get_nodes():
            instance = node.instance
            lemmas = self.map_instance_to_lemmas(instance)
            lemma = self.find_corresponding_lemma(instance, lemmas, amr)
            self.update_graph(graph, node, instance, lemma)

    def map_instance_to_lemmas(self, instance):
        """
        Get the candidate lemmas which can be used to represent the instance.
        """
        # Make sure it's a string and not quoted.
        assert isinstance(instance, str) and not re.search(r'^".*"$', instance)
        if re.search(r'-\d\d$', instance):  # frame
            lemmas = self.node_utils.get_lemmas(instance)
        else:
            lemmas = [instance]
        return lemmas

    def find_corresponding_lemma(self, instance, lemmas, amr):
        # TODO: Add more align rules.
        # amr_lemma is case-sensitive, so try casing it in different ways: Aaa, AAA, aaa.
        self.amr_instance_count += 1
        aligned_lemma = None
        stems = [self.stemmer(l) for l in amr.lemmas]
        for lemma in lemmas:
            if lemma in amr.lemmas:
                aligned_lemma = lemma
                break
            lemma_stem = self.stemmer(lemma)
            if lemma_stem in stems:
                amr.lemmas[stems.index(lemma_stem)] = lemma
                aligned_lemma = lemma
                break

        # Make sure it can be correctly restored.
        if aligned_lemma is not None:
            restored_frame = self.node_utils.get_frames(aligned_lemma)[0]
            if restored_frame != instance:
                aligned_lemma = None

        if aligned_lemma is None:
            self.no_aligned_instances.add(instance)
        else:
            self.aligned_instance_count += 1

        return aligned_lemma

    def update_graph(self, graph, node, old, new):
        if new is not None:
            graph.replace_node_attribute(node, 'instance', old, new)
            self.try_restore(old, new)
        else:
            self.try_restore(old, old)

    def try_restore(self, old, new):
        new = re.sub(r'~e.\d+$', '', new)
        _old = self.node_utils.get_frames(new)[0]
        self.restore_count += int(old == _old)

    def reset_statistics(self):
        self.aligned_instance_count = 0
        self.amr_instance_count = 0
        self.restore_count = 0
        self.no_aligned_instances = set()

    def print_statistics(self):
        logger.info('align rate: {}% ({}/{})'.format(
            self.aligned_instance_count / self.amr_instance_count,
            self.aligned_instance_count, self.amr_instance_count))
        logger.info('restore rate: {}% ({}/{})'.format(
            self.restore_count / self.amr_instance_count,
            self.restore_count, self.amr_instance_count))
        logger.info('size of no align lemma set: {}'.format(len(self.no_aligned_instances)))


if __name__ == '__main__':
    import argparse

    from stog.data.dataset_readers.amr_parsing.node_utils import NodeUtilities as NU

    parser = argparse.ArgumentParser('aligner.py')
    parser.add_argument('--amr_files', nargs='+', required=True)
    parser.add_argument('--util_dir', default='./temp')

    args = parser.parse_args()

    node_utils = NU.from_json(args.util_dir, 5)

    aligner = Aligner(node_utils)

    for file_path in args.amr_files:
        with open(file_path + '.align', 'w', encoding='utf-8') as f:
            for amr in aligner.align_file(file_path):
                f.write(str(amr) + '\n\n')
        aligner.reset_statistics()
