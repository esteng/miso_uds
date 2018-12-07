import re

from stog.data.dataset_readers.amr_parsing.io import AMRIO


class Aligner:

    def __init__(self, node_utils):
        self.node_utils = node_utils
        self.reset_statistics()

    def align_instance(self, amr):
        graph = amr.graph
        for node in graph.get_nodes():
            for attr, value in node.attributes:
                amr_lemma = self.get_amr_lemma(value)
                lemma = self.align_amr_lemma(amr_lemma, amr)
                self.update_graph(graph, node, attr, value, lemma)

    def align_file(self, file_path):
        for amr in AMRIO.read(file_path):
            self.align_instance(amr)
            yield amr

    def get_amr_lemma(self, value):
        # TODO: add more rules.
        # Get AMR lemma.
        if isinstance(value, str):
            if re.search(r'^".*"$', value):  # literal constant
                amr_lemma = value
            elif re.search(r'-\d\d$', value):  # frame
                amr_lemma = self.node_utils.get_lemma(value)
            else:
                amr_lemma = value
        else:
            # int, float
            amr_lemma = value
        return amr_lemma

    def align_amr_lemma(self, amr_lemma, amr):
        # TODO: Add more align rules.
        # amr_lemma is case-sensitive, so try casing it in different ways: Aaa, AAA, aaa.
        self.amr_lemma_count += 1
        if str(amr_lemma) in amr.lemmas:
            self.align_count += 1
        elif re.search(r'^".*"$', str(amr_lemma)) and amr_lemma[1:-1] in amr.lemmas:  # literal constant
            self.align_count += 1
        else:
            self.no_align_lemma_set.add(amr_lemma)
        return amr_lemma

    def update_graph(self, graph, node, attr, old, new):
        graph.replace_node_attribute(node, attr, old, new)
        self.try_restore(old, new)

    def try_restore(self, old, new):
        if not isinstance(old, str):
            if isinstance(old, int):
                _old = int(new)
            else:
                _old = float(new)
        else:
            _old = self.node_utils.get_frame(new)
        self.restore_count += int(old == _old)

    def reset_statistics(self):
        self.align_count = 0
        self.amr_lemma_count = 0
        self.restore_count = 0
        self.no_align_lemma_set = set()

    def print_statistics(self):
        print('align rate: {}% ({}/{})'.format(
            self.align_count / self.amr_lemma_count, self.align_count, self.amr_lemma_count))
        print('restore rate: {}% ({}/{})'.format(
            self.restore_count / self.amr_lemma_count, self.restore_count, self.amr_lemma_count))
        print('size of no align lemma set: {}'.format(len(self.no_align_lemma_set)))


if __name__ == '__main__':
    import argparse

    from stog.data.dataset_readers.amr_parsing.node_utils import NodeUtilities as NU

    parser = argparse.ArgumentParser('aligner.py')
    parser.add_argument('--amr_train_files', nargs='+')
    parser.add_argument('--amr_dev_files', nargs='+', required=True)
    parser.add_argument('--json_dir', default='./temp')

    args = parser.parse_args()

    node_utils = NU.from_json(args.json_dir)

    aligner = Aligner(node_utils)

    if len(args.amr_train_files) != 0:
        for file_path in args.amr_train_files:
            with open(file_path + '.align', 'w', encoding='utf-8') as f:
                for amr in aligner.align_file(file_path):
                    f.write(str(amr) + '\n\n')

        aligner.print_statistics()
        aligner.reset_statistics()

    for file_path in args.amr_dev_files:
        with open(file_path + '.align', 'w', encoding='utf-8') as f:
            for amr in aligner.align_file(file_path):
                f.write(str(amr) + '\n\n')

    aligner.print_statistics()
