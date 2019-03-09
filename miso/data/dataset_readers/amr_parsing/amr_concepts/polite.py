import re


class Polite:

    lemma_map = {
        'can': 'possible'
    }

    def __init__(self, amr, dry=False):
        self.amr = amr
        self.dry = dry
        self.heads = []
        self.true_positive = 0
        self.false_positive = 0

    def remove_polite(self):
        count = 0
        for node in self.amr.graph.get_nodes():
            for attr, value in node.attributes:
                if attr == 'polite':
                    if not self.dry:
                        self.amr.graph.remove_node_attribute(node, attr, value)
                    count += 1
        return count

    def predict_polite(self):
        for i in range(len(self.amr.tokens)):
            if self.amr.lemmas[i] == 'please':
                if self.amr.lemmas[i + 1: i + 3] == ['take', 'a']:
                    self.heads.append((i, i + 3))
                elif i - 2 >= 0 and self.amr.lemmas[i - 2] == 'can':
                    self.heads.append((i, i - 2))
                else:
                    self.heads.append((i, i + 1))

    def restore_polite(self):
        for polite_index, head_index in self.heads:
            for node in self.amr.graph.get_nodes():
                if self.is_match(head_index, node):
                    self.restore_node_polite(node)

    def restore_node_polite(self, node):
        if self.dry:
            for attr, value in node.attributes:
                if attr == 'polite':
                    self.true_positive += 1
                    break
            else:
                self.false_positive += 1
        else:
            self.amr.graph.add_node_attribute(node, 'polite', '+')

    def is_match(self, index, node):
        instance_lemma = re.sub(r'-\d\d$', '', node.instance)
        lemma = self.amr.lemmas[index]
        lemma = self.lemma_map.get(lemma, lemma)
        if instance_lemma == lemma:
            return True
        return False


if __name__ == '__main__':
    import argparse
    from miso.data.dataset_readers.amr_parsing.io import AMRIO

    parser = argparse.ArgumentParser('polite.py')
    parser.add_argument('--amr_files', nargs='+', default=[])
    args = parser.parse_args()

    total, tp, fp = 0, 0, 0
    for file_path in args.amr_files:
        for i, amr in enumerate(AMRIO.read(file_path), 1):
            if i % 1000 == 0:
                print('Precision: {}/{}\nRecall: {}/{}\n'.format(tp, tp + fp, tp, total))
            p = Polite(amr, True)
            p.i = i
            total += p.remove_polite()
            p.predict_polite()
            p.restore_polite()
            tp += p.true_positive
            fp += p.false_positive
        print('Precision: {}/{}\nRecall: {}/{}\n'.format(tp, tp + fp, tp, total))

