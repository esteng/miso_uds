import re
from collections import Counter, defaultdict

from stog.data.dataset_readers.amr_parsing.io import AMRIO


def main(file_path):
    attributes = []
    edge_labels = defaultdict(int)
    for amr in AMRIO.read(file_path):
        graph = amr.graph
        for node in graph.get_nodes():
            # if re.search(r'(^".*"$|^[^a-zA-Z0-9]+$)', node.instance):
            #     attributes.append(node.instance)
            # for attr, value in node.attributes:
            #     if not isinstance(value, str):
            #         continue
            #     if re.search(r'^[^a-zA-Z0-9]+$', value):
            #         import pdb; pdb.set_trace()
            #         attributes.append((attr, value))
            #
            for source, target in graph._G.edges(node):
                label = graph._G[source][target]['label']
                edge_labels[label] = 0

            for attr, value in node.attributes:
                if attr not in edge_labels:
                    edge_labels[attr] = 1


            # for source, target in graph._G.edges(node):
            #     label = graph._G[source][target]['label']
            #     if label.startswith('ARG'):
            #         continue
            #     if re.search('.*\d.*', label):
            #         attributes.append(label)
            # for attr, value in node.attributes:
            #     if attr.startswith('ARG'):
            #         continue
            #     if re.search('.*\d.*', attr):
            #         attributes.append(attr)
    print('\n'.join(l for l in edge_labels if edge_labels[l] == 1))
    counter = Counter(attributes)
    print('\n'.join('{}\t{}'.format(k, v) for k, v in counter.most_common(100)))


if __name__ == '__main__':
    import sys
    main(sys.argv[1])