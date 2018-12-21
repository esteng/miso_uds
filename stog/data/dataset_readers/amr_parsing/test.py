import re
from collections import Counter

from stog.data.dataset_readers.amr_parsing.io import AMRIO


def main(file_path):
    attributes = []
    for amr in AMRIO.read(file_path):
        graph = amr.graph
        for node in graph.get_nodes():
            if node.instance == 'date-entity':
                # for source, target in list(graph._G.edges(node)):
                #     attributes.append(graph._G[source][target]['label'])
                # continue
                continue
            else:
                for attr, value in node.attributes:
                    if attr not in ('quant',):
                        continue
                    attributes.append(value)
            continue
            if graph.is_name_node(node):
                attributes += [attr for attr, _ in node.attributes]
    counter = Counter(attributes)
    print('\n'.join('{}\t{}'.format(k, v) for k, v in counter.most_common(100)))


if __name__ == '__main__':
    import sys
    main(sys.argv[1])