import re
from collections import Counter

from stog.data.dataset_readers.amr_parsing.io import AMRIO


def main(file_path):
    attributes = []
    for amr in AMRIO.read(file_path):
        graph = amr.graph
        for node in graph.get_nodes():
            if node.instance == 'ordinal-entity':
                # for source, target in list(graph._G.in_edges(node)):
                #     attributes.append('-> label: ' + graph._G[source][target]['label'])
                # for source, target in list(graph._G.edges(node)):
                #     if graph._G[source][target]['label'] == 'value':
                #         import pdb; pdb.set_trace()
                for attr, value in node.attributes:
                    attributes.append(value)
            continue
    counter = Counter(attributes)
    print('\n'.join('{}\t{}'.format(k, v) for k, v in counter.most_common(100)))


if __name__ == '__main__':
    import sys
    main(sys.argv[1])