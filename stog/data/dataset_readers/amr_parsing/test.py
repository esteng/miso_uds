import re
from collections import Counter

from stog.data.dataset_readers.amr_parsing.io import AMRIO


def main(file_path):
    attributes = []
    for amr in AMRIO.read(file_path):
        graph = amr.graph
        for node in graph.get_nodes():
            if node.instance.endswith('entity'):
                attributes.append(node.instance)
            if node.instance == 'score-entity':
                for attr, value in node.attributes:
                    if attr.startswith('op') or attr == 'instance':
                        attributes.append(attr)
                    else:
                        import pdb; pdb.set_trace()
                for source, target in list(graph._G.edges(node)):
                    label = graph._G[source][target]['label']
                    if not label.startswith('op'):
                        import pdb; pdb.set_trace()
                    if len(list(graph._G.edges(target))) != 0:
                        import pdb; pdb.set_trace()
                    attributes.append(label)
            continue
    counter = Counter(attributes)
    print('\n'.join('{}\t{}'.format(k, v) for k, v in counter.most_common(100)))


if __name__ == '__main__':
    import sys
    main(sys.argv[1])