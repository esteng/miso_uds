import re
from collections import Counter

from stog.data.dataset_readers.amr_parsing.io import AMRIO


def main(file_path):
    attributes = []
    for amr in AMRIO.read(file_path):
        graph = amr.graph
        for node in graph.get_nodes():
            # if 'polarity' in node.instance:
            #     attributes.append(node.instance)
            # for source, target in graph._G.edges(node):
            #     label = graph._G[source][target]['label']
            #     if 'polarity' in label:
            #         attributes.append(label)
            # for attr, value in node.attributes:
            #     if 'polarity' in attr:
            #         attributes.append(value)

            if re.search(r'(^".*"$|^[^a-zA-Z0-9]+$)', node.instance):
                attributes.append(node.instance)
            for attr, value in node.attributes:
                if not isinstance(value, str):
                    continue
                if re.search(r'^[^a-zA-Z0-9]+$', value):
                    import pdb; pdb.set_trace()
                    attributes.append((attr, value))
    counter = Counter(attributes)
    print('\n'.join('{}\t{}'.format(k, v) for k, v in counter.most_common(100)))


if __name__ == '__main__':
    import sys
    main(sys.argv[1])