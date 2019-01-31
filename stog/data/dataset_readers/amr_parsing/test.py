import re
from collections import Counter, defaultdict

from stog.data.dataset_readers.amr_parsing.io import AMRIO


def main(file_path):
    attributes = []
    for amr in AMRIO.read(file_path):
        graph = amr.graph
        for node in graph.get_nodes():
            for source, target in graph._G.edges(node):
                label = graph._G[source][target]['label']
                if re.search(r'^(ARG|op|snt)', label):
                    continue
                attributes.append(graph._G[source][target]['label'])
            # if re.search(r'(^".*"$|^[^a-zA-Z0-9]+$)', node.instance):
            #     attributes.append(node.instance)
            # for attr, value in node.attributes:
            #     if not isinstance(value, str):
            #         continue
            #     if re.search(r'^[^a-zA-Z0-9]+$', value):
            #         import pdb; pdb.set_trace()
            #         attributes.append((attr, value))
            #

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

            # if re.search(r'^"?[0-9/]+"?$', node.instance):
            #     attributes.append(node.instance)
            # if node.instance.endswith('-quantity'):
            #     attributes.append(node.instance)
            # for attr, value in node.attributes:
            #     if not isinstance(value, str) or re.search(r'^"?[0-9/]+"?$', value):
            #         attributes.append(attr)
    counter = Counter(attributes)
    print('\n'.join('{}\t{}'.format(k, v) for k, v in counter.most_common(100)))
    print(' '.join(k for k, v in counter.most_common()))
    print(len(attributes))


if __name__ == '__main__':
    import sys
    main(sys.argv[1])