import re

from stog.data.dataset_readers.amr_parsing.io import AMRIO


class NodeRestore:

    def __init__(self, node_utils):
        self.node_utils = node_utils

    def restore_instance(self, amr):
        graph = amr.graph
        for node in graph.get_nodes():
            if graph.is_name_node(node):
                # Add quote to wiki and op attributes.
                for attr, value in node.attributes:
                    if str(value) == '-':
                        new = value
                    elif re.search(r'^(wiki|op\d+|time)$', attr) and not re.search(r'^".*"$', str(value)):
                        new = '"' + str(value) + '"'
                    elif re.search(r'^".*"$', str(value)):
                        new = str(value)
                    elif not isinstance(value, str):
                        new = value
                    else:
                        new = self.node_utils.get_frame(value)
                    graph.replace_node_attribute(node, attr, value, new)
            else:
                for attr, value in node.attributes:
                    if isinstance(value, str):
                        if str(value) == '-':
                            new = value
                        elif re.search(r'^".*"$', value):
                            new = value
                        elif re.search(r'^(wiki|time)$', attr) and not re.search(r'^".*"$', value):
                            new = '"' + value + '"'
                        else:
                            new = self.node_utils.get_frame(value)
                    else:
                        new = value
                    graph.replace_node_attribute(node, attr, value, new)

    def restore_file(self, file_path):
        for amr in AMRIO.read(file_path):
            self.restore_instance(amr)
            yield amr


if __name__ == '__main__':
    import argparse

    from stog.data.dataset_readers.amr_parsing.node_utils import NodeUtilities as NU

    parser = argparse.ArgumentParser('node_restore.py')
    parser.add_argument('--amr_train_files', nargs='+', required=True)
    parser.add_argument('--amr_dev_files', nargs='+', required=True)
    parser.add_argument('--propbank_dir', default='data/misc/propbank-frames-xml-2016-03-08/')
    parser.add_argument('--verbalize_file', default='data/misc/verbalization-list-v1.06.txt')

    args = parser.parse_args()

    node_utils = NU(args.amr_train_files, args.propbank_dir, args.verbalize_file, args.amr_dev_files)

    nr = NodeRestore(node_utils)

    for file_path in args.amr_dev_files:
        with open(file_path + '.restore', 'w', encoding='utf-8') as f:
            for amr in nr.restore_file(file_path):
                f.write(str(amr) + '\n\n')
