from miso.data.dataset_readers.amr_parsing.amr import AMRGraph
from miso.utils.string import START_SYMBOL, END_SYMBOL


def dry_load(amr):
    list_data = amr.graph.get_list_data(amr, START_SYMBOL, END_SYMBOL)
    nodes = list_data['tgt_tokens'][1:-1]
    heads = list_data['head_indices']
    head_labels = list_data['head_tags']
    corefs = []
    for i, copy_index in enumerate(list_data['tgt_copy_indices']):
        if copy_index == 0:
            corefs.append(i)
        else:
            corefs.append(copy_index)
    corefs = corefs[1:-1]
    if False: # amr.id.startswith('bolt12_6455_6562.10'):
        print(amr)
        print(nodes)
        print(corefs)
        print(heads)
        print(head_labels)
        import pdb; pdb.set_trace()
    graph = AMRGraph.from_prediction(dict(
        nodes=nodes, heads=heads, head_labels=head_labels, corefs=corefs
    ))
    amr.graph = graph
    return amr


if __name__ == '__main__':
    import argparse
    from miso.data.dataset_readers.amr_parsing.io import AMRIO

    parser = argparse.ArgumentParser('amr_loader_sanity_check.py')
    parser.add_argument('--data_files', nargs='+', help='AMR data files.')

    args = parser.parse_args()

    for i, file_path in enumerate(args.data_files, 1):
        with open('amr_gold_{}.txt'.format(i), 'w', encoding='utf-8') as f1:
            with open('amr_load_{}.txt'.format(i), 'w', encoding='utf-8') as f2:
                for amr in AMRIO.read(file_path):
                    f1.write(str(amr) + '\n\n')
                    amr = dry_load(amr)
                    f2.write(str(amr) + '\n\n')
