import sys
import argparse

from stog.data.dataset_readers.abstract_meaning_representation import AbstractMeaningRepresentationDatasetReader


parser = argparse.ArgumentParser('extract_amr_token.py')
parser.add_argument('file_path')
parser.add_argument('--show_stats', action='store_true')
args = parser.parse_args()

dataset_reader = AbstractMeaningRepresentationDatasetReader(skip_first_line=False)
dataset_reader.set_evaluation()

num_token_list = []
for instance in dataset_reader._read(args.file_path):
    amr_tokens = [x.text for x in instance.fields["tgt_tokens"].tokens[1:-1]]
    num_token_list.append(len(amr_tokens))
    if not args.show_stats:
        print(' '.join(amr_tokens))

list_length = len(num_token_list)
if args.show_stats:
    num_token_list.sort()
    print('min: {}\n50%: {}\nmax: {}'.format(
        num_token_list[0], num_token_list[int(list_length / 2)], num_token_list[-1]
    ))
    print('>=95%')
    for i in range(95, 100):
        print('  {}%: {}'.format(i, num_token_list[int(list_length * i / 100)]))

    print('#instances (> 50): {}'.format(len([x for x in num_token_list if x > 50])))
