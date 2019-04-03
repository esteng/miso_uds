"""
Script to compare predictedand golden tokens
"""
import sys
import argparse

from stog.data.dataset_readers.semantic_dependency_parsing import \
    SemanticDependenciesDatasetReader

from collections import defaultdict, Counter
from stog.utils.exception_hook import ExceptionHook

sys.excepthook = ExceptionHook()

parser = argparse.ArgumentParser('compare-tgt-tokens-sdp.py')
parser.add_argument('--pred-tokens')
parser.add_argument('--gold')
parser.add_argument('--show_stats', action='store_true')
args = parser.parse_args()

dataset_reader = SemanticDependenciesDatasetReader()
dataset_reader.set_analysis()

instance_list = defaultdict(list)

with open(args.pred_tokens) as f:
    for line in f:
        instance_list["pred"].append(
            line.split()
        )

for instance in dataset_reader._read(args.gold):
    instance_list["gold"].append(instance)

assert len(instance_list["pred"]) == len(instance_list["gold"])

p = 0
g = 0
c = 0
precision = 0
recall = 0
num_missing_edges = 0
missing_token_types = Counter()
missing_edge_types = Counter()

for i in range(len(instance_list["pred"])):
    pred_tgt_token = instance_list["pred"][i]
    gold_tgt_token = [x.text for x in instance_list["gold"][i].fields["tgt_tokens"].tokens[1:-1]]
    pred_tgt_token_set = set(pred_tgt_token)
    gold_tgt_token_set = set(gold_tgt_token)
    p += len(pred_tgt_token_set)
    g += len(gold_tgt_token_set)
    c += len(pred_tgt_token_set & gold_tgt_token_set)
    #print(pred_tgt_token)
    #print(gold_tgt_token)
    #print(
    #    len(pred_tgt_token_set),
    #    len(gold_tgt_token_set),
    #    len(pred_tgt_token_set & gold_tgt_token_set)
    #)
    if len(gold_tgt_token_set - pred_tgt_token_set & gold_tgt_token_set) > 0:
        gold_src_token = [x.text for x in instance_list["gold"][i].fields["src_tokens"].tokens]
        missing_edges = set()
        for token in list(gold_tgt_token_set - pred_tgt_token_set & gold_tgt_token_set):
            if token in gold_src_token:
                token_index = gold_src_token.index(token)
                missing_edges |= {x for x in instance_list["gold"][i].fields['arc_indices'].metadata if token_index in x}
                _missing_pos = instance_list["gold"][i].fields["annotated_sentence"].metadata[token_index]["pos"]
                missing_token_types[_missing_pos] += 1
        num_missing_edges += len(missing_edges)
        continue 

        print(pred_tgt_token)
        print(gold_tgt_token)
        print(
            len(pred_tgt_token_set),
            len(gold_tgt_token_set),
            len(pred_tgt_token_set & gold_tgt_token_set)
        )
        print(gold_tgt_token_set - pred_tgt_token_set & gold_tgt_token_set)

print("Precision: {}, Recall: {}".format(c / p, c / g))
print("Num missing edges: {}".format(num_missing_edges))
print("Num misssing token: {}".format(sum(missing_token_types.values())))
for pos, num in missing_token_types.most_common():
    print("\t{}: {}".format(pos, num))
