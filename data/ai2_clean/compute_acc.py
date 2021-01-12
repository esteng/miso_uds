import sys 
import json 
from conllu import parse 

import pdb 

def get_grand_head(block, idx):
    head = block[idx]['head'] - 1
    grandhead = block[head]['head'] - 1
    grandlabel = block[head]['deprel']
    return grandhead, grandlabel
    

def compute(adp_data, pred_block, gold_block):
    uas, las, total = 0, 0, 0
    for idx, grand_idx in adp_data:
        #assert(pred_block[idx]['upostag'] == 'ADP') 
        assert(gold_block[idx]['upostag'] == 'ADP') 
        pred_grandhead, pred_grandlabel = get_grand_head(pred_block, idx) 
        gold_grandhead, gold_grandlabel = get_grand_head(gold_block, idx) 
        try:
            assert(gold_grandhead == grand_idx) 
        except AssertionError:
            #pdb.set_trace() 
            pass
        if pred_grandhead == gold_grandhead: 
            uas += 1

            # obl edge label introduced in UDv2, nmod used to subsume in v1
            if (pred_grandlabel == gold_grandlabel) or \
                (pred_grandlabel == "nmod" and gold_grandlabel == "obl"):
                las += 1

            else:
                print(f"pred {pred_grandlabel} true {gold_grandlabel}") 
        total += 1
    return uas, las, total 

def parse_conllu(path):
    with open(path) as f1:
        data = f1.read()

    return parse(data) 

def parse_json(path):
    with open(path) as f1:
        data = json.load(f1) 
    return data 

if __name__ == "__main__": 
    pred_conllu = sys.argv[1]
    gold_conllu = sys.argv[2]
    json_file = sys.argv[3]

    pred_blocks = parse_conllu(pred_conllu) 
    gold_blocks = parse_conllu(gold_conllu) 
    json_data = parse_json(json_file) 

    total_uas, total_las, total_total  = 0, 0, 0
    for i, (pred_block, gold_block) in enumerate(zip(pred_blocks, gold_blocks)):
        uas, las, total = compute(json_data[str(i)], pred_block, gold_block) 
        total_uas += uas
        total_las += las 
        total_total += total

    uas_final = (total_uas/ total_total)*100
    las_final = (total_las/ total_total)*100
    print(f"UAS {uas_final:.2f} LAS {las_final:.2f}") 

        



