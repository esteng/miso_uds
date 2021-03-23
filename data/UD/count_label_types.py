import glob
import os
import csv 
import sys 
import re

all_label_types = set()
for path in glob.glob("/exp/estengel/ud_data/all_data/train/*.conllu"):
    label_types = set() 
    with open(path) as f1:
        contents = f1.read().split("\n\n") 
        for blob in contents:
            blob = blob.split("\n")[2:]
            for line in blob: 
                splitline = re.split("\t", line) 
                try:
                    deprel = splitline[7]
                    label_types |= set([deprel])
                except IndexError:
                    continue
        print(f"{path} has {len(label_types)}") 
        all_label_types |= label_types

print(f"TOTAL {len(all_label_types)}") 
