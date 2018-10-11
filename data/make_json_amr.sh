#!/bin/bash
set -e
path_to_AMR='./AMR_Little_Prince'
for split in train dev test
do
  ./amr2json.py < $path_to_AMR/amr-bank-struct-v1.6-${split}.txt > ./json/AMR_Little-Prince/${split}.json
done
