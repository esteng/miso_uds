#!/bin/bash
set -e
path_to_UD='./UD_English-EWT'
for split in train dev test
do
  ./conllu2json.py < $path_to_UD/en_ewt-ud-${split}.conllu > ./json/UD_English-EWT/${split}.json
done
