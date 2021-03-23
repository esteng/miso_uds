#!/bin/bash

cd /home/hltcoe/estengel/miso_research

for lang in af de fr fi hu gl hy kk; do
    echo "${lang}" 
    python miso/metrics/conllu.py \
        /exp/estengel/ud_data/all_data/test/${lang}-universal.conllu \
        /exp/estengel/miso_res/ud_models_tuned/transformer_pretrained/${lang}_syntax/test.conllu
    echo "${lang} ENCODER" 
    python miso/metrics/conllu.py \
        /exp/estengel/ud_data/all_data/test/${lang}-universal.conllu \
        /exp/estengel/miso_res/ud_models_tuned/transformer_pretrained/${lang}_encoder/test.conllu
    
    echo "${lang} INTERMEDIATE" 
    python miso/metrics/conllu.py \
        /exp/estengel/ud_data/all_data/test/${lang}-universal.conllu \
        /exp/estengel/miso_res/ud_models_tuned/transformer_pretrained/${lang}_intermediate/test.conllu
done
