#!/bin/bash

cd /home/hltcoe/estengel/miso_research

for lang in af de fr fi hu gl; do
    echo "${lang}" 
    python miso/metrics/conllu.py \
        /exp/estengel/ud_data/all_data/dev/${lang}-universal.conllu \
        /exp/estengel/miso_res/ud_models_transformer_fix/transformer_pretrained/${lang}_syntax/dev.conllu
    echo "${lang} ENCODER" 
    python miso/metrics/conllu.py \
        /exp/estengel/ud_data/all_data/dev/${lang}-universal.conllu \
        /exp/estengel/miso_res/ud_models_transformer_fix/transformer_pretrained/${lang}_encoder/dev.conllu
    
    #echo "${lang} INTERMEDIATE" 
    #python miso/metrics/conllu.py \
    #    /exp/estengel/ud_data/all_data/dev/${lang}-universal.conllu \
    #    /exp/estengel/miso_res/ud_models_vocab/transformer_pretrained/${lang}_intermediate/dev.conllu
done
