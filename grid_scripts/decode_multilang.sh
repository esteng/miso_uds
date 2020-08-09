#!/bin/bash

cd /home/hltcoe/estengel/miso_research
for lang in af de fr fi hu gl; do
    echo ${lang}
    echo "qsub -v \"CHECKPOINT_DIR=/exp/estengel/miso_res/ud_models_vocab/transformer/${lang}/,TEST_DATA=/exp/estengel/ud_data/all_data/dev\" grid_scripts/decode_decomp_conllu.sh" 
    #qsub -v "CHECKPOINT_DIR=/exp/estengel/miso_res/ud_models_vocab/transformer/${lang}/,TEST_DATA=/exp/estengel/ud_data/all_data/dev" grid_scripts/decode_decomp_conllu.sh 
    echo "qsub -v \"CHECKPOINT_DIR=/exp/estengel/miso_res/ud_models_vocab/transformer_pretrained/${lang}_encoder/,TEST_DATA=/exp/estengel/ud_data/all_data/dev\" grid_scripts/decode_decomp_conllu.sh" 
    #qsub -v "CHECKPOINT_DIR=/exp/estengel/miso_res/ud_models_vocab/transformer_pretrained/${lang}_encoder/,TEST_DATA=/exp/estengel/ud_data/all_data/dev" grid_scripts/decode_decomp_conllu.sh 
    echo "qsub -v \"CHECKPOINT_DIR=/exp/estengel/miso_res/ud_models_vocab/transformer_pretrained/${lang}_intermediate/,TEST_DATA=/exp/estengel/ud_data/all_data/dev\" grid_scripts/decode_decomp_conllu.sh" 
    #qsub -v "CHECKPOINT_DIR=/exp/estengel/miso_res/ud_models_vocab/transformer_pretrained/${lang}_intermediate/,TEST_DATA=/exp/estengel/ud_data/all_data/dev" grid_scripts/decode_decomp_conllu.sh 
done
