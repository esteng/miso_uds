#!/bin/bash

cd /home/hltcoe/estengel/miso_research
for lang in af de fr fi hu gl hy kk; do
#for lang in  de; do
    echo ${lang}
    echo "qsub -v \"CHECKPOINT_DIR=/exp/estengel/miso_res/ud_models_tuned/transformer_pretrained/${lang}_syntax/,TEST_DATA=/exp/estengel/ud_data/all_data/test\" grid_scripts/decode_decomp_multiling.sh" 
    qsub -v "CHECKPOINT_DIR=/exp/estengel/miso_res/ud_models_tuned/transformer_pretrained/${lang}_syntax/,TEST_DATA=/exp/estengel/ud_data/all_data/test" grid_scripts/decode_decomp_multiling.sh
    echo "qsub -v \"CHECKPOINT_DIR=/exp/estengel/miso_res/ud_models_tuned/transformer_pretrained/${lang}_encoder/,TEST_DATA=/exp/estengel/ud_data/all_data/test\" grid_scripts/decode_decomp_multiling.sh" 
    qsub -v "CHECKPOINT_DIR=/exp/estengel/miso_res/ud_models_tuned/transformer_pretrained/${lang}_encoder/,TEST_DATA=/exp/estengel/ud_data/all_data/test" grid_scripts/decode_decomp_multiling.sh
    echo "qsub -v \"CHECKPOINT_DIR=/exp/estengel/miso_res/ud_models_tuned/transformer_pretrained/${lang}_intermediate/,TEST_DATA=/exp/estengel/ud_data/all_data/test\" grid_scripts/decode_decomp_multiling.sh" 
    qsub -v "CHECKPOINT_DIR=/exp/estengel/miso_res/ud_models_tuned/transformer_pretrained/${lang}_intermediate/,TEST_DATA=/exp/estengel/ud_data/all_data/test" grid_scripts/decode_decomp_multiling.sh 
done
