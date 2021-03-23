#!/bin/bash

cd /home/hltcoe/estengel/miso_research

for lang in hy kk af de fr fi hu gl; do
    ENC_CKPT_DIR="/exp/estengel/miso_res/ud_models_tuned/transformer_pretrained/${lang}_encoder/"
    ENC_CONFIG="miso/training_config/ud_parsing_transformer_fix/transformer_pretrained/${lang}_encoder.jsonnet"

    #echo "qsub -v \"CHECKPOINT_DIR=${ENC_CKPT_DIR},TRAINING_CONFIG=${ENC_CONFIG}\" grid_scripts/decomp.sh "
    #qsub -v "CHECKPOINT_DIR=${ENC_CKPT_DIR},TRAINING_CONFIG=${ENC_CONFIG}" grid_scripts/decomp.sh

    SYN_CKPT_DIR="/exp/estengel/miso_res/ud_models_tuned/transformer_pretrained/${lang}_syntax/"
    SYN_CONFIG="miso/training_config/ud_parsing_transformer_fix/transformer_pretrained/${lang}_syntax.jsonnet"

    #echo "qsub -v \"CHECKPOINT_DIR=${SYN_CKPT_DIR},TRAINING_CONFIG=${SYN_CONFIG}\" grid_scripts/decomp.sh "
    #qsub -v "CHECKPOINT_DIR=${SYN_CKPT_DIR},TRAINING_CONFIG=${SYN_CONFIG}" grid_scripts/decomp.sh 

    INT_CKPT_DIR="/exp/estengel/miso_res/ud_models_tuned/transformer_pretrained/${lang}_intermediate/"
    INT_CONFIG="miso/training_config/ud_parsing_transformer_fix/transformer_pretrained/${lang}_intermediate.jsonnet"
    echo "qsub -v \"CHECKPOINT_DIR=${INT_CKPT_DIR},TRAINING_CONFIG=${INT_CONFIG}\" grid_scripts/decomp.sh "
    qsub -v "CHECKPOINT_DIR=${INT_CKPT_DIR},TRAINING_CONFIG=${INT_CONFIG}" grid_scripts/decomp.sh 
done
