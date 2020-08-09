#!/bin/bash

cd /home/hltcoe/estengel/miso_research

for lang in af de fr fi hu gl; do
    CKPT_DIR="/exp/estengel/miso_res/ud_models_vocab/transformer/${lang}/"
    CONFIG="miso/training_config/ud_parsing/transformer/${lang}_ud_transformer.jsonnet"

    echo "qsub -v \"CHECKPOINT_DIR=${CKPT_DIR},TRAINING_CONFIG=${CONFIG}\" grid_scripts/decomp.sh "
    #qsub -v "CHECKPOINT_DIR=${CKPT_DIR},TRAINING_CONFIG=${CONFIG}" grid_scripts/decomp.sh
done
