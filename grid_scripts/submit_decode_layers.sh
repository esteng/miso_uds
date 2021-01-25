#!/bin/bash

source ~/envs/miso_res/bin/activate
ml cuda10.0/toolkit
ml cudnn/7.5.0_cuda10.0

cd /home/hltcoe/estengel/miso_research

for i in $(seq 0 11); do
    encoder_checkpoint_dir=${CHECKPOINT_DIR}/layer_${i}/decomp_transformer_encoder_syn_opt_double 
    syntax_only_checkpoint_dir=${CHECKPOINT_DIR}/layer_${i}/decomp_syntax_only 
    base_checkpoint_dir=${CHECKPOINT_DIR}/layer_${i}/decomp_transformer_base 
    
    # syntax 
    #echo "qsub -v \"CHECKPOINT_DIR=${encoder_checkpoint_dir},TEST_DATA=${TEST_DATA}\" grid_scripts/decode_decomp_conllu.sh"
    #qsub -v "CHECKPOINT_DIR=${encoder_checkpoint_dir},TEST_DATA=${TEST_DATA}" grid_scripts/decode_decomp_conllu.sh
    #echo "qsub -v \"CHECKPOINT_DIR=${syntax_only_checkpoint_dir},TEST_DATA=${TEST_DATA}\" grid_scripts/decode_decomp_conllu.sh"
    #qsub -v "CHECKPOINT_DIR=${syntax_only_checkpoint_dir},TEST_DATA=${TEST_DATA}" grid_scripts/decode_decomp_conllu.sh
    #echo "qsub -v \"CHECKPOINT_DIR=${encoder_checkpoint_dir},TEST_DATA=${TEST_DATA}\" grid_scripts/decode_decomp_structure.sh"
    ## semantic structure 
    #qsub -v "CHECKPOINT_DIR=${encoder_checkpoint_dir},TEST_DATA=${TEST_DATA}" grid_scripts/decode_decomp_structure.sh
    #echo "qsub -v \"CHECKPOINT_DIR=${base_checkpoint_dir},TEST_DATA=${TEST_DATA}\" grid_scripts/decode_decomp_structure.sh"
    #qsub -v "CHECKPOINT_DIR=${base_checkpoint_dir},TEST_DATA=${TEST_DATA}" grid_scripts/decode_decomp_structure.sh
    # attributes 
    echo "qsub -v \"CHECKPOINT_DIR=${encoder_checkpoint_dir},TEST_DATA=${TEST_DATA}\" grid_scripts/decode_decomp_spr.sh"
    qsub -v "CHECKPOINT_DIR=${encoder_checkpoint_dir},TEST_DATA=${TEST_DATA}" grid_scripts/decode_decomp_spr.sh
    echo "qsub -v \"CHECKPOINT_DIR=${base_checkpoint_dir},TEST_DATA=${TEST_DATA}\" grid_scripts/decode_decomp_spr.sh" 
    qsub -v "CHECKPOINT_DIR=${base_checkpoint_dir},TEST_DATA=${TEST_DATA}" grid_scripts/decode_decomp_spr.sh
done 

