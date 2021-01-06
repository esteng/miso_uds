#! /bin/bash


cd /home/hltcoe/estengel/miso_research 

for i in $(seq 0 11); do
    encoder_checkpoint_dir=${CHECKPOINT_DIR}/layer_${i}/decomp_transformer_encoder_syn_opt_double 
    syntax_only_checkpoint_dir=${CHECKPOINT_DIR}/layer_${i}/decomp_syntax_only 

    ref_file=/home/hltcoe/estengel/glavas_g_2020/data/EWT_clean/en-ud-${SPLIT}.conllu 
    
    echo "SYNTAX LAYER ${i}" 
    python miso/metrics/conllu.py ${ref_file} ${syntax_only_checkpoint_dir}/${SPLIT}.conllu
    echo "ENCODER LAYER ${i}" 
    python miso/metrics/conllu.py ${ref_file} ${encoder_checkpoint_dir}/${SPLIT}.conllu
done  
