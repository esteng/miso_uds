#!/bin/bash

CHECKPOINT_DIR=$1

cd /home/hltcoe/estengel/miso_research
echo "GIT INFO\n==================\n"
git branch 
git reflog | head -n 1 

cd grid_scripts 

echo "\n\n"
echo "submitting syntax decode"
qsub -v "CHECKPOINT_DIR=${CHECKPOINT_DIR}" decode_decomp_syntax.sh

echo "submitting structure decode" 
qsub -v "CHECKPOINT_DIR=${CHECKPOINT_DIR}" decode_decomp_structure.sh

echo "submitting sem only decode" 
qsub -v "CHECKPOINT_DIR=${CHECKPOINT_DIR}" decode_decomp_semantics.sh

echo "submitting attr decode" 
qsub -v "CHECKPOINT_DIR=${CHECKPOINT_DIR}" decode_decomp_attr.sh

echo "submitting oracle decode" 
qsub -v "CHECKPOINT_DIR=${CHECKPOINT_DIR}" decode_decomp_spr.sh
