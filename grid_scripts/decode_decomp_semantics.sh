#!/bin/bash
#$ -j yes
#$ -N decomp_decomp
#$ -o /home/hltcoe/estengel/miso_research/grid_logs/decode_decomp
#$ -l 'mem_free=10G,h_rt=36:00:00,gpu=1'
#$ -q gpu.q
#$ -m ae -M elias@jhu.edu
#$ -cwd

source ~/envs/miso_res/bin/activate
ml cuda10.0/toolkit
ml cudnn/7.5.0_cuda10.0

cd /home/hltcoe/estengel/miso_research

./experiments/decomp_train.sh -a eval_sem  -d ${CHECKPOINT_DIR} 

#python -um miso.commands.s_score eval \
#    ${MODEL_DIR}/model.tar.gz \
#    dev \
#    --use-dataset-reader \
#    --beam-size 2 \
#    --batch-size 32 \
#    --cuda-device 0 \
#    --drop-syntax &> ${MODEL_DIR}/decode_structure.out
#    #--save-path ${MODEL_DIR}/outputs.pkl 
