#!/bin/bash
#$ -j yes
#$ -N decomp_decomp
#$ -o /home/hltcoe/estengel/miso_research/grid_logs/decode_decomp_spr
#$ -l 'mem_free=10G,h_rt=36:00:00'
#$ -m ae -M elias@jhu.edu
#$ -cwd

source ~/envs/miso_res/bin/activate
ml cuda10.0/toolkit
ml cudnn/7.5.0_cuda10.0

cd /home/hltcoe/estengel/miso_research

./experiments/decomp_train_cpu.sh -a spr_eval -d ${CHECKPOINT_DIR} &> ${CHECKPOINT_DIR}/dev.oracle_out
