#!/bin/bash
#$ -j yes
#$ -N decomp_vg
#$ -o /home/hltcoe/estengel/miso_research/grid_logs/decode_vg
#$ -l 'mem_free=10G,h_rt=36:00:00,gpu=1'
#$ -q gpu.q
#$ -m ae -M elias@jhu.edu
#$ -cwd

# -q gpu.q
source ~/envs/miso_res2/bin/activate
ml cuda10.0/toolkit
ml cudnn/7.5.0_cuda10.0

cd /home/hltcoe/estengel/miso_research

./experiments/vg_train.sh -a eval -d ${CHECKPOINT_DIR}

