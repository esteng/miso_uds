#!/bin/bash
#$ -j yes
#$ -N train_decomp2
#$ -l 'mem_free=50G,h_rt=24:00:00,gpu=1'
#$ -q gpu.q
#$ -m ae -M elias@jhu.edu
#$ -cwd
#$ -o /home/hltcoe/estengel/miso_research/grid_logs/train_decomp0.out

# -q gpu.q@@2080

ml cuda10.0/toolkit
ml cudnn/7.5.0_cuda10.0

cd /home/hltcoe/estengel/miso_research/
# copy current code 
mkdir -p ${CHECKPOINT_DIR}
cp -r miso ${CHECKPOINT_DIR}/miso

echo "RUNNING ON VERSION: " > ${CHECKPOINT_DIR}/stdout.log
git branch >> ${CHECKPOINT_DIR}/stdout.log
git reflog | head -n 1 >> ${CHECKPOINT_DIR}/stdout.log

source ~/envs/miso_res/bin/activate
echo "activated" >> ${CHECKPOINT_DIR}/stdout.log 

echo "Checkpoint dir:  ${CHECKPOINT_DIR}" >> ${CHECKPOINT_DIR}/stdout.log
echo "Trainng config: ${TRAINING_CONFIG}" >> ${CHECKPOINT_DIR}/stdout.log

./experiments/decomp_train.sh -a train  -c ${TRAINING_CONFIG} -d ${CHECKPOINT_DIR}
