#!/bin/bash
#$ -j yes
#$ -N train_decomp_cpu
#$ -l 'mem_free=50G,h_rt=24:00:00'
#$ -m ae -M elias@jhu.edu
#$ -cwd
#$ -o /home/hltcoe/estengel/miso_research/grid_logs/train_decomp.out


cd /home/hltcoe/estengel/miso_research/
echo "RUNNING ON VERSION: " > ${CHECKPOINT_DIR}/stdout.log
git branch >> ${CHECKPOINT_DIR}/stdout.log
git reflog | head -n 1 >> ${CHECKPOINT_DIR}/stdout.log

source ~/envs/miso_res2/bin/activate
echo "activated" >> ${CHECKPOINT_DIR}/stdout.log 

echo "Checkpoint dir:  ${CHECKPOINT_DIR}" >> ${CHECKPOINT_DIR}/stdout.log
echo "Trainng config: ${TRAINING_CONFIG}" >> ${CHECKPOINT_DIR}/stdout.log

./experiments/decomp_train.sh -a train  -c ${TRAINING_CONFIG} -d ${CHECKPOINT_DIR}
