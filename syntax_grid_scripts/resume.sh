#!/bin/bash
#$ -j yes
#$ -N resume_decomp
#$ -l 'mem_free=50G,h_rt=28:00:00,gpu=1'
#$ -q gpu.q@@2080
#$ -m ae -M elias@jhu.edu
#$ -cwd
#$ -o /home/hltcoe/estengel/miso_research/grid_logs/resume.out


ml cuda10.0/toolkit
ml cudnn/7.5.0_cuda10.0

cd /home/hltcoe/estengel/miso_research/
echo "RUNNING ON VERSION: "
git branch >> ${CHECKPOINT_DIR}/stdout.log
git reflog | head -n 1 >> ${CHECKPOINT_DIR}/stdout.log

source ~/envs/miso_res2/bin/activate
echo "activated" 

echo "Checkpoint dir:  ${CHECKPOINT_DIR}"
echo "Trainng config: ${TRAINING_CONFIG}" 

./experiments/decomp_train.sh -a resume -c ${TRAINING_CONFIG} -d ${CHECKPOINT_DIR}
