#!/bin/bash
#$ -j yes
#$ -N decomp_decomp
#$ -o /home/hltcoe/estengel/miso_research/grid_logs/decode_decomp_conllu
#$ -l 'mem_free=10G,h_rt=36:00:00,gpu=1'
#$ -q gpu.q
#$ -m ae -M elias@jhu.edu
#$ -cwd

source ~/envs/miso_res/bin/activate
ml cuda10.0/toolkit
ml cudnn/7.5.0_cuda10.0

cd /home/hltcoe/estengel/miso_research
echo "CHECKPOINT DIR: ${CHECKPOINT_DIR} test_DATA ${TEST_DATA}"

./syntax_experiments/decomp_train.sh -a conllu_predict_ai2 -d ${CHECKPOINT_DIR} -i ${TEST_DATA} #&> ${CHECKPOINT_DIR}/${TEST_DATA}.conllu.out
#./experiments/decomp_test.sh -a spr_eval

