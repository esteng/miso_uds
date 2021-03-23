#!/bin/bash
#$ -j yes
#$ -N train_decomp
#$ -l 'mem_free=50G,h_rt=24:00:00,gpu=1'
#$ -q gpu.q@@2080
#$ -cwd
#$ -o /home/hltcoe/estengel/miso_research/transformer_search/logs/tune_transformer_part2.out
#$ -t 1:39:1

idx=$SGE_TASK_ID
opt_file=/home/hltcoe/estengel/miso_research/transformer_search/opt_part_2.txt
args=$(head -n $idx $opt_file | tail -n 1)

echo "args are ${args}" 

ml cuda10.0/toolkit
ml cudnn/7.5.0_cuda10.0

cd /home/hltcoe/estengel/miso_research/

source ~/envs/miso_res/bin/activate
echo "activated" 

dash_args=$(echo ${args} | sed "s/ /-/g") 
echo "dash args ${dash_args}" 
CHECKPOINT_DIR=/exp/estengel/miso_res/tuning/${dash_args}.ckpt 

OG_TRAIN_CONFIG=transformer_search/configs/part2.jsonnet 

TRAINING_CONFIG=transformer_search/configs/${dash_args}.jsonnet

cp ${OG_TRAIN_CONFIG} ${TRAINING_CONFIG}

transformer_search/read_and_replace.sh ${TRAINING_CONFIG} ${args} 

echo "Checkpoint dir:  ${CHECKPOINT_DIR}"
echo "Trainng config: ${TRAINING_CONFIG}" 

./experiments/decomp_train.sh -a train  -c ${TRAINING_CONFIG} -d ${CHECKPOINT_DIR}
