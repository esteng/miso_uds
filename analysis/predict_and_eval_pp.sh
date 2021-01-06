#!/bin/bash
#$ -j yes
#$ -N decomp_pp
#$ -o /home/hltcoe/estengel/miso_research/analysis/no_sem.out
#$ -l 'mem_free=10G,h_rt=36:00:00'
#$ -cwd

source /home/hltcoe/estengel/envs/miso_res/bin/activate

TEST_DATA=/home/hltcoe/estengel/miso_research/analysis/pp_graphs.json

cd /home/hltcoe/estengel/miso_research
model_file=${CHECKPOINT_DIR}/model.tar.gz
output_file=${CHECKPOINT_DIR}/${TEST_DATA}.pred.txt
export PYTHONPATH=$(pwd)/miso:${PYTHONPATH}
echo ${PYTHONPATH}
python -m miso.commands.s_score conllu_eval \
${model_file} ${TEST_DATA} \
--predictor "decomp_syntax_parsing" \
--batch-size 64 \
--beam-size 2 \
--use-dataset-reader \
--cuda-device -1 \
--include-package miso.data.dataset_readers \
--include-package miso.data.tokenizers \
--include-package miso.modules.seq2seq_encoders \
--include-package miso.models \
--include-package miso.predictors \
--include-package miso.metrics
