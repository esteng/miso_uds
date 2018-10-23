#!/bin/bash

set -e
source activate stog

glove=/home/xma/xma/data/glove.840B.300d.txt
data_dir="/export/ssd/xma/data/all_amr"
python -u -m stog.commands.train  \
  --train_data ${data_dir}/train_amr.txt \
  --dev_data ${data_dir}/dev_amr.txt \
  --test_data ${data_dir}/test_amr.txt \
  --data_type AMR \
  --batch_size 64 \
  --token_emb_size 300 \
  --batch_first \
  --shuffle \
  --optimizer_type adam \
  --learning_rate 0.001 \
  --use_char_conv \
  --epochs 40 \
  --serialization_dir "ckpt" \
  --evaluate_on_test \
  --cuda_device 1 \
  --pretrain_token_emb ${glove}\
  --recover

python ${path_to_smatch}/smatch.py -f ./ckpt/predictions.txt ${data_dir}/test_amr.txt --pr
