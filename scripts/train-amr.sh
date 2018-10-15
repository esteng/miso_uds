#!/bin/bash

source activate stog

glove=${HOME}/data/glove/glove.840B.300d.zip
data_dir="./data/AMR_Little_Prince/"

python -u -m stog.commands.train  \
  --train_data ${data_dir}/amr-bank-struct-v1.6-training.txt \
  --dev_data ${data_dir}/amr-bank-struct-v1.6-dev.txt \
  --data_type AMR \
  --token_emb_size 300 \
  --batch_first \
  --shuffle \
  --optimizer_type adam \
  --learning_rate 0.001 \
  --use_char_conv \
  --epochs 40 \
  --serialization_dir "ckpt"
  #--gpu \
  # --pretrain_token_emb ${glove}
