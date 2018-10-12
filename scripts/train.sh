#!/bin/bash

source activate stog

glove=${HOME}/data/glove/glove.840B.300d.zip
data_dir=data/json/UD_English-EWT

python -u -m stog.commands.train  \
  --train_data ${data_dir}/train.json \
  --dev_data ${data_dir}/dev.json \
  --token_emb_size 300 \
  --batch_first \
  --shuffle \
  --optimizer_type adam \
  --learning_rate 0.001 \
  --use_char_conv \
  --epochs 40 \
  --gpu \
  --serialization_dir ckpt
  # --pretrain_token_emb ${glove}
