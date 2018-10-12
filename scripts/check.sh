#!/bin/bash

source activate stog

glove=${HOME}/data/glove/glove.840B.300d.zip
data_dir=data/json/UD_English-EWT

python -u -m stog.commands.train  \
  --train_data ${data_dir}/dev.json \
  --dev_data ${data_dir}/dev.json \
  --token_emb_size 100 \
  --encoder_layers 2 \
  --encoder_size 100 \
  --edge_hidden_size 100 \
  --epochs 40 \
  --batch_first \
  --shuffle \
  --optimizer_type adam \
  --learning_rate 0.001 \
  --emb_dropout 0 \
  --encoder_dropout 0 \
  --hidden_dropout 0 \
  --serialization_dir ckpt \
  --file_friendly_logging \
  --gpu
  # --pretrain_token_emb ${glove} \
