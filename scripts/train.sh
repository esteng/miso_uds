#!/bin/bash

source activate stog

glove=${HOME}/data/glove/glove.840B.300d.zip

python -u -m train  \
  --train_data ./data/json/train.json \
  --dev_data ./data/json/dev.json \
  --token_emb_size 300 \
  --batch_first \
  --shuffle \
  --optim adam \
  --learning_rate 0.001 \
  --use_char_conv \
  --epochs 40 \
  --gpu \
  --save_model model \
  --pretrain_token_emb ${glove}
