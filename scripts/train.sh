#!/bin/bash

source activate stog
stog=".."
python ${stog}/train.py \
  --train_data ${stog}/data/json/train.json \
  --dev_data ${stog}/data/json/dev.json \
  --token_emb_size 50 \
  --batch_first \
  --shuffle \
  --save_model model \
