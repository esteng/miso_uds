#!/bin/bash

source activate stog

python -u -m train  \
  --train_data ./data/json/dev.json \
  --dev_data ./data/json/dev.json \
  --token_emb_size 50 \
  --batch_first \
  --shuffle \
  --save_model model \
  --gpu
