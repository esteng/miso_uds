#!/bin/bash

source activate stog

glove=${HOME}/data/glove/glove.840B.300d.zip
data_dir=data/UD_English-EWT

python -u -m stog.commands.train  \
  --train_data ${data_dir}/en_ewt-ud-train.conllu \
  --dev_data ${data_dir}/en_ewt-ud-dev.conllu \
  --test_data ${data_dir}/en_ewt-ud-test.conllu \
  --data_type UD \
  --token_emb_size 300 \
  --batch_first \
  --shuffle \
  --optimizer_type adam \
  --learning_rate 0.001 \
  --use_char_conv \
  --epochs 40 \
  --serialization_dir ckpt \
  --evaluate_on_test \
  --gpu \
  --pretrain_token_emb ${glove}
