#!/bin/bash

ckpt_dir=$1

cwd=$(pwd)
cd ${ckpt_dir}/ckpt
cp best.th weights.th
tar -czvf model.tar.gz weights.th config.json vocabulary
cd ${cwd} 
