#!/bin/bash

smatch_dir=$1
cp $2 ${smatch_dir}
cp $3 ${smatch_dir}
cd ${smatch_dir}
gold=$(basename $2)
pred=$(basename $3)
out=`python score.py "$gold" "$pred"`
out=($out)
echo ${out[1]} ${out[3]} ${out[6]} | sed 's/.$//'

