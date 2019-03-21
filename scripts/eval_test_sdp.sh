#!/bin/bash
eval_dir=$1
gold=$2
pred=$3
first_line=`head -n 1 $pred`

if [[ ! $first_line =~ ^#SDP.* ]]
then
  sed -i "1s/^/#SDP2015\n/" $pred
fi

grep -v "#tgt_tokens:" $pred > $pred.graph
python $eval_dir/score.py $gold $pred.graph  2>&1 | head -n 18 | tail -n 4 | cut -d " " -f2

source activate stog
grep -E "#tgt_tokens:" $pred | cut -d" " -f2- > $pred.tokens
python /home/xma/projects/miso/stog/scripts/extract_sdp_token.py $gold | tail +2 > $pred.gold.tokens
/home/xma/tools/mosesdecoder/scripts/generic/multi-bleu.perl $pred.gold.tokens < $pred.tokens | cut -d " " -f3
