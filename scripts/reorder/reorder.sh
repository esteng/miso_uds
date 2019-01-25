#!/bin/bash
source activate stog

data=/home/xma/projects/stog/stog/data
train=${data}/train_amr.txt.features.preproc
dev=${data}/dev_amr.txt.features.preproc
fast_align=/home/xma/projects/stog/experiments/reorder/fast_align
BASEDIR=$(dirname $0)

#python ${BASEDIR}/make_amr_with_align.py --input $dev --align dev.align.output --type node
#exit 0

python ${BASEDIR}/linearize_amr.py ${train} > train.align.input
python ${BASEDIR}/linearize_amr.py ${dev} > dev.align.input

cat train.align.input dev.align.input > all.align.input

${fast_align}/build/fast_align -i all.align.input > all.align.output.forward
python ${BASEDIR}/check_align.py all.align.output.forward all.align.input > check.align.text
#${fast_align}/build/fast_align -i all.align.input -r > all.align.output.reverse
#${fast_align}/build/atools -i all.align.output.forward -j all.align.output.reverse -c intersect > all.align.output.intersect

head -n `wc -l < train.align.input` all.align.output.forward > train.align.output
tail -n `wc -l < dev.align.input` all.align.output.forward > dev.align.output

python ${BASEDIR}/make_amr_with_align.py --input $train --align train.align.output --type node > train_amr.reorder
python ${BASEDIR}/make_amr_with_align.py --input $dev --align dev.align.output --type node> dev_amr.reorder


