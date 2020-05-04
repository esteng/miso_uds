#! /bin/bash


CONFIG=$1
init=$2
n_layer=$3
n_head=$4
dropout=$5
warmup=$6

cp ${CONFIG} ${CONFIG}_orig

sed -e "s/\${INIT_SCALE}/${init}/" \
    -e "s/\${N_LAYERS}/${n_layer}/" \
    -e "s/\${DROPOUT}/${dropout}/" \
    -e "s/\${NHEAD}/${n_head}/" \
    -e "s/\${WARMUP}/${warmup}/" ${CONFIG}_orig > ${CONFIG} 

rm ${CONFIG}_orig



