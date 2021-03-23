#!/bin/bash

CHECKPOINT_DIR=$1

function clean {
    # remove .th files 
    echo "REMOVING";
    rm ${CHECKPOINT_DIR}/ckpt/model*th
    rm ${CHECKPOINT_DIR}/ckpt/weights.th
    rm ${CHECKPOINT_DIR}/ckpt/training*th

}

cd ${CHECKPOINT_DIR}/ckpt

# check if model.tar.gz exists
if test -f "model.tar.gz"; then
    echo "$(ls model.tar.gz)"
    echo "yes model";
    clean;
else
    echo "no model";
    # if not, create it and clean 
    cp best.th weights.th;
    tar -czvf model.tar.gz weights.th config.json vocabulary;
    clean; 
fi

