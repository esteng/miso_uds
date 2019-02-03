#!/usr/bin/env bash

set -e

# Start the CoreNLP server before running this script.
# https://stanfordnlp.github.io/CoreNLP/corenlp-server.html

# The compound file is downloaded from
# https://github.com/ChunchuanLv/AMR_AS_GRAPH_PREDICTION/blob/master/data/joints.txt
compound_file=data/amr_utils/joints.txt
amr_dir=data/amr_v1.0

python -u -m stog.data.dataset_readers.amr_parsing.preprocess.feature_annotator \
    ${amr_dir}/test_amr.txt ${amr_dir}/train_amr.txt ${amr_dir}/dev_amr.txt \
    --compound_file ${compound_file}
