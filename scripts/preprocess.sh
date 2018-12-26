#!/usr/bin/env bash

# Directory where intermediate utils will be saved to speed up processing.
utils_dir=preproc_utils

# AMR data with **features**
data_dir=data/exp
train_data=${data_dir}/train_amr.txt.features
dev_data=${data_dir}/dev_amr.txt.features
test_data=${data_dir}/test_amr.txt.features

# Directory of PropBank frame files.
# Copy it from LDC2017T10.
propbank_dir=data/misc/propbank-frames-xml-2016-03-08/

# Verbalization list file.
# Download it from amr.isi.edu.
verbalization_file=data/misc/verbalization-list-v1.06.txt

# ========== Set the above variables correctly ==========


printf "Creating utils...`date`\n"
mkdir -p ${utils_dir}
python -u -m stog.data.dataset_readers.amr_parsing.node_utils \
    --amr_train_files ${train_data} \
    --propbank_dir ${propbank_dir} \
    --verbalization_file ${verbalization_file} \
    --dump_dir ${utils_dir} || exit
printf "Done.`date`\n\n"


printf "Recategorizing subgraphs...`date`\n"
python -u -m stog.data.dataset_readers.amr_parsing.preprocess.recategorizer \
    --build_map \
    --amr_train_file ${train_data} \
    --dump_dir ${utils_dir} || exit
python -u -m stog.data.dataset_readers.amr_parsing.preprocess.recategorizer \
    --amr_files ${train_data} ${dev_data} ${test_data} \
    --dump_dir ${utils_dir} || exit
printf "Done.`date`\n\n"



printf "Creating alignment...`date`\n"
python -u -m stog.data.dataset_readers.amr_parsing.preprocess.aligner \
    --json_dir ${utils_dir} \
    --amr_train_files ${train_data}.recategorize \
    --amr_dev_files ${dev_data}.recategorize ${test_data}.recategorize || exit
printf "Done.`date`\n\n"
