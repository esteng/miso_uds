#!/usr/bin/env bash


# ############### AMR v2.0 ################
# # Directory where intermediate utils will be saved to speed up processing.
util_dir=data/AMR/amr_2.0_utils

# AMR data with **features**
data_dir=data/AMR/amr_2.0
train_data=${data_dir}/train_amr.txt.features
dev_data=${data_dir}/dev_amr.txt.features
test_data=${data_dir}/test_amr.txt.features

# Directory of PropBank frame files.
# Copy it from LDC2017T10.
propbank_dir=data/AMR/misc/propbank-frames-xml-2016-03-08/

# Verbalization list file.
# Download it from amr.isi.edu.
verbalization_file=data/AMR/misc/verbalization-list-v1.06.txt
morph_verbalization_file=data/AMR/misc/morph-verbalization-v1.01.txt

# ========== Set the above variables correctly ==========

# printf "Creating utils...`date`\n"
# mkdir -p ${util_dir}
# python -u -m miso.data.dataset_readers.amr_parsing.node_utils \
#     --amr_train_files ${train_data} \
#     --propbank_dir ${propbank_dir} \
#     --verbalization_file ${verbalization_file} \
#     --dump_dir ${util_dir} || exit
# printf "Done.`date`\n\n"
#
printf "Cleaning inputs...`date`\n"
python -u -m miso.data.dataset_readers.amr_parsing.preprocess.input_cleaner \
    --amr_files ${test_data} \
    ${train_data} ${dev_data} || exit
printf "Done.`date`\n\n"

printf "Recategorizing subgraphs...`date`\n"
# python -u -m miso.data.dataset_readers.amr_parsing.preprocess.recategorizer \
#     --build_utils \
#     --amr_train_file ${train_data}.input_clean \
#     --dump_dir ${util_dir} || exit
python -u -m miso.data.dataset_readers.amr_parsing.preprocess.recategorizer \
    --dump_dir ${util_dir} \
    --amr_files ${test_data}.input_clean \
    ${train_data}.input_clean ${dev_data}.input_clean || exit
printf "Done.`date`\n\n"

# printf "Morph verbalization...`date`\n"
# python -u -m miso.data.dataset_readers.amr_parsing.preprocess.morph \
#     --morph_verbalization_file ${morph_verbalization_file} \
#     --amr_files ${test_data}.input_clean.recategorize
# printf "Done.`date`\n\n"

printf "Creating alignment...`date`\n"
python -u -m miso.data.dataset_readers.amr_parsing.preprocess.sense_remover \
    --util_dir ${util_dir} \
    --amr_files ${test_data}.input_clean.recategorize \
    ${train_data}.input_clean.recategorize ${dev_data}.input_clean.recategorize || exit
printf "Done.`date`\n\n"

printf "Rename preprocessed files...`date`\n"
mv ${test_data}.input_clean.recategorize.nosense ${test_data}.preproc
mv ${train_data}.input_clean.recategorize.nosense ${train_data}.preproc
mv ${dev_data}.input_clean.recategorize.nosense ${dev_data}.preproc
rm ${data_dir}/*.input_clean*