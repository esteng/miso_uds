#!/usr/bin/env bash

# Directory where intermediate utils will be saved to speed up processing.
utils_dir=preproc_utils

# AMR data with **features**
data_dir=data/exp
test_data=${data_dir}/test_amr.txt.features.preproc

# Directory of PropBank frame files.
# Copy it from LDC2017T10.
propbank_dir=data/misc/propbank-frames-xml-2016-03-08/

# Verbalization list file.
# Download it from amr.isi.edu.
verbalization_file=data/misc/verbalization-list-v1.06.txt

# ========== Set the above variables correctly ==========


printf "Frame lookup...`date`\n"
python -u -m stog.data.dataset_readers.amr_parsing.postprocess.node_restore \
    --amr_files ${test_data} \
    --util_dir ${utils_dir} || exit
printf "Done.`date`\n\n"

# printf "Dump Spotlight Wikification...`date`\n"
# python -u -m stog.data.dataset_readers.amr_parsing.postprocess.wikification \
#     --dump_spotlight_wiki \
#     --amr_files ${test_data} \
#     --util_dir ${utils_dir} || exit
# printf "Done.`date`\n\n"

printf "Wikification...`date`\n"
python -u -m stog.data.dataset_readers.amr_parsing.postprocess.wikification \
    --amr_files ${test_data}.frame \
    --util_dir ${utils_dir} || exit
printf "Done.`date`\n\n"

printf "Expanding nodes...`date`\n"
python -u -m stog.data.dataset_readers.amr_parsing.postprocess.expander \
    --amr_files ${test_data}.frame.wiki \
    --util_dir ${utils_dir} || exit
printf "Done.`date`\n\n"

