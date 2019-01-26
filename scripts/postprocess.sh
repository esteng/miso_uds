#!/usr/bin/env bash

# Directory where intermediate utils will be saved to speed up processing.
util_dir=data/amr_utils

# AMR data with **features**
data_dir=data/exp
# test_data=${data_dir}/test_amr.txt.features.preproc
test_data=test.best.txt

# ========== Set the above variables correctly ==========

printf "Frame lookup...`date`\n"
python -u -m stog.data.dataset_readers.amr_parsing.postprocess.node_restore \
    --amr_files ${test_data} \
    --util_dir ${util_dir} || exit
printf "Done.`date`\n\n"

# printf "Dump Spotlight Wikification...`date`\n"
# python -u -m stog.data.dataset_readers.amr_parsing.postprocess.wikification \
#     --dump_spotlight_wiki \
#     --amr_files ${test_data} \
#     --util_dir ${util_dir} || exit
# printf "Done.`date`\n\n"

printf "Wikification...`date`\n"
python -u -m stog.data.dataset_readers.amr_parsing.postprocess.wikification \
    --amr_files ${test_data}.frame \
    --util_dir ${util_dir} || exit
printf "Done.`date`\n\n"

printf "Expanding nodes...`date`\n"
python -u -m stog.data.dataset_readers.amr_parsing.postprocess.expander \
    --amr_files ${test_data}.frame.wiki \
    --util_dir ${util_dir} || exit
printf "Done.`date`\n\n"

