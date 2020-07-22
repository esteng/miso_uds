#!/usr/bin/env bash

# Causes a pipeline (for example, curl -s http://sipb.mit.edu/ | grep foo)
# to produce a failure return code if any command errors
set -e
set -o pipefail

EXP_DIR=experiments
# Import utility functions.
source ${EXP_DIR}/utils.sh

#CHECKPOINT_DIR=decomp-synt-sem-ckpt
#TRAINING_CONFIG=miso/training_config/overfit_synt_sem.jsonnet
TEST_DATA=dev


function train() {
    rm -fr ${CHECKPOINT_DIR}
    log_info "Training a new transductive model for decomp parsing..."
    python -m allennlp.run train \
    --include-package miso.data.dataset_readers \
    --include-package miso.data.tokenizers \
    --include-package miso.models \
    --include-package miso.training \
    --include-package miso.metrics \
    --include-package miso.modules.seq2seq_encoders \
    -s ${CHECKPOINT_DIR} \
    ${TRAINING_CONFIG}
}

function resume() {
    python scripts/edit_config.py ${CHECKPOINT_DIR}/config.json ${TRAINING_CONFIG}
    python -m allennlp.run train \
    --include-package miso.data.dataset_readers \
    --include-package miso.data.tokenizers \
    --include-package miso.models \
    --include-package miso.training \
    --include-package miso.metrics \
    -s ${CHECKPOINT_DIR} \
    --recover \
    ${TRAINING_CONFIG}
}


function test() {
    log_info "Evaluating a transductive model for decomp parsing..."
    model_file=${CHECKPOINT_DIR}/model.tar.gz
    output_file=${CHECKPOINT_DIR}/test.pred.txt
    python -m allennlp.run predict \
    ${model_file} ${TEST_DATA} \
    --predictor "decomp_syntax_parsing" \
    --batch-size 1 \
    --use-dataset-reader \
    --include-package miso.data.dataset_readers \
    --include-package miso.data.tokenizers \
    --include-package miso.models \
    --include-package miso.predictors \
    --include-package miso.metrics
}

function eval() {
    log_info "Evaluating a transductive model for decomp parsing..."
    model_file=${CHECKPOINT_DIR}/model.tar.gz
    output_file=${CHECKPOINT_DIR}/test.pred.txt
    export PYTHONPATH=$(pwd)/miso:${PYTHONPATH}
    echo ${PYTHONPATH}
    python -m miso.commands.s_score eval \
    ${model_file} ${TEST_DATA} \
    --predictor "decomp_syntax_parsing" \
    --batch-size 1 \
    --beam-size 1 \
    --use-dataset-reader \
    --line-limit 2 \
    --cuda-device -1 \
    --include-package miso.data.dataset_readers \
    --include-package miso.data.tokenizers \
    --include-package miso.modules.seq2seq_encoders \
    --include-package miso.models \
    --include-package miso.predictors \
    --include-package miso.metrics
}

function conllu_eval() {
    log_info "Evaluating a transductive model for decomp parsing..."
    model_file=${CHECKPOINT_DIR}/model.tar.gz
    output_file=${CHECKPOINT_DIR}/test.pred.txt
    export PYTHONPATH=$(pwd)/miso:${PYTHONPATH}
    echo ${PYTHONPATH}
    python -m miso.commands.s_score conllu_eval \
    ${model_file} ${TEST_DATA} \
    --predictor "decomp_syntax_parsing" \
    --batch-size 1 \
    --beam-size 1 \
    --use-dataset-reader \
    --line-limit 2 \
    --cuda-device -1 \
    --include-package miso.data.dataset_readers \
    --include-package miso.data.tokenizers \
    --include-package miso.modules.seq2seq_encoders \
    --include-package miso.models \
    --include-package miso.predictors \
    --include-package miso.metrics
}


function usage() {

    echo -e 'usage: decomp_parsing.sh [-h] -a action'
    echo -e '  -a do [train|test|all].'
    echo -e "  -d checkpoint_dir (Default: ${CHECKPOINT_DIR})."
    echo -e "  -c training_config (Default: ${TRAINING_CONFIG})."
    echo -e "  -i test_data (Default: ${TEST_DATA})."
    echo -e 'optional arguments:'
    echo -e '  -h \t\t\tShow this help message and exit.'

    exit $1

}


function parse_arguments() {

    while getopts ":h:a:d:c:i:" OPTION
    do
        case ${OPTION} in
            h)
                usage 1
                ;;
            a)
                action=${OPTARG:='train'}
                ;;
            d)
                CHECKPOINT_DIR=${OPTARG:=${CHECKPOINT_DIR}}
                ;;
            c)
                TRAINING_CONFIG =${OPTARG:=${TRAINING_CONFIG}}
                ;;
            i)
                TEST_DATA=${OPTARG:=${TEST_DATA}}
                ;;
            ?)
                usage 1
                ;;
        esac
    done

    if [[ -z ${action} ]]; then
        echo ">> Action not provided"
        usage
        exit 1
    fi
}


function main() {

    parse_arguments "$@"
    if [[ "${action}" == "test" ]]; then
        test
    elif [[ "${action}" == "train" ]]; then
        train
    elif [[ "${action}" == "resume" ]]; then
        resume
    elif [[ "${action}" == "all" ]]; then
        train
        test
    elif [[ "${action}" == "eval" ]]; then
        eval
    elif [[ "${action}" == "conllu_eval" ]]; then
        conllu_eval
    fi
}


main "$@"
