#!/usr/bin/env bash

# Causes a pipeline (for example, curl -s http://sipb.mit.edu/ | grep foo)
# to produce a failure return code if any command errors
set -e
set -o pipefail

EXP_DIR=experiments
# Import utility functions.
source ${EXP_DIR}/utils.sh

CHECKPOINT_DIR=ckpt
TRAINING_CONFIG=miso/training_config/transductive_semantic_parsing.jsonnet
TEST_DATA=test.json


function train() {
    log_info "Training a new transductive model for AMR parsing..."
    python -m allennlp.run train \
    --include-package miso.data.dataset_readers \
    --include-package miso.models \
    --include-package miso.metrics \
    -s ${CHECKPOINT_DIR} \
    ${TRAINING_CONFIG}
}


function test() {
    log_info "Evaluating a transductive model for AMR parsing..."
}


function usage() {

    echo -e 'usage: amr_parsing.sh [-h] -a action'
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
    elif [[ "${action}" == "all" ]]; then
        train
        test
    fi
}


main "$@"
