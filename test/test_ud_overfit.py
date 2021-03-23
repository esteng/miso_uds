import pytest
import sys 
import os 

test_path = os.path.dirname(os.path.abspath(__file__))
path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
print(path) 
sys.path.insert(0, path) 

from allennlp.commands.train import Train, train_model_from_file

from utils import assert_successful_overfit, read_metrics, setup_checkpointing_and_args


def test_ud_de_lstm():
    config_path = os.path.join(test_path, "configs", "overfit_ud_de_lstm.jsonnet") 
    output_dir = os.path.join(test_path, "checkpoints", "overfit_ud_de_lstm.ckpt") 

    test_args = setup_checkpointing_and_args(config_path, output_dir) 
    train_model_from_file(test_args.param_path,
                          test_args.serialization_dir)

    metrics = read_metrics(output_dir) 
    assert_successful_overfit(metrics, {"validation_syn_uas": 100.0,
                                        "validation_syn_las": 100.0}) 


def test_ud_de_transformer():
    config_path = os.path.join(test_path, "configs", "overfit_ud_de_transformer.jsonnet") 
    output_dir = os.path.join(test_path, "checkpoints", "overfit_ud_de_transformer.ckpt") 

    test_args = setup_checkpointing_and_args(config_path, output_dir) 
    train_model_from_file(test_args.param_path,
                          test_args.serialization_dir)

    metrics = read_metrics(output_dir) 
    assert_successful_overfit(metrics, {"validation_syn_uas": 100.0,
                                        "validation_syn_las": 100.0}) 

def test_ud_ewt_lstm():
    config_path = os.path.join(test_path, "configs", "overfit_syntax_only.jsonnet") 
    output_dir = os.path.join(test_path, "checkpoints", "overfit_syntax_only.ckpt") 

    test_args = setup_checkpointing_and_args(config_path, output_dir) 
    train_model_from_file(test_args.param_path,
                          test_args.serialization_dir)

    metrics = read_metrics(output_dir) 
    assert_successful_overfit(metrics, {"validation_syn_uas": 100.0,
                                        "validation_syn_las": 100.0}) 

