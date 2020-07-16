import pytest
import sys 
import os 

test_path = os.path.dirname(os.path.abspath(__file__))
path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, path) 

from allennlp.commands.train import Train, train_model_from_file

from utils import assert_successful_overfit, read_metrics, setup_checkpointing_and_args

def test_intermediate_lstm():
    config_path = os.path.join(test_path, "configs", "overfit_intermediate_lstm.jsonnet") 
    output_dir = os.path.join(test_path, "checkpoints", "overfit_intermediate_lstm.ckpt") 

    test_args = setup_checkpointing_and_args(config_path, output_dir) 
    train_model_from_file(test_args.param_path,
                          test_args.serialization_dir)

    metrics = read_metrics(output_dir) 
    assert_successful_overfit(metrics, {"validation_s_f1": 100.0, 
                                        "validation_syn_uas": 100.0,
                                        "validation_syn_las": 100.0}) 
    
def test_intermediate_transformer():
    config_path = os.path.join(test_path, "configs", "overfit_intermediate_transformer.jsonnet") 
    output_dir = os.path.join(test_path, "checkpoints", "overfit_intermediate_transformer.ckpt") 

    test_args = setup_checkpointing_and_args(config_path, output_dir) 
    train_model_from_file(test_args.param_path,
                          test_args.serialization_dir)

    metrics = read_metrics(output_dir) 
    assert_successful_overfit(metrics, {"validation_s_f1": 100.0, 
                                        "validation_syn_uas": 100.0,
                                        "validation_syn_las": 100.0}) 
