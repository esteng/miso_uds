import pytest
import sys 
import os 

test_path = os.path.dirname(os.path.abspath(__file__))
path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, path) 

from allennlp.commands.train import Train, train_model_from_file

from utils import assert_successful_overfit, read_metrics, setup_checkpointing_and_args

#def test_decomp_transformer_dot_product_overfit():
#    config_path = os.path.join(test_path, "configs", "overfit_decomp_transformer_dot_attention.jsonnet") 
#    output_dir = os.path.join(test_path, "checkpoints", "overfit_decomp_transformer_dot_attention.ckpt") 
#
#    test_args = setup_checkpointing_and_args(config_path, output_dir) 
#    train_model_from_file(test_args.param_path,
#                          test_args.serialization_dir)
#
#    metrics = read_metrics(output_dir) 
#    assert_successful_overfit(metrics, {"validation_s_f1": 100.0, 
#                                        "training_uas": 100.0,
#                                         "training_las": 100.0}) 
    
def test_decomp_transformer_no_coverage_overfit():
    config_path = os.path.join(test_path, "configs", "overfit_decomp_transformer_no_coverage.jsonnet") 
    output_dir = os.path.join(test_path, "checkpoints", "overfit_decomp_transformer_no_coverage.ckpt") 

    test_args = setup_checkpointing_and_args(config_path, output_dir) 
    train_model_from_file(test_args.param_path,
                          test_args.serialization_dir)

    metrics = read_metrics(output_dir) 
    assert_successful_overfit(metrics, {"validation_s_f1": 100.0, 
                                        "training_uas": 100.0,
                                         "training_las": 100.0}) 

def test_decomp_transformer_overfit():
    config_path = os.path.join(test_path, "configs", "overfit_decomp_transformer.jsonnet") 
    output_dir = os.path.join(test_path, "checkpoints", "overfit_decomp_transformer.ckpt") 

    test_args = setup_checkpointing_and_args(config_path, output_dir) 
    train_model_from_file(test_args.param_path,
                          test_args.serialization_dir)

    metrics = read_metrics(output_dir) 
    assert_successful_overfit(metrics, {"validation_s_f1": 100.0, 
                                        "training_uas": 100.0,
                                         "training_las": 100.0}) 

