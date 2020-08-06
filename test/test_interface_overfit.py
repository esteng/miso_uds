import pytest
import sys 
import os 

test_path = os.path.dirname(os.path.abspath(__file__))
path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, path) 

from allennlp.commands.train import Train, train_model_from_file

from utils import assert_successful_overfit, read_metrics, setup_checkpointing_and_args

def test_decomp_overfit():
    config_path = os.path.join(test_path, "configs", "overfit_decomp_base.jsonnet") 
    output_dir = os.path.join(test_path, "checkpoints", "overfit_decomp_base.ckpt") 

    test_args = setup_checkpointing_and_args(config_path, output_dir) 
    train_model_from_file(test_args.param_path,
                          test_args.serialization_dir)

    metrics = read_metrics(output_dir) 
    assert_successful_overfit(metrics, {"validation_s_f1": 100.0, 
                                        "training_uas": 100.0,
                                         "training_las": 100.0}) 

                                        #"training_node_pearson": 0.97728,
                                        #"training_edge_pearson": 0.99999,

def test_interface_concat_after():
    config_path = os.path.join(test_path, "configs", "overfit_synt_sem_concat_after.jsonnet") 
    output_dir = os.path.join(test_path, "checkpoints", "overfit_interface_concat_after.ckpt") 

    test_args = setup_checkpointing_and_args(config_path, output_dir) 
    train_model_from_file(test_args.param_path,
                          test_args.serialization_dir)

    metrics = read_metrics(output_dir) 
    assert_successful_overfit(metrics, {"validation_s_f1": 100.0, 
                                        "validation_syn_uas": 100.0,
                                        "validation_syn_las": 100.0}) 

def test_interface_concat_before():
    config_path = os.path.join(test_path, "configs", "overfit_synt_sem_concat_before.jsonnet") 
    output_dir = os.path.join(test_path, "checkpoints", "overfit_interface_concat_before.ckpt") 

    test_args = setup_checkpointing_and_args(config_path, output_dir) 
    train_model_from_file(test_args.param_path,
                          test_args.serialization_dir)

    metrics = read_metrics(output_dir) 
    assert_successful_overfit(metrics, {"validation_s_f1": 100.0, 
                                        "validation_syn_uas": 100.0,
                                        "validation_syn_las": 100.0}) 


def test_interface_encoder_side():
    config_path = os.path.join(test_path, "configs", "overfit_synt_sem_encoder.jsonnet") 
    output_dir = os.path.join(test_path, "checkpoints", "overfit_interface_encoder_side.ckpt") 

    test_args = setup_checkpointing_and_args(config_path, output_dir) 
    train_model_from_file(test_args.param_path,
                          test_args.serialization_dir)

    metrics = read_metrics(output_dir) 
    assert_successful_overfit(metrics, {"validation_s_f1": 100.0, 
                                        "validation_syn_uas": 100.0,
                                        "validation_syn_las": 100.0}) 
    
def test_interface_encoder_side_transformer():
    config_path = os.path.join(test_path, "configs", "overfit_synt_sem_transformer_encoder.jsonnet") 
    output_dir = os.path.join(test_path, "checkpoints", "overfit_interface_encoder_side_transformer.ckpt") 

    test_args = setup_checkpointing_and_args(config_path, output_dir) 
    train_model_from_file(test_args.param_path,
                          test_args.serialization_dir)

    metrics = read_metrics(output_dir) 
    assert_successful_overfit(metrics, {"validation_s_f1": 100.0, 
                                        "validation_syn_uas": 100.0,
                                        "validation_syn_las": 100.0}) 


#def test_interface_concat_before_transformer():
#    config_path = os.path.join(test_path, "configs", "overfit_synt_sem_transformer_concat_before.jsonnet") 
#    output_dir = os.path.join(test_path, "checkpoints", "overfit_interface_concat_before_transformer.ckpt") 
#
#    test_args = setup_checkpointing_and_args(config_path, output_dir) 
#    train_model_from_file(test_args.param_path,
#                          test_args.serialization_dir)
#
#    metrics = read_metrics(output_dir) 
#    assert_successful_overfit(metrics, {"validation_s_f1": 100.0, 
#                                        "validation_syn_uas": 100.0,
#                                        "validation_syn_las": 100.0}) 
#
#def test_interface_concat_after_transformer():
#    config_path = os.path.join(test_path, "configs", "overfit_synt_sem_transformer_concat_after.jsonnet") 
#    output_dir = os.path.join(test_path, "checkpoints", "overfit_interface_concat_after_transformer.ckpt") 
#
#    test_args = setup_checkpointing_and_args(config_path, output_dir) 
#    train_model_from_file(test_args.param_path,
#                          test_args.serialization_dir)
#
#    metrics = read_metrics(output_dir) 
#    assert_successful_overfit(metrics, {"validation_s_f1": 100.0, 
#                                        "validation_syn_uas": 100.0,
#                                        "validation_syn_las": 100.0}) 
