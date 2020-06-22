import pytest
import sys 
import os 
import copy
import shutil
import json 
from collections import namedtuple

test_path = os.path.dirname(os.path.abspath(__file__))
path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, path) 

from allennlp.commands.train import Train, train_model_from_file
from allennlp.common.util import import_submodules

TOL = 0.0001

def assert_successful_overfit(metrics, keys_and_expected_values):
    for k, ev in keys_and_expected_values.items():
        assert(abs(metrics[k] - ev) < TOL) 

def read_metrics(output_dir):
    metrics_path = os.path.join(output_dir, "metrics.json") 
    assert(os.path.exists(metrics_path))
    with open(metrics_path) as f1:
        metrics = json.load(f1) 
    return metrics 

TrainArgs = namedtuple("TrainArgs", ["serialization_dir", "param_path"])

def setup_checkpointing_and_args(config_path, output_dir):        
    assert(os.path.exists(config_path))
    # construct args
    test_args = TrainArgs(serialization_dir = output_dir,
                                   param_path = config_path)

    # import miso 

    include_package = ["miso.data.dataset_readers",
                      "miso.models",
                      "miso.modules.seq2seq_encoders",
                      "miso.training",
                      "miso.metrics"]

    path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    print(f"path is {path}" )
    sys.path.insert(0, path) 
    miso_path = os.path.join(path, "miso") 
    sys.path.insert(0, miso_path) 

    for package_name in include_package:
        import_submodules(package_name)

    # delete output dir if exists 
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir) 

    return test_args

    
def test_decomp_overfit():
    config_path = os.path.join(test_path, "configs", "overfit_decomp_base.jsonnet") 
    output_dir = os.path.join(test_path, "checkpoints", "overfit_decomp_base.ckpt") 

    test_args = setup_checkpointing_and_args(config_path, output_dir) 
    train_model_from_file(test_args.param_path,
                          test_args.serialization_dir)

    metrics = read_metrics(output_dir) 
    assert_successful_overfit(metrics, {"validation_s_f1": 100.0, 
                                        "training_node_pearson": 0.97728,
                                        "training_edge_pearson": 0.99999,
                                        "training_uas": 100.0,
                                         "training_las": 100.0}) 

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
    
def test_intermediate_encoder_side():
    pass 
    
