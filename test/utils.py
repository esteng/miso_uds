import json 
import sys 
import os 
import shutil
from collections import namedtuple

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
    sys.path.insert(0, path) 
    miso_path = os.path.join(path, "miso") 
    sys.path.insert(0, miso_path) 

    for package_name in include_package:
        import_submodules(package_name)

    # delete output dir if exists 
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir) 

    return test_args
