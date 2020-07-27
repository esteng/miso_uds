import json
import sys
import os
import shutil
import numpy as np
import pytest 
from copy import copy
import torch

test_path = os.path.dirname(os.path.abspath(__file__))
path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, path) 

from allennlp.common import Params
from allennlp.models import Model
from allennlp.common.util import  import_submodules
from miso.models.decomp_parser import DecompParser
from utils import TOL

@pytest.fixture()
def setup():
    # import miso 
    include_package = ["miso.data.dataset_readers",
                      "miso.models",
                      "miso.data.tokenizers",
                      "miso.modules.seq2seq_encoders",
                      "miso.modules",
                      "miso.training",
                      "miso.metrics"]


    for package_name in include_package:
        import_submodules(package_name)

def test_load_state_dict(setup):
    decomp_checkpoint_path = os.path.join(test_path, "checkpoints", "overfit_interface_encoder_side.ckpt")
    config_path = os.path.join(test_path, "configs", "overfit_synt_sem_encoder.jsonnet")
    params = Params.from_file(config_path) 
    model = Model.load(params, decomp_checkpoint_path) 

    prev_weight = copy(model.biaffine_parser.edge_type_query_linear.weight.data.numpy())

    ud_checkpoint_path = os.path.join(test_path, "checkpoints",  "overfit_syntax_only.ckpt") 
    model.load_partial(os.path.join(ud_checkpoint_path, "best.th")) 

    curr_weight = model.biaffine_parser.edge_type_query_linear.weight.data.numpy()
    
    # assert that weights have changed 
    assert(np.abs(np.sum(prev_weight - curr_weight)) > TOL) 


def test_load_ud_to_uds(setup):
    # load ud states 
    ud_checkpoint_path = os.path.join(test_path, "checkpoints",  "overfit_syntax_only.ckpt", "best.th") 
    ud_state_dict = torch.load(ud_checkpoint_path)

    # create a uds model 
    decomp_checkpoint_path = os.path.join(test_path, "checkpoints", "overfit_interface_encoder_side.ckpt")
    config_path = os.path.join(test_path, "configs", "load_weights_ud_to_uds.jsonnet")
    params = Params.from_file(config_path)
    params["model"]["pretrained_weights"] = ud_checkpoint_path
    model = Model.load(params, decomp_checkpoint_path) 
    model.load_partial(ud_checkpoint_path)
    
    # compare state dicts 
    key = "biaffine_parser.edge_type_query_linear.weight"
    loaded_weights = model.state_dict()[key].data.numpy()
    saved_weights = ud_state_dict[key].data.numpy()

    print(f"loaded from model {loaded_weights}") 
    print(f"pretrained {saved_weights}") 

    # assert same 
    assert(np.abs(np.sum(loaded_weights - saved_weights)) < TOL) 

def test_load_uds_to_ud(setup):
    # load ud states 
    uds_checkpoint_path = os.path.join(test_path, "checkpoints",  "overfit_interface_encoder_side.ckpt", "best.th") 
    uds_state_dict = torch.load(uds_checkpoint_path)

    # create a ud model 
    decomp_checkpoint_path = os.path.join(test_path, "checkpoints", "overfit_syntax_only.ckpt")
    config_path = os.path.join(test_path, "configs", "load_weights_uds_to_ud.jsonnet")
    params = Params.from_file(config_path)
    params["model"]["pretrained_weights"] = uds_checkpoint_path
    model = Model.load(params, decomp_checkpoint_path) 
    model.load_partial(uds_checkpoint_path)

    # compare state dicts 
    key = "biaffine_parser.edge_type_query_linear.weight"
    loaded_weights = model.state_dict()[key].data.numpy()
    saved_weights = uds_state_dict[key].data.numpy()

    # assert same 
    assert(np.abs(np.sum(loaded_weights - saved_weights)) < TOL) 

