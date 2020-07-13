import pytest
import sys 
import os 

test_path = os.path.dirname(os.path.abspath(__file__))
path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, path) 

from allennlp.commands import ArgumentParserWithDefaults
from allennlp.commands.subcommand import Subcommand
from allennlp.common.util import import_submodules

from utils import assert_successful_overfit, read_metrics, setup_checkpointing_and_args

sys.path.insert(0, test_path) 
from test_interface_overfit import * 

from miso.commands.s_score import SScore
from miso.commands.conllu_score import ConlluScore

def setup_and_test(func, model_path): 
    parser = ArgumentParserWithDefaults(description="Run AllenNLP")
    subparsers = parser.add_subparsers(title='Commands', metavar='')

    subcommands = {
            # Default commands
            "eval": SScore(),
            "spr_eval": SScore(),
            "conllu_eval": ConlluScore()
    }

    for name, subcommand in subcommands.items():
        subparser = subcommand.add_subparser(name, subparsers)
        # configure doesn't need include-package because it imports
        # whatever classes it needs.
        if name != "configure":
            subparser.add_argument('--include-package',
                                   type=str,
                                   action='append',
                                   default=[],
                                   help='additional packages to include')

    arg_list = [f"{func}", f"{model_path}", "dev",
    "--predictor", "decomp_parsing",
    "--batch-size", "1",
    "--beam-size",  "1", 
    "--use-dataset-reader",
    "--line-limit", "2",
    "--cuda-device", "-1",
    "--include-package",  "miso.data.dataset_readers",
    "--include-package", "miso.modules.seq2seq_encoders",
    "--include-package", "miso.models",
    "--include-package", "miso.predictors",
    "--include-package", "miso.metrics"]

    print(" ".join(arg_list) ) 

    args = parser.parse_args(arg_list) 
    if 'func' in dir(args):
        # Import any additional modules needed (to register custom classes).
        for package_name in getattr(args, 'include_package', ()):
            import_submodules(package_name)
        args.func(args)

def test_s_score_base():
    model_path = os.path.join(test_path, "checkpoints", "overfit_decomp_base.ckpt", "model.tar.gz") 
    # if checkpoint doesn't exist, first run other test 
    try: 
        assert(os.path.exists(model_path)) 
    except AssertionError:
        test_decomp_overfit()  

    setup_and_test("eval", model_path) 


def test_s_score_concat_after():
    model_path = os.path.join(test_path, "checkpoints", "overfit_interface_concat_after.ckpt", "model.tar.gz") 
    # if checkpoint doesn't exist, first run other test 
    try: 
        assert(os.path.exists(model_path)) 
    except AssertionError:
        test_interface_concat_after()  

    setup_and_test("eval", model_path) 

def test_s_score_concat_before():
    model_path = os.path.join(test_path, "checkpoints", "overfit_interface_concat_before.ckpt", "model.tar.gz") 
    # if checkpoint doesn't exist, first run other test 
    try: 
        assert(os.path.exists(model_path)) 
    except AssertionError:
        test_interface_concat_before()  

    setup_and_test("eval", model_path) 

def test_s_score_encoder_side():
    model_path = os.path.join(test_path, "checkpoints", "overfit_interface_encoder_side.ckpt", "model.tar.gz") 
    # if checkpoint doesn't exist, first run other test 
    try: 
        assert(os.path.exists(model_path)) 
    except AssertionError:
        test_interface_encoder_side()  

    setup_and_test("eval", model_path) 

