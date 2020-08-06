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

def setup_and_test(func, model_path, predictor = "decomp_parsing"): 
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
    "--predictor", predictor,
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

    args = parser.parse_args(arg_list) 
    if 'func' in dir(args):
        # Import any additional modules needed (to register custom classes).
        for package_name in getattr(args, 'include_package', ()):
            import_submodules(package_name)
        args.func(args)

def base_s_score_test(model_path, backoff_func, capsys, predictor = "decomp_parsing"): 
    # if checkpoint doesn't exist, first run other test 
    try: 
        assert(os.path.exists(model_path)) 
    except AssertionError:
        backoff_func()  

    setup_and_test("eval", model_path, predictor = predictor)
    
    out, err = capsys.readouterr() 
    out = out.strip().split("\n")[0]
    expected = "Precision: 1.0, Recall: 1.0, F1: 1.0" 
    assert(out.strip() == expected.strip())

def base_connlu_test(model_path, backoff_func, capsys): 
    # if checkpoint doesn't exist, first run other test 
    try: 
        assert(os.path.exists(model_path)) 
    except AssertionError:
        backoff_func()  

    setup_and_test("conllu_eval", model_path, predictor = "decomp_syntax_parsing") 
    
    out, err = capsys.readouterr() 
    out = out.strip().split("\n")[1]
    expected = "UAS: 100.0, LAS: 100.0, MLAS: 100.0, BLEX: 100.0" 
    assert(out.strip() == expected.strip())


def test_mst_decode():
    pass 


