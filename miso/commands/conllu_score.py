import argparse
import pickle as pkl 
from typing import List, Iterator, Dict  
from collections import namedtuple
import os
import overrides
import tempfile

import numpy as np
import torch
from tqdm import tqdm
import networkx as nx
import logging

from allennlp.commands.predict import _get_predictor, Predict
from allennlp.commands import ArgumentParserWithDefaults
from allennlp.commands.subcommand import Subcommand
from allennlp.common.checks import check_for_gpu, ConfigurationError
from allennlp.common.file_utils import cached_path
from allennlp.common.util import lazy_groups_of
from allennlp.models.archival import load_archive
from allennlp.predictors.predictor import Predictor, JsonDict
from allennlp.data import Instance
from allennlp.common.util import import_submodules

from miso.data.dataset_readers.decomp_parsing.decomp import DecompGraph
from miso.data.dataset_readers.decomp_parsing.decomp_with_syntax import DecompGraphWithSyntax
from miso.metrics.conllu import evaluate_wrapper, UDError
from miso.commands.predict import _ReturningPredictManager 

logger = logging.getLogger(__name__) 

compute_args = {"gold_file":None, "system_file": None}  

ComputeTup = namedtuple("compute_args", sorted(compute_args))

class ArgNamespace:
    def __init__(self,
                input_file,
                batch_size,
                silent,
                beam_size,
                save_pred_path):
        self.input_file = input_file
        self.batch_size = batch_size
        self.silent = silent
        self.beam_size = beam_size
        self.save_pred_path = save_pred_path


class ConlluScore(Subcommand): 
    def add_subparser(self, name: str, parser: argparse._SubParsersAction) -> argparse.ArgumentParser:
        self.name = name 
        description = """Run the specified model against a JSON-lines input file."""
        subparser = parser.add_parser(
            self.name, description=description, help="Use a trained model to make predictions."
        )

        subparser.add_argument(
            "archive_file", type=str, help="the archived model to make predictions with"
        )
        subparser.add_argument("input_file", type=str, help="path to or url of the input file")

        subparser.add_argument("--output-file", type=str, help="path to output file")
        subparser.add_argument(
            "--weights-file", type=str, help="a path that overrides which weights file to use"
        )

        batch_size = subparser.add_mutually_exclusive_group(required=False)
        batch_size.add_argument(
            "--batch-size", type=int, default=1, help="The batch size to use for processing"
        )

        subparser.add_argument(
            "--silent", action="store_true", help="do not print output to stdout"
        )

        cuda_device = subparser.add_mutually_exclusive_group(required=False)
        cuda_device.add_argument(
            "--cuda-device", type=int, default=-1, help="id of GPU to use (if any)"
        )

        subparser.add_argument(
            "--use-dataset-reader",
            action="store_true",
            help="Whether to use the dataset reader of the original model to load Instances. "
            "The validation dataset reader will be used if it exists, otherwise it will "
            "fall back to the train dataset reader. This behavior can be overridden "
            "with the --dataset-reader-choice flag.",
        )

        subparser.add_argument(
            "--dataset-reader-choice",
            type=str,
            choices=["train", "validation"],
            default="validation",
            help="Indicates which model dataset reader to use if the --use-dataset-reader "
            "flag is set.",
        )

        subparser.add_argument(
            "-o",
            "--overrides",
            type=str,
            default="",
            help="a JSON structure used to override the experiment configuration",
        )

        subparser.add_argument(
            "--predictor", type=str, help="optionally specify a specific predictor to use"
        )
        # my options
        subparser.add_argument('--beam-size',
                          type=int,
                          default=1,
                          help="Beam size for seq2seq decoding")

        subparser.add_argument("--save-pred-path", type=str, required=False, 
                                help="optionally specify a path for output pkl") 

        subparser.add_argument("--load-path", type=str, required=False, 
                                help="path to precomuted predicitons") 
        
        subparser.add_argument("--semantics-only", action="store_true", default=False)

        subparser.add_argument("--drop-syntax", action="store_true", default=False)

        subparser.add_argument("--include-attribute-scores", action="store_true", default=False)

        subparser.add_argument("--line-limit", type=int, default=None)

        subparser.add_argument("--json-output-file", type=str, required=False,
                                help="optionally specify a path to output json dict") 

        subparser.add_argument("--oracle", action = "store_true") 

        subparser.set_defaults(func=_construct_and_predict)

        return subparser

def _construct_and_predict(args: argparse.Namespace) -> None:
    predictor = _get_predictor(args)
    args.predictor = predictor
    scorer = ConlluScorer.from_params(args)

    las, mlas, blex = scorer.predict_and_compute()
    print(f"averaged scores") 
    print(f"LAS: {las}, MLAS: {mlas}, BLEX: {blex}") 

class ConlluScorer:
    """
    Reads, predicts, and scores syntactic conllu score 
    """
    def __init__(self,
                predictor = None,
                input_file = "dev",
                batch_size = 32,
                silent = True,
                beam_size = 5, 
                save_pred_path = None,
                load_path = None,
                semantics_only = False,
                drop_syntax = True,
                line_limit = None,
                include_attribute_scores = False,
                oracle = False,
                json_output_file = None):

        self.load_path = load_path
        if self.load_path is not None:
            # don't use a predictor if provided with pre-computed graphs
            pass
        else:
            # otherwise we need to predict 
            self.predictor = predictor
            if predictor is not None:
                self.pred_args = ArgNamespace(input_file, batch_size, silent, beam_size, save_pred_path)

        self.semantics_only = semantics_only
        self.drop_syntax = drop_syntax
        self.line_limit = line_limit
        self.include_attribute_scores = include_attribute_scores
        self.oracle = oracle
        self.json_output_file = json_output_file

        
        self.manager = _ReturningPredictManager(self.predictor,
                                    self.pred_args.input_file,
                                    None,
                                    self.pred_args.batch_size,
                                    not self.pred_args.silent,
                                    True,
                                    self.pred_args.beam_size,
                                    line_limit = self.line_limit,
                                    oracle = self.oracle,
                                    json_output_file = self.json_output_file)

    @staticmethod
    def conllu_dict_to_str(conllu_dict):
        conllu_str = ""
        colnames = ["ID", "form", "lemma", "upos", "xpos", "feats", "head", "deprel", "deps", "misc"]
        for row in conllu_dict:
            vals = [row[cn] for cn in colnames]
            conllu_str += "\t".join(vals) + "\n"
        conllu_str += '\n' 
        return conllu_str

    def predict_and_compute(self):
        assert(self.predictor is not None)
        input_instances, output_graphs = self.manager.run()
   
        # ignore everything except conllu graph 
        if len(output_graphs) > 0 and type(output_graphs[0]) == tuple:
            output_graphs = [x[-1] for x in output_graphs]

        input_graphs = [inst.fields['graph'].metadata for inst in input_instances] 
        input_sents = [inst.fields['src_tokens_str'].metadata for inst in input_instances]

        las_scores = []
        mlas_scores = []
        blex_scores = []

        for i in range(len(input_instances)):
            true_conllu_str = ConlluScorer.conllu_dict_to_str(input_instances[i]['true_conllu_dict'].metadata)
            pred_conllu_str = output_graphs[i]
        

        # make temp files
            with tempfile.NamedTemporaryFile("w") as true_file, \
                tempfile.NamedTemporaryFile("w") as pred_file:

                true_file.write(true_conllu_str)
                pred_file.write(pred_conllu_str) 
                true_file.seek(0) 
                pred_file.seek(0) 
                compute_args["gold_file"] = true_file.name
                compute_args["system_file"] = pred_file.name

                args = ComputeTup(**compute_args)
                try:
                    print(f"score between \n{true_conllu_str} \n{pred_conllu_str}") 
                    score = evaluate_wrapper(args)
                    print(f"score between \n{true_conllu_str} \n{pred_conllu_str}\n{score}") 
                    las_scores.append(100 * score["LAS"].f1)
                    mlas_scores.append(100 * score["MLAS"].f1)
                    blex_scores.append(100 * score["BLEX"].f1)
                except UDError:
                    las_scores.append(0)
                    mlas_scores.append(0)
                    blex_scores.append(0)

        return np.mean(las_scores), np.mean(mlas_scores), np.mean(blex_scores)  
 
    @classmethod
    def from_params(cls, args):
        return cls(predictor=args.predictor,
                   input_file = args.input_file,
                   batch_size = args.batch_size,
                   silent = args.silent,
                   beam_size = args.beam_size, 
                   save_pred_path = args.save_pred_path,
                   load_path = args.load_path,
                   semantics_only = args.semantics_only,
                   drop_syntax = args.drop_syntax,
                   line_limit = args.line_limit,
                   include_attribute_scores = args.include_attribute_scores,
                   oracle = args.oracle,
                   json_output_file = args.json_output_file
                   )

