from typing import List, Iterator, Optional
import argparse
import sys
import json

from allennlp.commands.subcommand import Subcommand
from allennlp.common.checks import check_for_gpu, ConfigurationError
from allennlp.common.file_utils import cached_path
from allennlp.common.util import lazy_groups_of
from allennlp.models.archival import load_archive
from allennlp.predictors.predictor import Predictor, JsonDict
from allennlp.data import Instance

def _predict(args: argparse.Namespace) -> None:
    predictor = _get_predictor(args)

    if args.silent and not args.output_file:
        print("--silent specified without --output-file.")
        print("Exiting early because no output will be created.")
        sys.exit(0)

    manager = _PredictManager(predictor,
                              args.input_file,
                              args.output_file,
                              args.batch_size,
                              not args.silent,
                              args.use_dataset_reader)
    manager.run()

class _ReturningPredictManager(_PredictManager):
    """
    Extends the _PredictManager class to be able to return data
    which is required for spr scoring, to avoid unneccessary IO
    """
    def __init__(self,
                 predictor: Predictor,
                 input_file: str,
                 output_file: Optional[str],
                 batch_size: int,
                 print_to_console: bool,
                 has_dataset_reader: bool,
                 beam_size: int,
                 line_limit = None) -> None:
        super(_ReturningPredictManager, self).__init__(predictor,
                                                       input_file,
                                                       None,
                                                       batch_size,
                                                       False,
                                                       has_dataset_reader,
                                                       beam_size,
                                                       line_limit = line_limit)

    def run(self):
        has_reader = self._dataset_reader is not None
        instances, results = [], []
        if has_reader:
            for batch in lazy_groups_of(self._get_instance_data(), self._batch_size):
                for model_input_instance, result in zip(batch, self._predict_instances(batch)):
                    instances.append(model_input_instance)
                     
                    results.append(result)
        
        return instances, results

class Predict(Subcommand):
    def add_subparser(self, name: str, parser: argparse._SubParsersAction) -> argparse.ArgumentParser:
        # pylint: disable=protected-access
        description = '''Run the specified model against a JSON-lines input file.'''
        subparser = parser.add_parser(
                name, description=description, help='Use a trained model to make predictions.')

        subparser.add_argument('archive_file', type=str, help='the archived model to make predictions with')
        subparser.add_argument('input_file', type=str, help='path to or url of the input file')

        subparser.add_argument('--output-file', type=str, help='path to output file')
        subparser.add_argument('--weights-file',
                               type=str,
                               help='a path that overrides which weights file to use')

        batch_size = subparser.add_mutually_exclusive_group(required=False)
        batch_size.add_argument('--batch-size', type=int, default=1, help='The batch size to use for processing')

        subparser.add_argument('--silent', action='store_true', help='do not print output to stdout')

        cuda_device = subparser.add_mutually_exclusive_group(required=False)
        cuda_device.add_argument('--cuda-device', type=int, default=-1, help='id of GPU to use (if any)')

        subparser.add_argument('--use-dataset-reader',
                               action='store_true',
                               help='Whether to use the dataset reader of the original model to load Instances. '
                                    'The validation dataset reader will be used if it exists, otherwise it will '
                                    'fall back to the train dataset reader. This behavior can be overridden '
                                    'with the --dataset-reader-choice flag.')

        subparser.add_argument('--dataset-reader-choice',
                               type=str,
                               choices=['train', 'validation'],
                               default='validation',
                               help='Indicates which model dataset reader to use if the --use-dataset-reader '
                                    'flag is set.')

        subparser.add_argument('-o', '--overrides',
                               type=str,
                               default="",
                               help='a JSON structure used to override the experiment configuration')

        subparser.add_argument('--predictor',
                               type=str,
                               help='optionally specify a specific predictor to use')

        subparser.set_defaults(func=_predict)

        return subparser
