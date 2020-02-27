import logging
import math
import numpy
import subprocess
from typing import Dict, Optional, List, Tuple, Iterable

import torch

from allennlp.common import Params
from allennlp.common.checks import ConfigurationError, parse_cuda_device
from allennlp.training import Trainer
from allennlp.training.trainer_pieces import TrainerPieces
from allennlp.data.instance import Instance
from allennlp.data.iterators.data_iterator import DataIterator, TensorDict
from allennlp.models import Model
from allennlp.common.util import lazy_groups_of
from allennlp.common.tqdm import Tqdm
from allennlp.nn import util as nn_util
from allennlp.training import util as training_util
from allennlp.training.checkpointer import Checkpointer
from allennlp.training.learning_rate_schedulers import LearningRateScheduler
from allennlp.training.momentum_schedulers import MomentumScheduler
from allennlp.training.moving_average import MovingAverage
from allennlp.training.optimizers import Optimizer

from miso.data.dataset_readers.amr_parsing.amr import AMRGraph

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


@Trainer.register("amr_parsing")
class AMRTrainer(Trainer):

    def __init__(self,
                 evaluation_script_path: str,
                 smatch_tool_path: str,
                 validation_data_path: str,
                 validation_prediction_path: str,
                 *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.evaluation_script_path = evaluation_script_path
        self.smatch_tool_path = smatch_tool_path
        self.validation_data_path = validation_data_path
        self.validation_prediction_path = validation_prediction_path

    def _update_validation_smatch_score(self, instances: List[Dict[str, numpy.ndarray]]):
        """Write the validation output in Penman format, and compute the Smatch score."""
        logger.info("Computing Smatch")

        # Write the predicted AMR to disk.
        with open(self.validation_prediction_path, "w", encoding="utf8") as dump_f:
            for instance in instances:
                amr = instance["gold_amr"]
                nodes = instance["nodes"]
                node_indices = [x + 1 for x in instance["node_indices"]]
                edge_heads = instance["edge_heads"]
                edge_types = instance["edge_types"]
                pred_graph = AMRGraph.from_prediction({
                    "nodes": nodes, "corefs": node_indices, "heads": edge_heads, "head_labels": edge_types
                })
                gold_graph = amr.graph
                # Replace the gold graph with the predicted.
                amr.graph = pred_graph

                node_comp = "# ::gold_nodes {}\n# ::pred_nodes {}\n# ::save-date".format(
                    " ".join(nodes), " ".join(gold_graph.get_tgt_tokens()))
                serialized_graph = str(amr).replace("# ::save-date", node_comp)
                dump_f.write(serialized_graph + "\n\n")

        try:
            ret = subprocess.check_output([
                self.evaluation_script_path,
                self.smatch_tool_path,
                self.validation_data_path,
                self.validation_prediction_path
            ]).decode().split()
            self.model.validation_smatch_precision = float(ret[0]) * 100
            self.model.validation_smatch_recall = float(ret[1]) * 100
            self.model.validation_smatch_f1 = float(ret[2]) * 100
        except Exception as e:
            logger.info('Exception threw out when computing smatch.')
            logger.error(e, exc_info=True)
            self.model.validation_smatch_precision = 0
            self.model.validation_smatch_recall = 0
            self.model.validation_smatch_f1 = 0

    def _validation_forward(self, batch_group: List[TensorDict]) \
            -> TensorDict:
        """
        Does a forward pass on the given batches and returns the output dict (key, value)
        where value has the shape: [batch_size, *].
        """
        assert len(batch_group) == 1
        batch = batch_group[0]
        batch = nn_util.move_to_device(batch, self._cuda_devices[0])
        output_dict = self.model(**batch)

        return output_dict

    def _validation_loss(self) -> Tuple[float, int]:
        """
        Computes the validation loss. Returns it and the number of batches.
        """
        logger.info("Validating")

        self.model.eval()

        # Replace parameter values with the shadow values from the moving averages.
        if self._moving_average is not None:
            self._moving_average.assign_average_value()

        if self._validation_iterator is not None:
            val_iterator = self._validation_iterator
        else:
            val_iterator = self.iterator

        # Disable multiple gpus in validation.
        num_gpus = 1

        raw_val_generator = val_iterator(self._validation_data,
                                         num_epochs=1,
                                         shuffle=False)
        val_generator = lazy_groups_of(raw_val_generator, num_gpus)
        num_validation_batches = math.ceil(val_iterator.get_num_batches(self._validation_data)/num_gpus)
        val_generator_tqdm = Tqdm.tqdm(val_generator,
                                       total=num_validation_batches)
        val_outputs: List[Dict[str, numpy.ndarray]] = []
        for batch_group in val_generator_tqdm:

            batch_output = self._validation_forward(batch_group)

            # Update the description with the latest metrics
            val_metrics = training_util.get_metrics(self.model, 0, 0)
            description = training_util.description_from_metrics(val_metrics)
            val_generator_tqdm.set_description(description, refresh=False)

            # Update the validation outputs.
            batch_size = list(batch_output.values())[0].size(0)
            instance_separated_output: List[Dict[str, numpy.ndarray]] = [{} for _ in range(batch_size)]
            for name, value in batch_output.items():
                if isinstance(value, torch.Tensor):
                    # NOTE(markn): This is a hack because 0-dim pytorch tensors are not iterable.
                    # This occurs with batch size 1, because we still want to include the loss in that case.
                    if value.dim() == 0:
                        value = value.unsqueeze(0)
                    # shape: [batch_size, *]
                    value = value.detach().cpu().numpy()
                for instance_output, batch_element in zip(instance_separated_output, value):
                    instance_output[name] = batch_element
            val_outputs += instance_separated_output

        # Now restore the original parameter values.
        if self._moving_average is not None:
            self._moving_average.restore()

        self._update_validation_smatch_score(val_outputs)

        return 0, 0

    @classmethod
    def from_params(cls,  # type: ignore
                    params: Params,
                    serialization_dir: str,
                    recover: bool = False,
                    cache_directory: str = None,
                    cache_prefix: str = None):
        pieces = TrainerPieces.from_params(params,  # pylint: disable=no-member
                                           serialization_dir,
                                           recover,
                                           cache_directory,
                                           cache_prefix)
        return _from_params(cls,
                            pieces.model,
                            serialization_dir,
                            pieces.iterator,
                            pieces.train_dataset,
                            pieces.validation_dataset,
                            pieces.params,
                            pieces.validation_iterator)


# An ugly way to inherit ``from_params`` of the ``Trainer`` class in AllenNLP.
def _from_params(cls,  # type: ignore
                 model: Model,
                 serialization_dir: str,
                 iterator: DataIterator,
                 train_data: Iterable[Instance],
                 validation_data: Optional[Iterable[Instance]],
                 params: Params,
                 validation_iterator: DataIterator = None) -> AMRTrainer:
    # pylint: disable=arguments-differ
    patience = params.pop_int("patience", None)
    validation_metric = params.pop("validation_metric", "-loss")
    shuffle = params.pop_bool("shuffle", True)
    num_epochs = params.pop_int("num_epochs", 20)
    cuda_device = parse_cuda_device(params.pop("cuda_device", -1))
    grad_norm = params.pop_float("grad_norm", None)
    grad_clipping = params.pop_float("grad_clipping", None)
    lr_scheduler_params = params.pop("learning_rate_scheduler", None)
    momentum_scheduler_params = params.pop("momentum_scheduler", None)

    evaluation_script_path = params.pop("evaluation_script_path")
    smatch_tool_path = params.pop("smatch_tool_path")
    validation_data_path = params.pop("validation_data_path")
    validation_prediction_path = params.pop("validataion_prediction_path")

    if isinstance(cuda_device, list):
        model_device = cuda_device[0]
    else:
        model_device = cuda_device
    if model_device >= 0:
        # Moving model to GPU here so that the optimizer state gets constructed on
        # the right device.
        model = model.cuda(model_device)

    parameters = [[n, p] for n, p in model.named_parameters() if p.requires_grad]
    optimizer = Optimizer.from_params(parameters, params.pop("optimizer"))
    if "moving_average" in params:
        moving_average = MovingAverage.from_params(params.pop("moving_average"), parameters=parameters)
    else:
        moving_average = None

    if lr_scheduler_params:
        lr_scheduler = LearningRateScheduler.from_params(optimizer, lr_scheduler_params)
    else:
        lr_scheduler = None
    if momentum_scheduler_params:
        momentum_scheduler = MomentumScheduler.from_params(optimizer, momentum_scheduler_params)
    else:
        momentum_scheduler = None

    if 'checkpointer' in params:
        if 'keep_serialized_model_every_num_seconds' in params or \
                'num_serialized_models_to_keep' in params:
            raise ConfigurationError(
                    "Checkpointer may be initialized either from the 'checkpointer' key or from the "
                    "keys 'num_serialized_models_to_keep' and 'keep_serialized_model_every_num_seconds'"
                    " but the passed config uses both methods.")
        checkpointer = Checkpointer.from_params(params.pop("checkpointer"))
    else:
        num_serialized_models_to_keep = params.pop_int("num_serialized_models_to_keep", 20)
        keep_serialized_model_every_num_seconds = params.pop_int(
                "keep_serialized_model_every_num_seconds", None)
        checkpointer = Checkpointer(
                serialization_dir=serialization_dir,
                num_serialized_models_to_keep=num_serialized_models_to_keep,
                keep_serialized_model_every_num_seconds=keep_serialized_model_every_num_seconds)
    model_save_interval = params.pop_float("model_save_interval", None)
    summary_interval = params.pop_int("summary_interval", 100)
    histogram_interval = params.pop_int("histogram_interval", None)
    should_log_parameter_statistics = params.pop_bool("should_log_parameter_statistics", True)
    should_log_learning_rate = params.pop_bool("should_log_learning_rate", False)
    log_batch_size_period = params.pop_int("log_batch_size_period", None)

    params.assert_empty(cls.__name__)
    return cls(model, optimizer, iterator,
               train_data, validation_data,
               evaluation_script_path=evaluation_script_path,
               smatch_tool_path=smatch_tool_path,
               validation_data_path=validation_data_path,
               validation_prediction_path=validation_prediction_path,
               patience=patience,
               validation_metric=validation_metric,
               validation_iterator=validation_iterator,
               shuffle=shuffle,
               num_epochs=num_epochs,
               serialization_dir=serialization_dir,
               cuda_device=cuda_device,
               grad_norm=grad_norm,
               grad_clipping=grad_clipping,
               learning_rate_scheduler=lr_scheduler,
               momentum_scheduler=momentum_scheduler,
               checkpointer=checkpointer,
               model_save_interval=model_save_interval,
               summary_interval=summary_interval,
               histogram_interval=histogram_interval,
               should_log_parameter_statistics=should_log_parameter_statistics,
               should_log_learning_rate=should_log_learning_rate,
               log_batch_size_period=log_batch_size_period,
               moving_average=moving_average)

