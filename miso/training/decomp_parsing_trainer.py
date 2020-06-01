import logging
import math
import numpy
import subprocess
from typing import Dict, Optional, List, Tuple, Iterable
import sys

import torch

from allennlp.common import Params
from allennlp.common.checks import ConfigurationError, parse_cuda_device
from allennlp.training import Trainer
from allennlp.training.trainer_pieces import TrainerPieces
from allennlp.training.trainer_base import TrainerBase
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

from miso.data.dataset_readers.decomp_parsing.decomp import DecompGraph
#from miso.data.iterators.data_iterator import DecompDataIterator, DecompBasicDataIterator 
from miso.metrics.s_metric.s_metric import S, compute_s_metric

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


@TrainerBase.register("decomp_parsing")
class DecompTrainer(Trainer):

    def __init__(self,
                 validation_data_path: str,
                 validation_prediction_path: str,
                 semantics_only: bool,
                 drop_syntax: bool,
                 include_attribute_scores: bool = False,
                 warmup_epochs: int = 0,
                 syntactic_method:str = None,
                 *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.validation_data_path = validation_data_path
        self.validation_prediction_path = validation_prediction_path
        self.semantics_only=semantics_only
        self.drop_syntax=drop_syntax
        self.include_attribute_scores=include_attribute_scores

        self._warmup_epochs = warmup_epochs
        self._curr_epoch = 0

    def _update_validation_s_score(self, pred_instances: List[Dict[str, numpy.ndarray]],
                                         true_instances):
        """Write the validation output in pkl format, and compute the S score."""
        logger.info("Computing S")

        for batch in true_instances:
            assert(len(batch) == 1)

        true_graphs = [true_inst for batch in true_instances for true_inst in batch[0]['graph'] ]
        true_sents = [true_inst for batch in true_instances for true_inst in batch[0]['src_tokens_str']]
        pred_graphs = [DecompGraph.from_prediction(pred_inst) for pred_inst in pred_instances]

        ret = compute_s_metric(true_graphs, pred_graphs, true_sents, 
                               self.semantics_only, 
                               self.drop_syntax, 
                               self.include_attribute_scores)

        self.model.val_s_precision = float(ret[0]) * 100
        self.model.val_s_recall = float(ret[1]) * 100
        self.model.val_s_f1 = float(ret[2]) * 100

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
        Computes the validation loss and updates loss weight. 
        Returns it and the number of batches.
        """
        # update loss weight 
        if self.model.loss_mixer is not None:
            self.model.loss_mixer.update_weights(
                                curr_epoch = self._curr_epoch, 
                                total_epochs = self._num_epochs
                                                )

        if self._curr_epoch < self._warmup_epochs:
            # skip the validation step for the warmup period 
            # this greatly reduces train time, since much of it is spent in validation 
            # with non-viable models 
            self._curr_epoch += 1
            return -1, -1

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
        batches_this_epoch = 0
        val_loss = 0
        val_true_instances  = []
        val_outputs: List[Dict[str, numpy.ndarray]] = []
        for batch_group in val_generator_tqdm:
            val_true_instances.append(batch_group)

            batch_output = self._validation_forward(batch_group)
            loss = batch_output.pop("loss", None)
            if loss is not None:
                # You shouldn't necessarily have to compute a loss for validation, so we allow for
                # `loss` to be None.  We need to be careful, though - `batches_this_epoch` is
                # currently only used as the divisor for the loss function, so we can safely only
                # count those batches for which we actually have a loss.  If this variable ever
                # gets used for something else, we might need to change things around a bit.
                batches_this_epoch += 1
                val_loss += loss.detach().cpu().numpy()

            # Update the description with the latest metrics
            val_metrics = training_util.get_metrics(self.model, val_loss, batches_this_epoch)
            description = training_util.description_from_metrics(val_metrics)
            val_generator_tqdm.set_description(description, refresh=False)

            # Update the validation outputs.
            peek = list(batch_output.values())[0]
            batch_size = peek.size(0) if isinstance(peek, torch.Tensor) else len(peek)
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

        self._update_validation_s_score(val_outputs, val_true_instances)

        return val_loss, batches_this_epoch

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
                 validation_iterator: DataIterator = None) -> DecompTrainer:
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

    validation_data_path = params.pop("validation_data_path", None)
    validation_prediction_path = params.pop("validation_prediction_path", None)

    semantics_only = params.pop("semantics_only", False)
    drop_syntax = params.pop("drop_syntax", True)
    include_attribute_scores = params.pop("include_attribute_scores", False)

    warmup_epochs = params.pop("warmup_epochs", 0) 

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
    syntactic_method = params.pop("syntactic_method", None)

    params.assert_empty(cls.__name__)
    return cls(model=model,
               optimizer=optimizer,
               iterator=iterator,
               train_dataset=train_data,
               validation_dataset=validation_data,
               validation_data_path=validation_data_path,
               validation_prediction_path=validation_prediction_path,
               semantics_only=semantics_only,
               warmup_epochs = warmup_epochs, 
               syntactic_method = syntactic_method,
               drop_syntax=drop_syntax,
               include_attribute_scores=include_attribute_scores,
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

