import re
import os
import shutil
import time
import datetime
import traceback
from typing import Dict, Optional, List, Union

import torch

from stog.utils import logging
from stog.training.tensorboard import TensorboardWriter
from stog.utils.environment import device_mapping, peak_memory_mb, gpu_memory_mb
from stog.utils.checks import  ConfigurationError
from stog.utils.tqdm import Tqdm
from stog.utils.time import time_to_str


logger = logging.init_logger()



class Trainer:

    def __init__(
            self,
            model,
            optimizer,
            iterator,
            training_dataset,
            dev_dataset = None,
            dev_iterator = None,
            dev_metric = 'loss',
            use_gpu = False,
            patience = None,
            grad_clipping = None,
            shuffle = True,
            num_epochs = 20,
            serialization_dir = None,
            num_serialized_models_to_keep = 20,
            model_save_interval = None,
            summary_interval = 100
    ):
        self._model = model
        self._optimizer = optimizer
        self._iterator = iterator
        self._training_dataset = training_dataset
        self._dev_dataset = dev_dataset
        self._dev_iterator = dev_iterator
        self._dev_metric = dev_metric
        self._use_gpu = use_gpu
        self._patience = patience
        self._grad_clipping = grad_clipping
        self._shuffle = shuffle
        self._num_epochs = num_epochs
        self._serialization_dir = serialization_dir
        self._num_serialized_models_to_keep = num_serialized_models_to_keep
        self._model_save_interval = model_save_interval
        self._summary_interval = summary_interval

        self._num_trained_batches = 0
        self._serialized_paths = []

        if serialization_dir is not None:
            train_log = os.path.join(serialization_dir, 'log', 'train')
            dev_log = os.path.join(serialization_dir, 'log', 'dev')
            self._tensorboard = TensorboardWriter(train_log, dev_log)
        else:
            self._tensorboard = TensorboardWriter()

    def _batch_loss(self, batch: torch.Tensor, for_training: bool) -> torch.Tensor:
        """
        Does a forward pass on the given batch and returns the ``loss`` value in the result.
        If ``for_training`` is `True` also applies regularization penalty.
        """
        output_dict = self._model(**batch)

        try:
            loss = output_dict["loss"]
            if for_training:
                loss += self.model.get_regularization_penalty()
        except KeyError:
            if for_training:
                raise RuntimeError("The model you are trying to optimize does not contain a"
                                   " 'loss' key in the output of model.forward(inputs).")
            loss = None

        return loss

    def _get_metrics(self, total_loss: float, num_batches: int, reset: bool = False) -> Dict[str, float]:
        """
        Gets the metrics but sets ``"loss"`` to
        the total loss divided by the ``num_batches`` so that
        the ``"loss"`` metric is "average loss per batch".
        """
        metrics = self._model.get_metrics(reset=reset)
        metrics["loss"] = float(total_loss / num_batches) if num_batches > 0 else 0.0
        return metrics

    def _train_epoch(self, epoch):
        logger.info('Epoch {}/{}', epoch, self._num_epochs - 1)
        logger.info(f'Peak CPU memory usage MB: {peak_memory_mb()}')
        for gpu, memory in gpu_memory_mb().items():
            logger.info(f"GPU {gpu} memory usage MB: {memory}")

        training_loss = 0.0
        # Set the model to "train" mode.
        self._model.train()

        # Get tqdm for the training batches
        train_generator = self._iterator(self._training_dataset,
                                         num_epochs=1,
                                         shuffle=self._shuffle,
                                         use_gpu=self._use_gpu)
        num_training_batches = self._iterator.get_num_batches(self._training_dataset)

        logger.info('Training...')
        last_save_time = time.time()
        batches_this_epoch = 0
        train_generator_tqdm = Tqdm.tqdm(train_generator, total=num_training_batches)

        for batch in train_generator_tqdm:
            batches_this_epoch += 1
            self._num_trained_batches += 1

            self._optimizer.zero_grad()
            loss = self._batch_loss(batch, for_training=True)
            loss.backward()
            training_loss += loss.item()
            self._optimizer.step()

            # Update the description with the latest metrics
            metrics = self._get_metrics(training_loss, batches_this_epoch)
            description = self._description_from_metrics(metrics)

            train_generator_tqdm.set_description(description, refresh=False)

            # Log parameter values to Tensorboard
            if self._num_trained_batches % self._summary_interval == 0:
                self._tensorboard.add_train_scalar(
                    "loss/loss_train", metrics["loss"], self._num_trained_batches)
                self._metrics_to_tensorboard(
                    self._num_trained_batches,
                    {"epoch_metrics/" + k: v for k, v in metrics.items()})

            # Save model if needed.
            if self._model_save_interval is not None and (
                    time.time() - last_save_time > self._model_save_interval
            ):
                last_save_time = time.time()
                self._save_checkpoint(
                    '{0}.{1}'.format(epoch, time_to_str(int(last_save_time))), [], is_best=False
                )
        return self._get_metrics(training_loss, batches_this_epoch, reset=True)

    def _metrics_to_tensorboard(self,
                                epoch: int,
                                train_metrics: dict,
                                dev_metrics: dict = None) -> None:
        """
        Sends all of the train metrics (and dev metrics, if provided) to tensorboard.
        """
        metric_names = set(train_metrics.keys())
        if dev_metrics is not None:
            metric_names.update(dev_metrics.keys())
        dev_metrics = dev_metrics or {}

        for name in metric_names:
            train_metric = train_metrics.get(name)
            if train_metric is not None:
                self._tensorboard.add_train_scalar(name, train_metric, epoch)
            dev_metric = dev_metrics.get(name)
            if dev_metric is not None:
                self._tensorboard.add_dev_scalar(name, dev_metric, epoch)

    def _metrics_to_console(self,  # pylint: disable=no-self-use
                            train_metrics: dict,
                            dev_metrics: dict = None) -> None:
        """
        Logs all of the train metrics (and validation metrics, if provided) to the console.
        """
        dev_metrics = dev_metrics or {}
        dual_message_template = "%s |  %8.3f  |  %8.3f"
        no_dev_message_template = "%s |  %8.3f  |  %8s"
        no_train_message_template = "%s |  %8s  |  %8.3f"
        header_template = "%s |  %-10s"

        metric_names = set(train_metrics.keys())
        if dev_metrics:
            metric_names.update(dev_metrics.keys())

        name_length = max([len(x) for x in metric_names])

        logger.info(header_template, "Training".rjust(name_length + 13), "Dev")
        for name in metric_names:
            train_metric = train_metrics.get(name)
            dev_metric = dev_metrics.get(name)

            if dev_metric is not None and train_metric is not None:
                logger.info(dual_message_template, name.ljust(name_length), train_metric, dev_metric)
            elif dev_metric is not None:
                logger.info(no_train_message_template, name.ljust(name_length), "N/A", dev_metric)
            elif train_metric is not None:
                logger.info(no_dev_message_template, name.ljust(name_length), train_metric, "N/A")


    def _validate_dev(self):
        """
        Computes the dev loss. Returns it and the number of batches.
        """
        logger.info("Validating on dev")

        self._model.eval()

        if self._dev_iterator is not None:
            dev_iterator = self._dev_iterator
        else:
            dev_iterator = self._iterator

        dev_generator = dev_iterator(self._dev_dataset,
                                     num_epochs=1,
                                     shuffle=False,
                                     use_gpu=self._use_gpu)
        num_dev_batches = dev_iterator.get_num_batches(self._dev_dataset)
        dev_generator_tqdm = Tqdm.tqdm(dev_generator,
                                       total=num_dev_batches)
        batches_this_epoch = 0
        dev_loss = 0
        for batch in dev_generator_tqdm:

            loss = self._batch_loss(batch, for_training=False)
            if loss is not None:
                # You shouldn't necessarily have to compute a loss for validation, so we allow for
                # `loss` to be None.  We need to be careful, though - `batches_this_epoch` is
                # currently only used as the divisor for the loss function, so we can safely only
                # count those batches for which we actually have a loss.  If this variable ever
                # gets used for something else, we might need to change things around a bit.
                batches_this_epoch += 1
                dev_loss += loss.item()

            # Update the description with the latest metrics
            dev_metrics = self._get_metrics(dev_loss, batches_this_epoch)
            description = self._description_from_metrics(dev_metrics)
            dev_generator_tqdm.set_description(description, refresh=False)

        return self._get_metrics(dev_loss, batches_this_epoch, reset=True)

    def train(self):
        """
                Trains the supplied model with the supplied parameters.
                """
        try:
            epoch_counter, dev_metric_per_epoch = self._restore_checkpoint()
        except RuntimeError:
            traceback.print_exc()
            raise ConfigurationError(
                "Could not recover training from the checkpoint.  Did you mean to output to "
                "a different serialization directory or delete the existing serialization "
                "directory?")

        self._enable_gradient_clipping()

        logger.info('Start training...')

        epochs_trained_this_time = 0
        training_metrics = {}
        dev_metrics = {}
        is_best_so_far = True
        best_epoch_dev_metrics = {}
        this_epoch_dev_metric = None
        training_start_time = time.time()

        for epoch in range(epoch_counter, self._num_epochs):
            epoch_start_time = time.time()
            training_metrics = self._train_epoch(epoch)

            if self._dev_dataset is not None:
                with torch.no_grad():
                    dev_metrics = self._validate_dev()

                    # Check dev metric for early stopping
                    this_epoch_dev_metric = dev_metrics[self._dev_metric]

                    # Check validation metric to see if it's the best so far
                    is_best_so_far = self._is_best_so_far(this_epoch_dev_metric, dev_metric_per_epoch)
                    if is_best_so_far:
                        best_epoch_dev_metrics = dev_metrics.copy()
                    dev_metric_per_epoch.append(this_epoch_dev_metric)
                    if self._should_stop_early(dev_metric_per_epoch):
                        logger.info("Ran out of patience.  Stopping training.")
                        break

            self._save_checkpoint(epoch, dev_metric_per_epoch, is_best=is_best_so_far)
            self._metrics_to_tensorboard(epoch, training_metrics, dev_metrics=dev_metrics)
            self._metrics_to_console(training_metrics, dev_metrics=dev_metrics)
            self._tensorboard.add_dev_scalar('learning_rate', self._optimizer.lr, epoch)

            epoch_elapsed_time = time.time() - epoch_start_time
            logger.info("Epoch duration: %s", time.strftime("%H:%M:%S", time.gmtime(epoch_elapsed_time)))

            if epoch < self._num_epochs - 1:
                training_elapsed_time = time.time() - training_start_time
                estimated_time_remaining = training_elapsed_time * \
                    ((self._num_epochs - epoch_counter) / float(epoch - epoch_counter + 1) - 1)
                formatted_time = str(datetime.timedelta(seconds=int(estimated_time_remaining)))
                logger.info("Estimated training time remaining: %s", formatted_time)

            epochs_trained_this_time += 1

        training_elapsed_time = time.time() - training_start_time
        metrics = dict(
            training_duration=time.strftime("%H:%M:%S", time.gmtime(training_elapsed_time)),
            training_start_epoch=epoch_counter,
            training_epochs=epochs_trained_this_time
        )
        for key, value in training_metrics.items():
            metrics["training_" + key] = value
        for key, value in dev_metrics.items():
            metrics["dev_" + key] = value

        if dev_metric_per_epoch:
            # We may not have had validation data, so we need to hide this behind an if.
            best_dev_metric = max(dev_metric_per_epoch)
            metrics.update({f"best_dev_{k}": v for k, v in best_epoch_dev_metrics.items()})
            metrics['best_epoch'] = [i for i, value in enumerate(dev_metric_per_epoch)
                                     if value == best_dev_metric][-1]
        return metrics

    def _enable_gradient_clipping(self) -> None:
        if self._grad_clipping is not None:
            # Pylint is unable to tell that we're in the case that _grad_clipping is not None...
            # pylint: disable=invalid-unary-operand-type
            clip_function = lambda grad: grad.clamp(-self._grad_clipping, self._grad_clipping)
            for parameter in self._model.parameters():
                if parameter.requires_grad:
                    parameter.register_hook(clip_function)

    def _should_stop_early(self, metric_history: List[float]) -> bool:
        """
        uses patience and the validation metric to determine if training should stop early
        """
        if self._patience and self._patience < len(metric_history):
            # Is the best score in the past N epochs worse than or equal the best score overall?
            return max(metric_history[-self._patience:]) <= max(metric_history[:-self._patience])

        return False

    def _is_best_so_far(self,
                        this_epoch_dev_metric: float,
                        dev_metric_per_epoch: List[float]):
        if not dev_metric_per_epoch:
            return True
        else:
            return this_epoch_dev_metric > max(dev_metric_per_epoch)

    def _description_from_metrics(self, metrics: Dict[str, float]) -> str:
        return ', '.join(["%s: %.4f" % (name, value) for name, value in
                          metrics.items() if not name.startswith("_")]) + " ||"

    def _save_checkpoint(self,
                         epoch: Union[int, str],
                         dev_metric_per_epoch: List[float],
                         is_best: Optional[bool] = None) -> None:
        """
        Saves a checkpoint of the model to self._serialization_dir.
        Is a no-op if self._serialization_dir is None.

        Parameters
        ----------
        epoch : Union[int, str], required.
            The epoch of training.  If the checkpoint is saved in the middle
            of an epoch, the parameter is a string with the epoch and timestamp.
        is_best: bool, optional (default = None)
            A flag which causes the model weights at the given epoch to
            be copied to a "best.th" file. The value of this flag should
            be based on some validation metric computed by your model.
        """
        if self._serialization_dir is not None:
            model_path = os.path.join(self._serialization_dir, "model_state_epoch_{}.th".format(epoch))
            model_state = self._model.state_dict()
            torch.save(model_state, model_path)

            training_state = {'epoch': epoch,
                              'dev_metric_per_epoch': dev_metric_per_epoch,
                              'optimizer': self._optimizer.state_dict(),
                              'num_trained_batches': self._num_trained_batches}
            training_path = os.path.join(self._serialization_dir,
                                         "training_state_epoch_{}.th".format(epoch))
            torch.save(training_state, training_path)
            if is_best:
                logger.info("Best validation performance so far. "
                            "Copying weights to '%s/best.th'.", self._serialization_dir)
                shutil.copyfile(model_path, os.path.join(self._serialization_dir, "best.th"))

            if self._num_serialized_models_to_keep and self._num_serialized_models_to_keep >= 0:
                self._serialized_paths.append([time.time(), model_path, training_path])
                if len(self._serialized_paths) > self._num_serialized_models_to_keep:
                    paths_to_remove = self._serialized_paths.pop(0)
                    for fname in paths_to_remove[1:]:
                        os.remove(fname)

    def _find_latest_checkpoint(self):
        if self._serialization_dir is None:
            return None

        serialization_files = os.listdir(self._serialization_dir)
        model_checkpoints = [x for x in serialization_files if 'model_state_epoch' in x]
        # Get the last checkpoint file.  Epochs are specified as either an
        # int (for end of epoch files) or with epoch and timestamp for
        # within epoch checkpoints, e.g. 5.2018-02-02-15-33-42
        found_epochs = [
            # pylint: disable=anomalous-backslash-in-string
            re.search("model_state_epoch_([0-9\.\-]+)\.th", x).group(1)
            for x in model_checkpoints
        ]
        int_epochs = []
        for epoch in found_epochs:
            pieces = epoch.split('.')
            if len(pieces) == 1:
                # Just a single epoch without timestamp
                int_epochs.append([int(pieces[0]), 0])
            else:
                # has a timestamp
                int_epochs.append([int(pieces[0]), pieces[1]])
        last_epoch = sorted(int_epochs, reverse=True)[0]
        if last_epoch[1] == 0:
            epoch_to_load = str(last_epoch[0])
        else:
            epoch_to_load = '{0}.{1}'.format(last_epoch[0], last_epoch[1])

        # model state
        model_state_path = os.path.join(
            self._serialization_dir, 'model_state_epoch_{}.th'.format(epoch_to_load))
        # misc training state, e.g. optimizer state, epoch, etc.
        training_state_path = os.path.join(
            self._serialization_dir, 'training_state_epoch_{}.th'.format(epoch_to_load)
        )
        return (model_state_path, training_state_path)

    def _restore_checkpoint(self):
        last_checkpoint = self._find_latest_checkpoint()
        if last_checkpoint is None:
            return 0, []
        model_state_path, training_state_path = last_checkpoint
        model_state = torch.load(model_state_path, map_location=device_mapping(-1))
        self._model.load_state_dict(model_state)
        training_state = torch.load(training_state_path, map_location=device_mapping(-1))
        self._optimizer.set_state(training_state['optimizer'])
        self._num_trained_batches = training_state['num_trained_batches']
        starting_epoch = training_state['epoch'] + 1
        return starting_epoch, training_state['dev_metric_per_epoch']


