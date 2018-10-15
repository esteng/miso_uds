import os
import re
import argparse

import torch

from stog.utils import logging
from stog.utils.params import data_opts, model_opts, train_opts, Params
from stog import models as Models
from stog.data.dataset_builder import dataset_from_params, iterator_from_params
from stog.data.vocabulary import Vocabulary
from stog.training.trainer import Trainer
from stog.utils import environment
from stog.utils.checks import ConfigurationError
from stog.utils.archival import CONFIG_NAME, _DEFAULT_WEIGHTS, archive_model
from stog.commands.evaluate import evaluate
from stog.metrics import dump_metrics

logger = logging.init_logger()


def create_serialization_dir(params: Params) -> None:
    """
    This function creates the serialization directory if it doesn't exist.  If it already exists
    and is non-empty, then it verifies that we're recovering from a training with an identical configuration.
    Parameters
    ----------
    params: ``Params``
        A parameter object specifying an AllenNLP Experiment.
    serialization_dir: ``str``
        The directory in which to save results and logs.
    recover: ``bool``
        If ``True``, we will try to recover from an existing serialization directory, and crash if
        the directory doesn't exist, or doesn't match the configuration we're given.
    """
    if os.path.exists(params.serialization_dir) and os.listdir(params.serialization_dir):
        if not params.recover:
            raise ConfigurationError(f"Serialization directory ({params.serialization_dir}) already exists and is "
                                     f"not empty. Specify --recover to recover training from existing output.")

        logger.info(f"Recovering from prior training at {params.serialization_dir}.")

        recovered_config_file = os.path.join(params.serialization_dir, CONFIG_NAME)
        if not os.path.exists(recovered_config_file):
            raise ConfigurationError("The serialization directory already exists but doesn't "
                                     "contain a config.json. You probably gave the wrong directory.")
        else:
            loaded_params = Params.from_file(recovered_config_file)

            if params != loaded_params:
                raise ConfigurationError("Training configuration does not match the configuration we're "
                                         "recovering from.")
    else:
        if params.recover:
            raise ConfigurationError(f"--recover specified but serialization_dir ({params.serialization_dir}) "
                                     "does not exist.  There is nothing to recover from.")
        os.makedirs(params.serialization_dir, exist_ok=True)


def train_model(params: Params):
    """
    Trains the model specified in the given :class:`Params` object, using the data and training
    parameters also specified in that object, and saves the results.
    Parameters
    ----------
    params : ``Params``
        A parameter object specifying an AllenNLP Experiment.
    Returns
    -------
    best_model: ``Model``
        The model with the best epoch weights.
    """
    environment.set_seed(params.seed, params.numpy_seed, params.torch_seed)

    create_serialization_dir(params)
    environment.prepare_global_logging(params.serialization_dir, params.file_friendly_logging)

    environment.check_for_gpu(params.cuda_device)

    params.to_file(os.path.join(params.serialization_dir, CONFIG_NAME))

    dataset = dataset_from_params(params)

    train_data = dataset['train']
    dev_data = dataset.get('dev')
    test_data = dataset.get('test')

    # Vocabulary and iterator are created here.
    vocab = Vocabulary.from_instances(train_data)
    iterator = iterator_from_params(params)
    iterator.index_with(vocab)

    model = getattr(Models, params.model_type).from_params(vocab, params)

    no_grad_regexes = params.no_grad
    for name, parameter in model.named_parameters():
        if any(re.search(regex, name) for regex in no_grad_regexes):
            parameter.requires_grad_(False)

    frozen_parameter_names, tunable_parameter_names = \
        environment.get_frozen_and_tunable_parameter_names(model)
    logger.info("Following parameters are Frozen  (without gradient):")
    for name in frozen_parameter_names:
        logger.info(name)
    logger.info("Following parameters are Tunable (with gradient):")
    for name in tunable_parameter_names:
        logger.info(name)

    trainer = Trainer.from_params(model, train_data, dev_data, iterator, params)
    
    try:
        metrics = trainer.train()
    except KeyboardInterrupt:
        # if we have completed an epoch, try to create a model archive.
        if os.path.exists(os.path.join(params.serialization_dir, _DEFAULT_WEIGHTS)):
            logger.info("Training interrupted by the user. Attempting to create "
                         "a model archive using the current best epoch weights.")
            archive_model(params.serialization_dir)
        raise

    # Now tar up results
    archive_model(params.serialization_dir)

    logger.info("Loading the best epoch weights.")
    best_model_state_path = os.path.join(params.serialization_dir, 'best.th')
    best_model_state = torch.load(best_model_state_path)
    best_model = model
    best_model.load_state_dict(best_model_state)

    if test_data and params.evaluate_on_test:
        logger.info("The model will be evaluated using the best epoch weights.")
        test_metrics = evaluate(
                best_model, test_data, trainer._dev_iterator, params.batch_size,
                cuda_device=params.cuda_device # pylint: disable=protected-access
        )
        for key, value in test_metrics.items():
            metrics["test_" + key] = value

    dump_metrics(os.path.join(params.serialization_dir, "metrics.json"), metrics, log=True)

    return best_model


if __name__ == "__main__":
    parser = argparse.ArgumentParser('train.py')
    data_opts(parser)
    model_opts(parser)
    train_opts(parser)
    params = Params.from_parser(parser)
    train_model(params)

