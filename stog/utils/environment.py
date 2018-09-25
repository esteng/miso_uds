import random
import numpy
import torch
from stog.utils import logging


logger = logging.init_logger()

'''
Borrowed from AllenNLP: 
    https://github.com/allenai/allennlp/blob/606a61abf04e3108949022ae1bcea975b2adb560/allennlp/common/util.py
'''


def set_seed(seed=13370, numpy_seed=1337, torch_seed=133):
    """
    Sets random seeds for reproducible experiments. This may not work as expected
    if you use this from within a python project in which you have already imported Pytorch.
    If you use the scripts/run_model.py entry point to training models with this library,
    your experiments should be reasonably reproducible. If you are using this from your own
    project, you will want to call this function before importing Pytorch. Complete determinism
    is very difficult to achieve with libraries doing optimized linear algebra due to massively
    parallel execution, which is exacerbated by using GPUs.
    """

    if seed is not None:
        random.seed(seed)
    if numpy_seed is not None:
        numpy.random.seed(numpy_seed)
    if torch_seed is not None:
        torch.manual_seed(torch_seed)
        # Seed all GPUs with the same seed if available.
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(torch_seed)

    logger.info('Init random seeds:\n\tseed: {seed}\n\tnumpy_seed: {numpy_seed}\n\ttorch_seed: {torch_seed}\n'.format(
        seed=seed,
        numpy_seed=numpy_seed,
        torch_seed=torch_seed
    ))
