from typing import Dict, Any, Iterable

import torch

from stog.models.model import Model
from stog.utils import environment
from stog.utils import logging
from stog.utils.tqdm import Tqdm
from stog.utils.exception_hook import ExceptionHook
from stog.data.amr import AMRTree
from stog.data.data_writers import AbstractMeaningRepresentationDataWriter
from stog.models.utils import move_to_device
from collections import defaultdict
import sys

logger = logging.init_logger()

#if params.data_type == "AMR":
#    data_writer = AbstractMeaningRepresentationDataWriter()
#    data_writer.set_vocab(train_iterator.vocab)
#   self._data_writer.reset_file_epoch(epoch)

def evaluate(model: Model,
             instances,
             iterator,
             batch_size,
             cuda_device: int):
    environment.check_for_gpu(cuda_device)
    with torch.no_grad():
        model.eval()
        model.decode_type = 'mst'

        test_generator = iterator(
            instances=instances,
            shuffle=False,
            num_epochs = 1
        )

        predictions = defaultdict(list)
        logger.info("Iterating over dataset")
        generator_tqdm = Tqdm.tqdm(
            test_generator,
            total=iterator.get_num_batches(instances)
        )
        data_writer = AbstractMeaningRepresentationDataWriter()
        data_writer.set_vocab(iterator.vocab)
        for batch in generator_tqdm:
            batch = move_to_device(batch, cuda_device)
            output_dict = model(batch, for_training=False)
            predictions = add_predictions(output_dict, batch, predictions, data_writer)
            metrics = model.get_metrics(for_training=False)
            description = ', '.join(["%s: %.2f" % (name, value) for name, value in metrics.items()]) + " ||"
            generator_tqdm.set_description(description, refresh=False)

        return model.get_metrics(reset=True), predictions


def add_predictions(output_dict, batch, predictions, data_writer):
    def _interpret(data, lengths, vocab=None):
        instances = []
        assert data.size(0) == lengths.size(0)
        lengths = lengths.long().tolist()
        for instance_ids, length in zip(data, lengths):
            if vocab:
                instance = [ vocab.get_token_from_index(i, "head_tags") for i in instance_ids[:length].tolist()]
            else:
                instance = instance_ids[:length].tolist()
            instances.append(instance)
        return instances

    # decode tree
    batch_tree = list(data_writer.predict_instance_batch(output_dict, batch))
    predictions['tree'] += batch_tree

    ignored_names = [name for name in output_dict if 'loss' in name]
    for name in ignored_names:
        output_dict.pop(name, None)


    lengths = output_dict.pop('mask').detach().cpu().sum(dim=1)
    for name, pred in output_dict.items():
        pred = pred.detach().cpu()
        if name in ['relations']:
            instances = _interpret(pred, lengths, vocab=data_writer.vocab)
        else:
            instances = _interpret(pred, lengths)

        predictions[name] += instances
    return predictions
