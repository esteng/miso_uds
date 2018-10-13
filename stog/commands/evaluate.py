from typing import Dict, Any, Iterable

import torch

from stog.models.model import Model
from stog.utils import environment
from stog.utils import logging
from stog.utils.tqdm import Tqdm


logger = logging.init_logger()


def evaluate(model: Model,
             instances,
             Iterator,
             batch_size,
             cuda_device: int) -> Dict[str, Any]:
    environment.check_for_gpu(cuda_device)

    with torch.no_grad():
        model.eval()

        iterator = Iterator(
            instances,
            batch_size=batch_size,
            sort_key=None,
            repeat=False,
            shuffle=False,
            device=torch.device('cuda', cuda_device)
        )

        predictions = {}
        logger.info("Iterating over dataset")
        generator_tqdm = Tqdm.tqdm(iterator, total=len(iterator))
        for batch in generator_tqdm:
            output_dict = model(batch, for_training=False)
            predictions = add_predictions(output_dict, instances.fields, predictions)
            metrics = model.get_metrics()
            description = ', '.join(["%s: %.2f" % (name, value) for name, value in metrics.items()]) + " ||"
            generator_tqdm.set_description(description, refresh=False)

        return model.get_metrics(reset=True), predictions


def add_predictions(output_dict, fields, predictions):
    def _interpret(data, lengths, vocab=None):
        instances = []
        assert data.size(0) == lengths.size(0)
        for instance_ids, length in zip(data, lengths):
            if vocab:
                instance = [vocab.itos(i.item()) for i in instance_ids[:length]]
            else:
                instance = instance_ids.tolist()
            instances.append(instance)
        return instances

    output_dict.pop('loss', None)
    lengths = output_dict.pop('mask').detach().cpu().sum(dim=1)
    for name, pred in output_dict.items():
        if name in fields:
            instances = _interpret(pred, lengths, fields[name].vocab)
        else:
            instances = _interpret(pred, lengths)
        if name in predictions:
            predictions[name] += instances
        else:
            predictions[name] = instances
    return predictions

