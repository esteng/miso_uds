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

        logger.info("Iterating over dataset")
        generator_tqdm = Tqdm.tqdm(iterator, total=len(iterator))
        for batch in generator_tqdm:
            model(batch, for_training=False)
            metrics = model.get_metrics()
            description = ', '.join(["%s: %.2f" % (name, value) for name, value in metrics.items()]) + " ||"
            generator_tqdm.set_description(description, refresh=False)

        return model.get_metrics(reset=True)