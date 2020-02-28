from typing import List, Iterator, Any
import numpy
from contextlib import contextmanager

import torch
import spacy
import allennlp
from allennlp.predictors.predictor import Predictor
from allennlp.data import Instance
from allennlp.common.util import JsonDict

from miso.data.dataset_readers.amr_parsing.amr import AMR, AMRGraph


def sanitize(x: Any) -> Any:  # pylint: disable=invalid-name,too-many-return-statements
    """
    Sanitize turns PyTorch and Numpy types into basic Python types so they
    can be serialized into JSON.
    """
    if isinstance(x, (str, float, int, bool)):
        # x is already serializable
        return x
    elif isinstance(x, torch.Tensor):
        # tensor needs to be converted to a list (and moved to cpu if necessary)
        return x.cpu().tolist()
    elif isinstance(x, numpy.ndarray):
        # array needs to be converted to a list
        return x.tolist()
    elif isinstance(x, numpy.number):  # pylint: disable=no-member
        # NumPy numbers need to be converted to Python numbers
        return x.item()
    elif isinstance(x, AMR):
        return x
    elif isinstance(x, dict):
        # Dicts need their values sanitized
        return {key: sanitize(value) for key, value in x.items()}
    elif isinstance(x, (spacy.tokens.Token, allennlp.data.Token)):
        # Tokens get sanitized to just their text.
        return x.text
    elif isinstance(x, (list, tuple)):
        # Lists and Tuples need their values sanitized
        return [sanitize(x_i) for x_i in x]
    elif x is None:
        return "None"
    elif hasattr(x, 'to_json'):
        return x.to_json()
    else:
        raise ValueError(f"Cannot sanitize {x} of type {type(x)}. "
                         "If this is your own custom class, add a `to_json(self)` method "
                         "that returns a JSON-like object.")


@Predictor.register("amr_parsing")
class AMRParsingPredictor(Predictor):

    def dump_line(self, outputs: JsonDict) -> str:
        amr = outputs["gold_amr"]
        nodes = outputs["nodes"]
        node_indices = [x + 1 for x in outputs["node_indices"]]
        edge_heads = outputs["edge_heads"]
        edge_types = outputs["edge_types"]
        pred_graph = AMRGraph.from_prediction({
            "nodes": nodes, "corefs": node_indices, "heads": edge_heads, "head_labels": edge_types
        })
        gold_graph = amr.graph
        # Replace the gold graph with the predicted.
        amr.graph = pred_graph

        node_comp = "# ::gold_nodes {}\n# ::pred_nodes {}\n# ::save-date".format(
            " ".join(nodes), " ".join(gold_graph.get_tgt_tokens()))
        serialized_graph = str(amr).replace("# ::save-date", node_comp)
        return serialized_graph + "\n\n"

    @contextmanager
    def capture_model_internals(self) -> Iterator[dict]:
        """
        Context manager that captures the internal-module outputs of
        this predictor's model. The idea is that you could use it as follows:

        .. code-block:: python

            with predictor.capture_model_internals() as internals:
                outputs = predictor.predict_json(inputs)

            return {**outputs, "model_internals": internals}
        """
        results = {}
        hooks = []

        # First we'll register hooks to add the outputs of each module to the results dict.
        def add_output(idx: int):
            def _add_output(mod, _, outputs):
                results[idx] = {"name": str(mod), "output": sanitize(outputs)}
            return _add_output

        for idx, module in enumerate(self._model.modules()):
            if module != self._model:
                hook = module.register_forward_hook(add_output(idx))
                hooks.append(hook)

        # If you capture the return value of the context manager, you get the results dict.
        yield results

        # And then when you exit the context we remove all the hooks.
        for hook in hooks:
            hook.remove()

    def predict_instance(self, instance: Instance) -> JsonDict:
        outputs = self._model.forward_on_instance(instance)
        return sanitize(outputs)

    def predict_batch_instance(self, instances: List[Instance]) -> List[JsonDict]:
        outputs = self._model.forward_on_instances(instances)
        return sanitize(outputs)

