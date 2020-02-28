from allennlp.predictors.predictor import Predictor
from allennlp.common.util import JsonDict

from miso.data.dataset_readers.amr_parsing.amr import AMRGraph


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
        return serialized_graph
