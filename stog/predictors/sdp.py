from overrides import overrides
import re
import json
import sys
from collections import OrderedDict
from stog.utils.registrable import Registrable
from stog.utils.checks import ConfigurationError
from stog.utils.string import JsonDict, sanitize
from stog.data import DatasetReader, Instance
from stog.predictors.predictor import Predictor
from stog.utils.string import START_SYMBOL, END_SYMBOL
from stog.data.dataset_readers.amr_parsing.amr import AMRGraph
from stog.utils.exception_hook import ExceptionHook

sys.excepthook = ExceptionHook()


@Predictor.register('semantic_dependencies')
class SDPPredictor(Predictor):
    """
    Predictor for the :class:`~stog.models.stog` model.
    """

    def predict(self, source: str) -> JsonDict:
        return self.predict_json({"source": source})

    @overrides
    def _json_to_instance(self, json_dict: JsonDict) -> Instance:
        """
        Expects JSON that looks like ``{"source": "..."}``.
        """
        source = json_dict["source"]
        return self._dataset_reader.text_to_instance(source)

    @overrides
    def predict_batch_instance(self, instances):
        outputs = []
        _outputs = super(SDPPredictor, self).predict_batch_instance(instances)
        for instance, output in zip(instances, _outputs):
            copy_vocab = instance.fields['src_copy_vocab'].metadata
            node_indexes = output['nodes']
            head_indexes = output['heads']
            head_label_indexes = output['head_labels']
            corefs = output['corefs']

            nodes = []
            head_labels = []
            copy_indicators = []
            nodes_src_indices = []

            seq_len = 0 
            for i, index in enumerate(node_indexes):
                if index == 0:
                    seq_len = i
                    break

                copy_index = index
                if copy_index < 0:
                    import pdb;pdb.set_trace()
                nodes.append(copy_vocab.get_token_from_idx(copy_index))
                copy_indicators.append(1)
                nodes_src_indices.append(copy_index - 1)
                # Lookup the head label.
                head_labels.append(
                    self._model.vocab.get_token_from_index(
                        head_label_indexes[i], 'head_tags')
                )

            nodes = nodes[:seq_len]
            head_indexes = head_indexes[:len(nodes)]
            head_labels = head_labels[:len(nodes)]
            corefs = corefs[:len(nodes)]

            outputs.append(
                dict(
                    nodes=nodes,
                    heads=head_indexes,
                    corefs=corefs,
                    head_labels=head_labels,
                    copy_indicators=copy_indicators,
                    annotated_sentence=instance.fields[
                        "annotated_sentence"
                    ].metadata,
                    sentence_id=instance.fields["sentence_id"].metadata,
                    nodes_src_indices=nodes_src_indices
                )
            )

        return outputs

    def reselve_node_list(self, output):
        node_list = []
        tgt_index_to_node = {}
        src_index_to_node = {}
        # resolve coref
        for tgt_index, coref_index in enumerate(output["corefs"]):
            if tgt_index == int(coref_index - 1) \
                    and output["nodes_src_indices"][tgt_index] \
                    not in src_index_to_node:
                node_list.append(
                    {
                        "nodes": output["nodes"][tgt_index],
                        "heads": {
                            output["heads"][tgt_index]:
                                output["head_labels"][tgt_index]
                        },
                        "copy_indicators":
                            output["copy_indicators"][tgt_index],
                        "tgt_index": tgt_index,
                        "src_index": output["nodes_src_indices"][tgt_index],
                        "pred": 0,
                        "top": 0,
                        "parents": {},
                        "pred_parents": {},
                        "pred_idx": None
                    }
                )
                tgt_index_to_node[tgt_index] = node_list[-1]
                src_index_to_node[
                    output["nodes_src_indices"][tgt_index]
                ] = node_list[-1]
            else:
                if output["nodes_src_indices"][tgt_index] in src_index_to_node:
                    # This means the decoder copy a source token twice 
                    # and maybe failed to give a correct coref
                    # Consider it as a coref
                    tgt_index_to_node[tgt_index] = node_list[-1]
                    coref_tgt_index = src_index_to_node[
                        output["nodes_src_indices"][tgt_index]
                    ]["tgt_index"]
                else:
                    coref_tgt_index = int(coref_index - 1)
                    tgt_index_to_node[tgt_index] = tgt_index_to_node[
                        coref_tgt_index
                    ]

                if output["heads"][tgt_index] not in \
                        tgt_index_to_node[coref_tgt_index]["heads"]:
                    tgt_index_to_node[coref_tgt_index]["heads"][
                        output["heads"][tgt_index]
                    ] = output["head_labels"][tgt_index]
                    tgt_index_to_node[tgt_index] = tgt_index_to_node[coref_tgt_index]

        num_top = 0
        for node in node_list:
            for head_idx, head_label in node["heads"].items():
                if head_idx == 0:
                    node["top"] = 1
                    num_top += 1
                    continue
                if head_idx - 1 not in tgt_index_to_node:
                    # TODO Don't know what's wrong here but parser some generate impossible indices
                    continue

                if "_reversed" in head_label:
                    # It is a reverse relation
                    true_label = head_label.replace("_reversed", "")
                    node["pred"] = 1
                    tgt_index_to_node[head_idx - 1]["parents"][
                        node["tgt_index"]
                    ] = true_label
                else:
                    tgt_index_to_node[head_idx - 1]["pred"] = 1
                    node["parents"][head_idx - 1] = head_label

        if num_top > 1:
            import pdb;pdb.set_trace()
        pred_idx = 0
        for node in sorted(node_list, key=lambda x: x["src_index"]):
            if node["pred"] == 1 and node["src_index"] < 1000:
                node["pred_idx"] = pred_idx
                pred_idx += 1

        for node in node_list:
            for head_tgt_index, label in node["parents"].items():
                if tgt_index_to_node[head_tgt_index]["pred_idx"] is not None \
                        and head_tgt_index != node["tgt_index"]:
                    node["pred_parents"][
                        tgt_index_to_node[head_tgt_index]["pred_idx"]
                    ] = label

        return node_list, src_index_to_node, pred_idx

    @overrides
    def dump_line(self, output):
        string_to_print = "{}\n#tgt_tokens: {}\n".format(
            output["sentence_id"],
            " ".join(output["nodes"])
        )
        node_list, src_index_to_node, num_pred = self.reselve_node_list(output)
        for src_idx, item in enumerate(output["annotated_sentence"]):
            new_item = OrderedDict()
            new_item["id"] = item["id"]
            new_item["form"] = item["form"]
            new_item["lemma"] = item["lemma"]
            new_item["pos"] = item["pos"]

            if src_idx in src_index_to_node:
                node = src_index_to_node[src_idx]

                new_item["top"] = "+" if node["top"] == 1 else "-"
                new_item["pred"] = "+" if node["pred"] == 1 else "-"
                new_item["frame"] = "Y"

                new_item["relations"] = []
                for pred_idx in range(num_pred):
                    if pred_idx in node["pred_parents"]:
                        new_item["relations"].append(
                            node["pred_parents"][pred_idx]
                        )
                    else:
                        new_item["relations"].append("_")
            else:
                new_item["top"] = "-"
                new_item["pred"] = "-"
                new_item["frame"] = "N"

                new_item["relations"] = ["_" for i in range(num_pred)]

            new_item["relations"] = "\t".join(new_item["relations"])
            string_to_print += "\t".join(
                value for value in new_item.values()
            ) + '\n'

        return string_to_print + '\n'
