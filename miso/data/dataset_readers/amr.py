import logging
from typing import Dict

from overrides import overrides
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer
from allennlp.data.tokenizers import Tokenizer, Token
from allennlp.data.fields import TextField, ArrayField, SequenceLabelField, MetadataField, AdjacencyField
from allennlp.data.instance import Instance
from allennlp.common.util import START_SYMBOL, END_SYMBOL

from miso.data.dataset_readers.amr_parsing.io import AMRIO


logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


@DatasetReader.register("amr")
class AMRDatasetReader(DatasetReader):
    """
    AMR dataset reader.
    """
    def __init__(self,
                 source_token_indexers: Dict[str, TokenIndexer],
                 target_token_indexers: Dict[str, TokenIndexer],
                 generation_token_indexers: Dict[str, TokenIndexer],
                 tokenizer: Tokenizer = None,
                 evaluation: bool = False,
                 lazy: bool = False) -> None:

        super().__init__(lazy=lazy)
        self._source_token_indexers = source_token_indexers
        self._target_token_indexers = target_token_indexers
        self._generation_token_indexers = generation_token_indexers
        self._edge_type_indexers = {"edge_types": SingleIdTokenIndexer(namespace="edge_types")}
        self._tokenizer = tokenizer
        self._num_subtokens = 0
        self._num_subtoken_oovs = 0

        self.eval = evaluation

    def report(self):
        if self._num_subtokens != 0:
            logger.info('Subtoken OOV rate: {0:.4f} ({1}/{2})'.format(
                self._num_subtoken_oovs / self._num_subtokens, self._num_subtoken_oovs, self._num_subtokens,
            ))

    @overrides
    def _read(self, file_path):
        logger.info("Reading instances from lines in file at: %s", file_path)
        for i, amr in enumerate(AMRIO.read(file_path)):
            if i > 2000:
                break
            yield self.text_to_instance(amr)
        self.report()

    @overrides
    def text_to_instance(self, amr) -> Instance:
        # pylint: disable=arguments-differ

        # Preprocessing.
        max_tgt_length = None if self.eval else 80
        list_data = amr.graph.get_istog_data(amr, START_SYMBOL, END_SYMBOL, self._tokenizer, max_tgt_length)

        field_dict = dict()

        # Source-side input.
        field_dict["source_tokens"] = TextField(
            tokens=[Token(x) for x in list_data["src_tokens"]],
            token_indexers=self._source_token_indexers
        )
        if list_data['src_token_ids'] is not None:
            field_dict['source_subtoken_ids'] = ArrayField(list_data['src_token_ids'])
            self._number_subtokens += len(list_data['src_token_ids'])
            self._number_subtoken_oovs += len(
                [idx for idx in list_data['src_token_ids'] if idx == self._tokenizer.unk_idx])

        if list_data['src_token_subword_index'] is not None:
            field_dict['source_token_recovery_matrix'] = ArrayField(list_data['src_token_subword_index'])

        field_dict["source_anonymization_tags"] = SequenceLabelField(
            labels=list_data["src_anonym_indicators"],
            sequence_field=field_dict["source_tokens"],
            label_namespace="anonymaziation_tags"
        )

        if list_data["src_pos_tags"] is not None:
            field_dict["source_pos_tags"] = SequenceLabelField(
                labels=list_data["src_pos_tags"],
                sequence_field=field_dict["source_tokens"],
                label_namespace="pos_tags"
            )

        # Target-side input.
        # (exclude the last one <EOS>.)
        field_dict["target_tokens"] = TextField(
            tokens=[Token(x) for x in list_data["tgt_tokens"][:-1]],
            token_indexers=self._target_token_indexers
        )

        if list_data["tgt_pos_tags"] is not None:
            field_dict["target_pos_tags"] = SequenceLabelField(
                labels=list_data["tgt_pos_tags"][:-1],
                sequence_field=field_dict["target_tokens"],
                label_namespace="pos_tags"
            )

        field_dict["target_node_indices"] = SequenceLabelField(
            labels=list_data["tgt_indices"][:-1],
            sequence_field=field_dict["target_tokens"],
            label_namespace="node_indices",
        )

        # Target-side output.
        # Include <BOS> here because we want it in the generation vocabulary such that
        # at the inference starting stage, <BOS> can be correctly initialized.
        field_dict["generation_outputs"] = TextField(
            tokens=[Token(x) for x in list_data["tgt_tokens_to_generate"]],
            token_indexers=self._generation_token_indexers
        )
        field_dict["target_copy_indices"] = SequenceLabelField(
            labels=list_data["tgt_copy_indices"],
            sequence_field=field_dict["generation_outputs"],
            label_namespace="target_copy_indices",
        )

        field_dict["target_attention_map"] = AdjacencyField(  # TODO: replace it with ArrayField.
            indices=list_data["tgt_copy_map"],
            sequence_field=field_dict["generation_outputs"],
            padding_value=0
        )

        field_dict["source_copy_indices"] = SequenceLabelField(
            labels=list_data["src_copy_indices"],
            sequence_field=field_dict["generation_outputs"],
            label_namespace="source_copy_indices",
        )

        field_dict["source_attention_map"] = AdjacencyField(  # TODO: replace it with ArrayField.
            indices=list_data["src_copy_map"],
            sequence_field=TextField(
                [Token(x) for x in list_data["src_copy_vocab"].get_special_tok_list() + list_data["src_tokens"]], None
            ),
            padding_value=0
        )

        field_dict["edge_types"] = TextField(
            tokens=[Token(x) for x in list_data["head_tags"]],
            token_indexers=self._edge_type_indexers
        )

        field_dict["edge_heads"] = SequenceLabelField(
            labels=list_data["head_indices"],
            sequence_field=field_dict["edge_types"],
            label_namespace="edge_heads"
        )

        if list_data.get('node_mask', None) is not None:
            # Valid nodes are 1; pads are 0.
            field_dict['valid_node_mask'] = ArrayField(list_data['node_mask'])

        if list_data.get('edge_mask', None) is not None:
            # A matrix of shape [num_nodes, num_nodes] where entry (i, j) is 1
            # if and only if (1) j < i and (2) j is not an antecedent of i.
            # TODO: try to remove the second constrain.
            field_dict['edge_head_mask'] = ArrayField(list_data['edge_mask'])

        if self.eval:
            # Metadata fields for debugging
            field_dict["source_tokens_str"] = MetadataField(list_data["src_tokens"])
            field_dict["target_tokens_str"] = MetadataField(list_data.get("tgt_tokens", []))
            field_dict["source_copy_vocab"] = MetadataField(list_data["src_copy_vocab"])
            field_dict["tag_lut"] = MetadataField(dict(pos=list_data["pos_tag_lut"]))
            field_dict["source_copy_invalid_ids"] = MetadataField(list_data['src_copy_invalid_ids'])
            field_dict["amr"] = MetadataField(amr)

        return Instance(field_dict)
