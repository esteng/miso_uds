
from typing import Dict, List, Tuple
import logging
import os

from overrides import overrides
# NLTK is so performance orientated (ha ha) that they have lazy imports. Why? Who knows.
from stog.data.dataset_readers.amr_parsing.io import AMRIO
from stog.data.dataset_readers.amr_parsing.amr import AMRGraph
from stog.utils.file import cached_path
from stog.data.dataset_readers.dataset_reader import DatasetReader
from stog.data.fields import TextField, SpanField, SequenceLabelField, ListField, MetadataField, Field, AdjacencyField
from stog.data.instance import Instance
from stog.data.token_indexers import TokenIndexer, SingleIdTokenIndexer
from stog.data.tokenizers import Token
from stog.data.tokenizers.word_splitter import SpacyWordSplitter
from stog.data.dataset_readers.dataset_utils.span_utils import enumerate_spans
from stog.utils.checks import ConfigurationError
from stog.utils.string import START_SYMBOL, END_SYMBOL
import json

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


@DatasetReader.register("amr_trees")
class AbstractMeaningRepresentationDatasetReader(DatasetReader):
    '''
    Dataset reader for AMR data
    '''
    def __init__(self,
                 token_indexers: Dict[str, TokenIndexer] = None,
                 word_splitter = None,
                 lazy: bool = False,
                 skip_first_line: bool = True
                 ) -> None:

        super().__init__(lazy=lazy)
        self._token_indexers = token_indexers or {'tokens': SingleIdTokenIndexer()}
        self._word_splitter = word_splitter or SpacyWordSplitter()
        self._skip_first_line = skip_first_line

    @overrides
    def _read(self, file_path):
        # if `file_path` is a URL, redirect to the cache
        file_path = cached_path(file_path)
        directory, filename = os.path.split(file_path)

        logger.info("Reading instances from lines in file at: %s", file_path)
        for amr in AMRIO.read(file_path):
            yield self.text_to_instance(amr)

    @overrides
    def text_to_instance(self, amr) -> Instance:
        # pylint: disable=arguments-differ

        fields: Dict[str, Field] = {}

        list_data = amr.graph.get_list_data(amr, START_SYMBOL, END_SYMBOL)

        # These four fields are used for seq2seq model and target side self copy
        fields["src_tokens"] = TextField(
            tokens=[Token(x) for x in list_data["src_tokens"]],
            token_indexers={k: v for k, v in self._token_indexers.items() if 'encoder' in k}
        )

        fields["tgt_tokens"] = TextField(
            tokens=[Token(x) for x in list_data["tgt_tokens"]],
            token_indexers={k: v for k, v in self._token_indexers.items() if 'decoder' in k}
        )

        fields["src_pos_tags"] = SequenceLabelField(
            labels=list_data["src_pos_tags"],
            sequence_field=fields["src_tokens"],
            label_namespace="pos_tags"
        )

        fields["tgt_pos_tags"] = SequenceLabelField(
            labels=list_data["tgt_pos_tags"],
            sequence_field=fields["tgt_tokens"],
            label_namespace="pos_tags"
        )

        fields["tgt_copy_indices"] = SequenceLabelField(
            labels=list_data["tgt_copy_indices"],
            sequence_field=fields["tgt_tokens"],
            label_namespace="coref_tags",
        )

        fields["tgt_copy_mask"] = SequenceLabelField(
            labels=list_data["tgt_copy_mask"],
            sequence_field=fields["tgt_tokens"],
            label_namespace="coref_mask_tags",
        )

        fields["tgt_copy_map"] = AdjacencyField(
            indices=list_data["tgt_copy_map"],
            sequence_field=fields["tgt_tokens"],
            padding_value=0
        )

        # These two fields for source copy
        fields["src_copy_indices"] = SequenceLabelField(
            labels=list_data["src_copy_indices"],
            sequence_field=fields["tgt_tokens"],
            label_namespace="source_copy_target_tags",
        )

        fields["src_copy_map"] = AdjacencyField(
            indices=list_data["src_copy_map"],
            sequence_field=TextField(
                [
                    Token(x) for x in list_data["src_copy_vocab"].get_special_tok_list() + list_data["src_tokens"]
                ],
                None
            ),
            padding_value=0
        )

        # These two fields are used in biaffine parser
        fields["head_tags"] = SequenceLabelField(
            labels=list_data["head_tags"],
            sequence_field=fields["tgt_tokens"],
            label_namespace="head_tags",
            strip_sentence_symbols=True
        )

        fields["head_indices"] = SequenceLabelField(
            labels=list_data["head_indices"],
            sequence_field=fields["tgt_tokens"],
            label_namespace="head_index_tags",
            strip_sentence_symbols=True
        )

        # Metadata fields, good for debugging
        fields["src_tokens_str"] = MetadataField(
            list_data["src_tokens"]
        )

        fields["tgt_tokens_str"] = MetadataField(
            list_data.get("tgt_tokens", [])
        )

        fields["src_copy_vocab"] = MetadataField(
            list_data["src_copy_vocab"]
        )

        fields["pos_tag_lut"] = MetadataField(
            list_data["pos_tag_lut"]
        )

        fields["amr"] = MetadataField(
            amr
        )

        return Instance(fields)


