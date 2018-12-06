
from typing import Dict, List, Tuple
import logging
import os

from overrides import overrides
# NLTK is so performance orientated (ha ha) that they have lazy imports. Why? Who knows.
from stog.data.amr import AMRTree
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

        stacked_lines = []
        sentence_conter = 0
        sentence_id = ""
        sentence_text = ""
        with open(file_path, 'r') as f:
            if self._skip_first_line:
                f.readline()
            for line in f:
                if len(line) <= 1 and len(stacked_lines) > 0:
                    sequence = ""
                    for line in stacked_lines:
                        if line[0] != "#":
                            sequence += line.strip()
                            sequence += " "
                        elif "# ::id" in line:
                            sentence_id = line.split(" ")[2]
                        elif "# ::snt" in line:
                            sentence_text = line.strip().split("snt")[-1]

                    tree = AMRTree(sequence.strip(), sentence_text)
                    stacked_lines = []
                    sentence_conter += 1
                    yield self.text_to_instance(tree)
                else:
                    stacked_lines.append(line)

    @overrides
    def text_to_instance(self, tree : AMRTree) -> Instance:
        # pylint: disable=arguments-differ

        fields: Dict[str, Field] = {}

        # These four fields are used for seq2seq model and target side self copy
        fields["src_tokens"] = TextField(
            tokens=[Token(x) for x in tree.get_original_sentence().split()],
            token_indexers={k: v for k, v in self._token_indexers.items() if 'encoder' in k}
        )

        fields["amr_tokens"] = TextField(
            tokens=[Token(x) for x in tree.get_instance(START_SYMBOL, END_SYMBOL)],
            token_indexers={k: v for k, v in self._token_indexers.items() if 'decoder' in k}
        )

        fields["coref_index"] = SequenceLabelField(
            labels=tree.get_coref(bos=True, eos=True),
            sequence_field=fields["amr_tokens"],
            label_namespace="coref_tags",
        )

        fields["coref_map"] = AdjacencyField(
            indices=tree.get_coref_map(bos=True, eos=True),
            sequence_field=fields["amr_tokens"],
            padding_value=0
        )
        
        # These two fields are used in biaffine parser
        fields["head_tags"] = SequenceLabelField(
            labels=tree.get_relation(),
            sequence_field=fields["amr_tokens"],
            label_namespace="head_tags",
            strip_sentence_symbols=True
        )

        fields["head_indices"] = SequenceLabelField(
            labels=tree.get_parent(),
            sequence_field=fields["amr_tokens"],
            label_namespace="head_index_tags",
            strip_sentence_symbols=True
        )

        return Instance(fields)


