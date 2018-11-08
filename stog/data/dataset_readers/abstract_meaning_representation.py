
from typing import Dict, List, Tuple
import logging
import os

from overrides import overrides
# NLTK is so performance orientated (ha ha) that they have lazy imports. Why? Who knows.
from stog.data.amr import AMRTree
from stog.utils.file import cached_path
from stog.data.dataset_readers.dataset_reader import DatasetReader
from stog.data.fields import TextField, SpanField, SequenceLabelField, ListField, MetadataField, Field
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
    """
    Reads constituency parses from the WSJ part of the Penn Tree Bank from the LDC.
    This ``DatasetReader`` is designed for use with a span labelling model, so
    it enumerates all possible spans in the sentence and returns them, along with gold
    labels for the relevant spans present in a gold tree, if provided.

    Parameters
    ----------
    token_indexers : ``Dict[str, TokenIndexer]``, optional (default=``{"tokens": SingleIdTokenIndexer()}``)
        We use this to define the input representation for the text.  See :class:`TokenIndexer`.
        Note that the `output` tags will always correspond to single token IDs based on how they
        are pre-tokenised in the data file.
    use_pos_tags : ``bool``, optional, (default = ``True``)
        Whether or not the instance should contain gold POS tags
        as a field.
    lazy : ``bool``, optional, (default = ``False``)
        Whether or not instances can be consumed lazily.
    label_namespace_prefix : ``str``, optional, (default = ``""``)
        Prefix used for the label namespace.  The ``span_labels`` will use
        namespace ``label_namespace_prefix + 'labels'``, and if using POS
        tags their namespace is ``label_namespace_prefix + pos_label_namespace``.
    pos_label_namespace : ``str``, optional, (default = ``"pos"``)
        The POS tag namespace is ``label_namespace_prefix + pos_label_namespace``.
    """
    def __init__(self,
                 token_indexers: Dict[str, TokenIndexer] = None,
                 use_pos_tags: bool = True,
                 lazy: bool = False,
                 label_namespace_prefix: str = "",
                 pos_label_namespace: str = "pos",
                 word_splitter = None) -> None:
        super().__init__(lazy=lazy)
        self._token_indexers = token_indexers or {'tokens': SingleIdTokenIndexer()}
        self._use_pos_tags = use_pos_tags
        self._label_namespace_prefix = label_namespace_prefix
        self._pos_label_namespace = pos_label_namespace
        self._word_splitter = word_splitter or SpacyWordSplitter()

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

                    tree = AMRTree(sequence.strip())
                    stacked_lines = []
                    sentence_conter += 1
                    yield self.text_to_instance(tree, sentence_text, sentence_id)
                else:
                    stacked_lines.append(line)

    @overrides
    def text_to_instance(self, # type: ignore
                         tree : AMRTree,
                         sentence_text : str,
                         sentence_id : str) -> Instance:
        """
        We take `pre-tokenized` input here, because we don't have a tokenizer in this class.

        Parameters
        ----------
        tokens : ``List[str]``, required.
            The tokens in a given sentence.
        pos_tags ``List[str]``, optional, (default = None).
            The POS tags for the words in the sentence.
        gold_tree : ``Tree``, optional (default = None).
            The gold parse tree to create span labels from.

        Returns
        -------
        An ``Instance`` containing the following fields:
            tokens : ``TextField``
                The tokens in the sentence.
            pos_tags : ``SequenceLabelField``
                The POS tags of the words in the sentence.
                Only returned if ``use_pos_tags`` is ``True``
            spans : ``ListField[SpanField]``
                A ListField containing all possible subspans of the
                sentence.
            span_labels : ``SequenceLabelField``, optional.
                The constiutency tags for each of the possible spans, with
                respect to a gold parse tree. If a span is not contained
                within the tree, a span will have a ``NO-LABEL`` label.
            gold_tree : ``MetadataField(Tree)``
                The gold NLTK parse tree for use in evaluation.
        """
        # pylint: disable=arguments-differ
        fields: Dict[str, Field] = {}
        # TODO: Xutai
        tokens = TextField(
            [Token(START_SYMBOL)] + [Token(x) for x in tree.get_instance()] + [Token(END_SYMBOL)],
            token_indexers={k: v for k, v in self._token_indexers.items() if 'decoder' in k}
        )
        fields["amr_tokens"] = tokens

        # fields["head_tags"] = SequenceLabelField(tree.get_relation(),
        #                                          tokens,
        #                                          label_namespace="head_tags")
        # fields["head_indices"] = SequenceLabelField(tree.get_parent(),
        #                                             tokens,
        #                                             label_namespace="head_index_tags")
        # fields["coref"] = SequenceLabelField(tree.get_coref(),
        #                                      tokens,
        #                                      label_namespace="coref_tags"
        #                                      )
        # TODO: Xutai
        fields["src_tokens"] = TextField(
            self._word_splitter.split_words(sentence_text),
            token_indexers={k: v for k, v in self._token_indexers.items() if 'encoder' in k}
        )

        return Instance(fields)


