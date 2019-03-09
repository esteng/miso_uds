from typing import Dict, Tuple, List
import logging

from overrides import overrides
from conllu.parser import parse_line, DEFAULT_FIELDS

from miso.utils.file import cached_path
from miso.data.dataset_readers.dataset_reader import DatasetReader
from miso.data.fields import Field, TextField, SequenceLabelField, MetadataField
from miso.data.instance import Instance
from miso.data.token_indexers import SingleIdTokenIndexer, TokenIndexer
from miso.data.tokenizers import Token

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


def lazy_parse(text: str, fields: Tuple = DEFAULT_FIELDS):
    for sentence in text.split("\n\n"):
        if sentence:
            yield [parse_line(line, fields)
                   for line in sentence.split("\n")
                   if line and not line.strip().startswith("#")]


@DatasetReader.register("universal_dependencies")
class UniversalDependenciesDatasetReader(DatasetReader):
    """
    Reads a file in the conllu Universal Dependencies format.

    Parameters
    ----------
    token_indexers : ``Dict[str, TokenIndexer]``, optional (default=``{"tokens": SingleIdTokenIndexer()}``)
        The token indexers to be applied to the words TextField.
    use_language_specific_pos : ``bool``, optional (default = False)
        Whether to use UD POS tags, or to use the language specific POS tags
        provided in the conllu format.
    """
    def __init__(self,
                 token_indexers: Dict[str, TokenIndexer] = None,
                 use_language_specific_pos: bool = False,
                 lazy: bool = False) -> None:
        super().__init__(lazy)
        self._token_indexers = token_indexers or {'tokens': SingleIdTokenIndexer()}
        self.use_language_specific_pos = use_language_specific_pos

    @overrides
    def _read(self, file_path: str):
        # if `file_path` is a URL, redirect to the cache
        file_path = cached_path(file_path)

        with open(file_path, 'r') as conllu_file:
            logger.info("Reading UD instances from conllu dataset at: %s", file_path)
            for annotation in  lazy_parse(conllu_file.read()):
                # CoNLLU annotations sometimes add back in words that have been elided
                # in the original sentence; we remove these, as we're just predicting
                # dependencies for the original sentence.
                # We filter by None here as elided words have a non-integer word id,
                # and are replaced with None by the conllu python library.
                annotation = [x for x in annotation if x["id"] is not None]

                heads = [x["head"] for x in annotation]
                tags = [x["deprel"] for x in annotation]
                words = [x["form"] for x in annotation]
                if self.use_language_specific_pos:
                    pos_tags = [x["xpostag"] for x in annotation]
                else:
                    pos_tags = [x["upostag"] for x in annotation]
                yield self.text_to_instance(words, pos_tags, list(zip(tags, heads)))

    @overrides
    def text_to_instance(self,  # type: ignore
                         words: List[str],
                         upos_tags: List[str],
                         dependencies: List[Tuple[str, int]] = None) -> Instance:
        # pylint: disable=arguments-differ
        """
        Parameters
        ----------
        words : ``List[str]``, required.
            The words in the sentence to be encoded.
        upos_tags : ``List[str]``, required.
            The universal dependencies POS tags for each word.
        dependencies ``List[Tuple[str, int]]``, optional (default = None)
            A list of  (head tag, head index) tuples. Indices are 1 indexed,
            meaning an index of 0 corresponds to that word being the root of
            the dependency tree.

        Returns
        -------
        An instance containing words, upos tags, dependency head tags and head
        indices as fields.
        """
        fields: Dict[str, Field] = {}
        # TODO:Leave non-integer indexed token for now
        new_words = []
        new_upos_tags = []
        new_dependencies = []
        for i in range(len(dependencies)):
            if dependencies[i][1] is not None:
                new_words.append(words[i])
                new_upos_tags.append(upos_tags[i])
                new_dependencies.append(dependencies[i])

        dependencies = new_dependencies
        upos_tags = new_upos_tags
        words = new_words

        tokens = TextField([Token(w) for w in words], self._token_indexers)
        fields["words"] = tokens
        fields["pos_tags"] = SequenceLabelField(upos_tags, tokens, label_namespace="pos_tags")
        if dependencies is not None:
            # We don't want to expand the label namespace with an additional dummy token, so we'll
            # always give the 'ROOT_HEAD' token a label of 'root'.
            fields["head_tags"] = SequenceLabelField([x[0] for x in dependencies],
                                                     tokens,
                                                     label_namespace="head_tags")
            fields["head_indices"] = SequenceLabelField([int(x[1]) for x in dependencies],
                                                        tokens,
                                                        label_namespace="head_index_tags")

        fields["metadata"] = MetadataField({"words": words, "pos": upos_tags})
        return Instance(fields)
