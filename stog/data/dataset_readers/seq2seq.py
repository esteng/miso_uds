from typing import Dict
import logging

from overrides import overrides

from stog.utils.checks import ConfigurationError
from stog.utils.file import cached_path
from stog.utils.string import START_SYMBOL, END_SYMBOL
from stog.data.dataset_readers.dataset_reader import DatasetReader
from stog.data.fields import TextField
from stog.data.instance import Instance
from stog.data.tokenizers import Token, Tokenizer, WordTokenizer
from stog.data.token_indexers import TokenIndexer, SingleIdTokenIndexer

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


@DatasetReader.register("seq2seq")
class Seq2SeqDatasetReader(DatasetReader):
    """
    Read a tsv file containing paired sequences, and create a dataset suitable for a
    ``SimpleSeq2Seq`` model, or any model with a matching API.

    Expected format for each input line: <source_sequence_string>\t<target_sequence_string>

    The output of ``read`` is a list of ``Instance`` s with the fields:
        source_tokens: ``TextField`` and
        target_tokens: ``TextField``

    `START_SYMBOL` and `END_SYMBOL` tokens are added to the source and target sequences.

    Parameters
    ----------
    source_tokenizer : ``Tokenizer``, optional
        Tokenizer to use to split the input sequences into words or other kinds of tokens. Defaults
        to ``WordTokenizer()``.
    target_tokenizer : ``Tokenizer``, optional
        Tokenizer to use to split the output sequences (during training) into words or other kinds
        of tokens. Defaults to ``source_tokenizer``.
    source_token_indexers : ``Dict[str, TokenIndexer]``, optional
        Indexers used to define input (source side) token representations. Defaults to
        ``{"tokens": SingleIdTokenIndexer()}``.
    target_token_indexers : ``Dict[str, TokenIndexer]``, optional
        Indexers used to define output (target side) token representations. Defaults to
        ``source_token_indexers``.
    source_add_start_token : bool, (optional, default=True)
        Whether or not to add `START_SYMBOL` to the beginning of the source sequence.
    """
    def __init__(self,
                 token_indexers: Dict[str, TokenIndexer] = None,
                 source_add_start_token : bool = False,
                 lazy: bool = False) -> None:
        super().__init__(lazy)
        self._token_indexers = token_indexers
        self._source_add_start_token = source_add_start_token

    @overrides
    def _read(self, file_path):
        with open(cached_path(file_path), "r") as data_file:
            logger.info("Reading instances from lines in file at: %s", file_path)
            for line_num, line in enumerate(data_file):
                line = line.strip("\n")

                if not line:
                    continue

                line_parts = line.split('\t')
                if len(line_parts) != 2:
                    raise ConfigurationError("Invalid line format: %s (line number %d)" % (line, line_num + 1))
                source_sequence, target_sequence = line_parts
                yield self.text_to_instance(source_sequence, target_sequence)

    @overrides
    def text_to_instance(self, source_string: str, target_string: str = None) -> Instance:  # type: ignore
        # pylint: disable=arguments-differ
        #tokenized_source = self._source_tokenizer.tokenize(source_string)
        tokenized_source = [Token(x) for x in source_string.split(' ')]
        if self._source_add_start_token:
            tokenized_source.insert(0, Token(START_SYMBOL))
        tokenized_source.append(Token(END_SYMBOL))
        
        

        tokenized_target = [Token(x) for x in target_string.split(" ")]
        tokenized_target.insert(0, Token(START_SYMBOL))
        tokenized_target.append(Token(END_SYMBOL))
        
        source_field = TextField(
            tokenized_source,
            token_indexers={k: v for k, v in self._token_indexers.items() if 'encoder' in k}
        )
        target_field = TextField(
            tokenized_target,
            token_indexers={k: v for k, v in self._token_indexers.items() if 'decoder' in k}
        )

        fields: Dict[str, Field] = {}
        fields["src_tokens"] = source_field
        fields["amr_tokens"] = target_field
        
        return Instance(fields)
