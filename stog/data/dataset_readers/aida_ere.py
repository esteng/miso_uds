from typing import Dict
import logging

from overrides import overrides

from stog.utils.file import cached_path
from stog.utils.tqdm import Tqdm
from stog.data.instance import Instance
from stog.data.tokenizers import WordTokenizer
from stog.data.tokenizers.word_splitter import JustSpacesWordSplitter
from stog.data.dataset_readers.dataset_reader import DatasetReader
from stog.data.token_indexers import TokenIndexer, SingleIdTokenIndexer
from stog.data.fields import (TextField,
                              MetadataField,
                              SequenceLabelField,
                              SpanField,
                              ListField)

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


@DatasetReader.register("aida")
class AidaEreDatasetReader(DatasetReader):
    """
    Reads an LTF file and converts it into a ``Dataset`` suitable for training a relation/event model.
    LTF files are pre-tokenized and segmented.

    Parameters
    ----------
    tokens_per_instance : ``int``, optional (default=``None``)
        If this is ``None``, we will have each training instance be a single sentence.  If this is
        not ``None``, we will instead take all sentences, including their start and stop tokens,
        line them up, and split the tokens into groups of this number, for more efficient training
        of the language model.
    token_indexers : ``Dict[str, TokenIndexer]``, optional (default=``{"tokens": SingleIdTokenIndexer()}``)
        We use this to define the input representation for the text.  See :class:`TokenIndexer`.
        Note that the `output` representation will always be single token IDs - if you've specified
        a ``SingleIdTokenIndexer`` here, we use the first one you specify.  Otherwise, we create
        one with default parameters.
    """
    def __init__(self,
                 tokens_per_instance: int = None,
                 token_indexers: Dict[str, TokenIndexer] = None,
                 lazy: bool = False) -> None:
        super().__init__(lazy)
        self._token_indexers = token_indexers or {"tokens": SingleIdTokenIndexer()}
        self._tokens_per_instance = tokens_per_instance
        self._tokenizer = WordTokenizer(word_splitter=JustSpacesWordSplitter())  # no-op tokenizer, since LTF is pre-tokenized

        # No matter how you want to represent the input, we'll always represent the output as a
        # single token id.  This code lets you concatenate word embeddings with character-level encoders.
        self._output_indexer: Dict[str, TokenIndexer] = None
        for name, indexer in self._token_indexers.items():
            if isinstance(indexer, SingleIdTokenIndexer):
                self._output_indexer = {name: indexer}
                break
        else:
            self._output_indexer = {"tokens": SingleIdTokenIndexer()}

    def _read_ltf(self, file_path):
        """
        Reads an LTF file and extracts its text and corresponding annotations.
        """
        segments = self._get_segments(file_path)
        provenance = self._file2provenance(file_path)

        instances = []
        for seg in segments:
            instances.append(self._segment2instance(seg, provenance))

        return instances

    def _file2provenance(self, datafile):
        import os
        return os.path.basename(datafile).split(".")[0]

    def _get_segments(self, datafile):
        import xml.etree.ElementTree as ET

        tree = ET.parse(datafile)
        root = tree.getroot()
        segs = [child for child in root[0][0]]

        return segs

    def _segment2instance(self, segment, provenance):
        # train at the segment/sentence level
        segment_id = segment.attrib['id']
        segment_text = segment[0].text
        segment_tokens = " ".join([tok.text for tok in segment[1:]])
        segment_tokens = self._tokenizer.tokenize(segment_tokens)  # convert `str` into `Token`
        packet = {"input_tokens": TextField(segment_tokens, self._token_indexers)}

        packet.update({"metadata": MetadataField({"text": segment_text,  # untokenized; not used but passed through just in case it's needed downstream
                                                  "provenance": provenance,
                                                  "segment_id": segment_id})})

        # TODO(sethebner): get entity spans, relation/event spans from annotation files
        rel_spans = [[0,0]]
        evt_spans = [[0,0]]
        ent_spans = [[0,0]]

        packet.update({"entity_spans": ListField([SpanField(i,j,packet["input_tokens"]) for (i,j) in ent_spans]),
                       "relation_spans": ListField([SpanField(i,j,packet["input_tokens"]) for (i,j) in rel_spans]),
                       "event_spans": ListField([SpanField(i,j,packet["input_tokens"]) for (i,j) in evt_spans])})

        # TODO(sethebner): typing information?

        return packet

    @overrides
    def _read(self, file_path: str):
        # gather files
        import os
        if os.path.isdir(file_path):
            file_paths = os.listdir(file_path)
        else:
            import glob
            file_paths = glob.glob(file_path)  # `file_path` may include wildcard characters

        file_paths = [f for f in file_paths if f.endswith(".ltf.xml")]  # filter

        for fp in file_paths:
            #fp = cached_path(fp)  # TODO(sethebner): LTF files aren't downloadable, so remove this?
            instances = self._read_ltf(fp)
            # TODO(sethebner): bucket segments by length? or batch in order they appear in document?
            for instance in instances:
                yield Instance(instance)

    @overrides
    def text_to_instance(self, sentence: str) -> Instance:  # type: ignore
        # pylint: disable=arguments-differ
        input_field = TextField(sentence, self._token_indexers)

        # TODO(sethebner): text metadata (segment id, provenance, etc.)?
        return Instance({'input_tokens': input_field})
