from typing import (Dict, List, Tuple, Union, Iterator, NewType)
import logging

from overrides import overrides

from stog.utils.file import cached_path
from stog.utils.tqdm import Tqdm
from stog.data.instance import Instance
from stog.data.tokenizers import WordTokenizer
from stog.data.tokenizers.word_splitter import JustSpacesWordSplitter
from stog.data.dataset_readers.dataset_reader import DatasetReader
from stog.data.token_indexers import TokenIndexer, SingleIdTokenIndexer
from stog.data.fields import (Field,
                              TextField,
                              MetadataField,
                              SequenceLabelField,
                              SpanField,
                              LabelField,
                              ListField)

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name

SegmentType = NewType('SegmentType', object)
ProvenanceType = NewType('ProvenanceType', str)
FilepathType = NewType('FilepathType', str)

NO_OFFSET_FOUND = -1

def contains(x: Union[int, float], y: Union[int, float]) -> bool:
    """
    Returns `True` if span `y` is contained in span `x` (inclusive bounds).
    """
    a, b = x
    c, d = y
    if not ((a <= b) and (c <= d)):
        raise ValueError(f"Left endpoint must be given before right endpoint: [{a}, {b}] does not contain [{c}, {d}]")
    return (a <= c) and (b >= d)

def str2int(s: str, default: int = 0) -> int:
    try:
        x = int(s)
        return x
    except ValueError:
        return default

def offset_str2int(x: str) -> int:
    return str2int(x, default=NO_OFFSET_FOUND)

class Mention(object):
    def __init__(self, start_char: int, end_char: int, kind: str, type: str):
        self.start_char = start_char
        self.end_char = end_char
        self.kind = kind  # entity, relation, event
        self.type = type  # ontology type

        assert self.kind in ["ent", "rel", "evt"]

        self.start_token: int = None
        self.end_token: int = None


@DatasetReader.register("aida")
class AidaEreDatasetReader(DatasetReader):
    """
    Reads an LTF file and converts it into a ``Dataset`` suitable for training a relation/event extraction model.
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

    def _read_ltf(self, file_path: FilepathType) -> List[Instance]:
        """
        Reads an LTF file and extracts its text and corresponding annotations.
        """
        segments: List[SegmentType] = self._get_segments(file_path)
        provenance: ProvenanceType = self._file2provenance(file_path)

        instances = []
        for seg in segments:
            instances.append(self._segment2instance(seg, provenance))

        return instances

    def _file2provenance(self, datafile: FilepathType) -> ProvenanceType:
        import os
        return os.path.basename(datafile).split(".")[0]

    def _get_segments(self, datafile: FilepathType) -> List[SegmentType]:
        import xml.etree.ElementTree as ET

        tree = ET.parse(datafile)
        root = tree.getroot()
        segs = [child for child in root[0][0]]

        return segs

    def _segment2instance(self, segment: SegmentType, provenance: ProvenanceType) -> Instance:
        """
        Create an Instance containing metadata and annotations for the given segment.
        Instances are at the segment/sentence level.
        """

        # Get metadata.
        segment_id = segment.attrib['id']
        segment_text = segment[0].text
        segment_tokens = " ".join([tok.text for tok in segment[1:]])
        segment_tokens = self._tokenizer.tokenize(segment_tokens)  # convert `str` into `Token`
        packet = {"input_tokens": TextField(segment_tokens, self._token_indexers)}

        packet.update({"metadata": MetadataField({"text": segment_text,  # untokenized; not used but passed through just in case it's needed downstream
                                                  "provenance": provenance,
                                                  "segment_id": segment_id})})

        # Get annotation information.
        mentions = self._segment2mentions(segment, provenance)

        ent_spans = [(mention.start_token, mention.end_token) for mention in mentions["ent"]]
        rel_spans = [(mention.start_token, mention.end_token) for mention in mentions["rel"]]
        evt_spans = [(mention.start_token, mention.end_token) for mention in mentions["evt"]]

        ent_types = [mention.type for mention in mentions["ent"]]
        rel_types = [mention.type for mention in mentions["rel"]]
        evt_types = [mention.type for mention in mentions["evt"]]

        assert len(ent_spans) == len(ent_types)
        assert len(rel_spans) == len(rel_types)
        assert len(evt_spans) == len(evt_types)

        NO_MENTIONS = ListField([SpanField(-1,-1,packet["input_tokens"])])
        packet.update({"entity_spans": ListField([SpanField(i,j,packet["input_tokens"]) for (i,j) in ent_spans]) if ent_spans else NO_MENTIONS,
                       "relation_spans": ListField([SpanField(i,j,packet["input_tokens"]) for (i,j) in rel_spans]) if rel_spans else NO_MENTIONS,
                       "event_spans": ListField([SpanField(i,j,packet["input_tokens"]) for (i,j) in evt_spans]) if evt_spans else NO_MENTIONS})

        NO_ENTITY_TYPES = ListField([LabelField("NO_TYPE", label_namespace="entity_type_labels")])
        NO_RELATION_TYPES = ListField([LabelField("NO_TYPE", label_namespace="entity_type_labels")])
        NO_EVENT_TYPES = ListField([LabelField("NO_TYPE", label_namespace="entity_type_labels")])
        packet.update({"entity_types": ListField([LabelField(type, label_namespace="entity_type_labels") for type in ent_types]) if ent_types else NO_ENTITY_TYPES,
                       "relation_types": ListField([LabelField(type, label_namespace="relation_type_labels") for type in rel_types]) if rel_types else NO_RELATION_TYPES,
                       "event_types": ListField([LabelField(type, label_namespace="event_type_labels") for type in evt_types]) if evt_types else NO_EVENT_TYPES})

        return packet

    def _segment2mentions(self, segment: SegmentType, provenance: ProvenanceType) -> Dict[str, List[Mention]]:
        """
        Gather all annotated mentions for a given segment.
        """
        segment_char_start = int(segment.attrib["start_char"])  # relative to document
        segment_char_end = int(segment.attrib["end_char"])  # relative to document

        mentions = dict(ent=[], rel=[], evt=[])

        for kind in self._annotations[provenance]:
            for mention in self._annotations[provenance][kind]:
                if contains((segment_char_start, segment_char_end), (mention.start_char, mention.end_char)):
                    mentions[kind].append(mention)

        # adjust mention char offsets -> segment token offsets
        # e.g., entity mention: document characters [103, 110] -> segment tokens [0, 1]
        segment_tokens_char_offsets = self._get_tokens_char_offsets(segment)
        for mention_kind in mentions:
            for i,mention in enumerate(mentions[mention_kind]):
                mentions[mention_kind][i] = self._recast_mention_offsets(mention, segment_tokens_char_offsets)

        return mentions

    def _get_tokens_char_offsets(self, segment: SegmentType) -> List[Tuple[int, int]]:
        """
        Gather character offsets for each token in a segment.
        """
        segment_char_offsets = []
        tokens = segment[1:]
        for i,token in enumerate(tokens):
            token_idx = int(token.attrib["id"].split("-")[-1])  # relative to segment
            assert i == token_idx

            token_char_start = int(token.attrib["start_char"])  # relative to document
            token_char_end = int(token.attrib["end_char"])  # relative to document
            segment_char_offsets.append((token_char_start, token_char_end))

        return segment_char_offsets

    def _recast_mention_offsets(self, mention: Mention, segment_tokens_char_offsets: List[Tuple[int, int]]) -> Mention:
        """
        Convert character offsets into token indexes.
        """
        token_indexes = self._char_span2token_idx((mention.start_char, mention.end_char), segment_tokens_char_offsets)
        mention.start_token, mention.end_token = token_indexes[0], token_indexes[-1]
        return mention

    def _char_span2token_idx(self, mention_char_offsets: Tuple[int, int], segment_tokens_char_offsets: List[Tuple[int, int]]) -> List[int]:
        """
        Find which token(s) in the segment correspond to the (annotated) mention.
        """
        hits = []
        mention_char_start, mention_char_end = mention_char_offsets
        for i,(token_char_start, token_char_end) in enumerate(segment_tokens_char_offsets):
            if contains(mention_char_offsets, (token_char_start, token_char_end)) or contains((token_char_start, token_char_end), mention_char_offsets):
                hits.append(i)
        assert len(hits) >= 1, f"{mention_char_offsets}, {segment_tokens_char_offsets}"
        return hits

    @overrides
    def _read(self, file_path: FilepathType) -> Iterator[Instance]:
        # gather files
        import os
        if os.path.isdir(file_path):
            file_paths = os.listdir(file_path)
        else:
            import glob
            file_paths = glob.glob(file_path)  # `file_path` may include wildcard characters

        file_paths = [f for f in file_paths if f.endswith(".ltf.xml")]  # filter

        # aggregate mentions by provenance and kind once before doing work for all segments
        provenances = [self._file2provenance(file_path) for file_path in file_paths]
        self._annotations = self._aggregate_annotations(provenances)

        for fp in file_paths:
            instances = self._read_ltf(fp)
            # TODO(sethebner): bucket segments by length? or batch in order they appear in document?
            for instance in instances:
                yield Instance(instance)

    def _aggregate_annotations(self, provenances: List[ProvenanceType]) -> Dict[ProvenanceType, Dict[str, List[Mention]]]:
        """
        Gather all annotations.
        """
        # annotations = {provenance : dict(ent=[], rel=[], evt=[]) for provenance in provenances}
        from collections import defaultdict
        annotations = defaultdict(lambda: dict(ent=[], rel=[], evt=[]))
        for topic in ["T101", "T102", "T103", "T105", "T106", "T107"]:
            for kind in ["ent", "rel", "evt"]:
                PROV = 3  # index of provenance in annotation mention
                CHAR_START = 4  # mention char start
                CHAR_END = 5  # mention char end
                TYPE = 8  # mention (ontology) type
                with open(f"/export/corpora/LDC/LDC2018E45/LDC2018E45_AIDA_Scenario_1_Seedling_Annotation_V6.0/data/{topic}/{topic}_{kind}_mentions.tab", "r", encoding="utf-8") as ann_f:
                    lines = ann_f.readlines()
                    lines = [line.strip().split('\t') for line in lines if line.strip().split('\t')[0] != 'tree_id']  # split on tabs, skip header line
                    for ann in lines:
                        provenance = ann[PROV]
                        if provenance not in provenances:
                            continue
                        mention_start_char = offset_str2int(ann[CHAR_START])
                        mention_end_char = offset_str2int(ann[CHAR_END])
                        type = ann[TYPE]

                        mention = Mention(start_char=mention_start_char, end_char=mention_end_char, kind=kind, type=type)
                        annotations[provenance][kind].append(mention)

        return annotations

    @overrides
    def text_to_instance(self, sentence: str) -> Instance:  # type: ignore
        # pylint: disable=arguments-differ
        input_field = TextField(sentence, self._token_indexers)

        # TODO(sethebner): text metadata (segment id, provenance, etc.)?
        return Instance({'input_tokens': input_field})
