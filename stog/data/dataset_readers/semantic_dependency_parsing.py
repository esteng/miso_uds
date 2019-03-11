from typing import Dict, List, Tuple
import logging
from overrides import overrides

from stog.utils.file import cached_path
from stog.data.dataset_readers.dataset_reader import DatasetReader
from stog.data.fields import AdjacencyField, MetadataField, SequenceLabelField
from stog.data.fields import Field, TextField
from stog.data.token_indexers import SingleIdTokenIndexer, TokenIndexer
from stog.data.tokenizers import Token
from stog.data.instance import Instance
from stog.data.dataset_readers.semantic_dependencies.sdp import SDPGraph
from stog.utils.string import START_SYMBOL, END_SYMBOL
from stog.data.tokenizers.bert_tokenizer import AMRBertTokenizer
from stog.data.fields import TextField, SpanField, SequenceLabelField, ListField, MetadataField, Field, AdjacencyField, ArrayField

logger = logging.getLogger(__name__) # pylint: disable=invalid-name

#FIELDS = ["id", "form", "lemma", "pos", "head", "deprel", "top", "pred", "frame"]
FIELDS = ["id", "form", "lemma", "pos", "top", "pred", "frame"]

def parse_sentence(sentence_blob: str) -> Tuple[List[Dict[str, str]], List[Tuple[int, int]], List[str]]:
    """
    Parses a chunk of text in the SemEval SDP format.

    Each word in the sentence is returned as a dictionary with the following
    format:
    'id': '1',
    'form': 'Pierre',
    'lemma': 'Pierre',
    'pos': 'NNP',
    'head': '2',   # Note that this is the `syntactic` head.
    'deprel': 'nn',
    'top': '-',
    'pred': '+',
    'frame': 'named:x-c'

    Along with a list of arcs and their corresponding tags. Note that
    in semantic dependency parsing words can have more than one head
    (it is not a tree), meaning that the list of arcs and tags are
    not tied to the length of the sentence.
    """
    annotated_sentence = []
    arc_indices = []
    arc_tags = []
    predicates = []

    lines = [line.split("\t") for line in sentence_blob.split("\n")
             if line and not line.strip().startswith("#")]
    for line_idx, line in enumerate(lines):
        annotated_token = {k:v for k, v in zip(FIELDS, line)}
        if annotated_token['pred'] == "+":
            predicates.append(line_idx)
        annotated_sentence.append(annotated_token)

    for line_idx, line in enumerate(lines):
        for predicate_idx, arg in enumerate(line[len(FIELDS):]):
            if arg != "_":
                arc_indices.append((line_idx, predicates[predicate_idx]))
                arc_tags.append(arg)
    return annotated_sentence, arc_indices, arc_tags


def lazy_parse(text: str):
    for sentence in text.split("\n\n"):
        if sentence:
            yield parse_sentence(sentence)


@DatasetReader.register("semantic_dependencies")
class SemanticDependenciesDatasetReader(DatasetReader):
    """
    Reads a file in the SemEval 2015 Task 18 (Broad-coverage Semantic Dependency Parsing)
    format.

    Parameters
    ----------
    token_indexers : ``Dict[str, TokenIndexer]``, optional (default=``{"tokens": SingleIdTokenIndexer()}``)
        The token indexers to be applied to the words TextField.
    """
    def __init__(self,
                 token_indexers: Dict[str, TokenIndexer] = None,
                 word_splitter = None,
                 lazy: bool = False,
                 evaluation: bool = False) -> None:
        super().__init__(lazy)
        self._token_indexers = token_indexers or {'tokens': SingleIdTokenIndexer()}
        if word_splitter is not None:
            self._word_splitter = AMRBertTokenizer.from_pretrained(
                word_splitter, do_lower_case=False)
        else:
            self._word_splitter = None
        
        self._evaluation = evaluation

        self._number_bert_ids = 0
        self._number_bert_oov_ids = 0
        self._number_non_oov_pos_tags = 0
        self._number_pos_tags = 0

    @overrides
    def _read(self, file_path: str):
        # if `file_path` is a URL, redirect to the cache
        file_path = cached_path(file_path)

        logger.info("Reading semantic dependency parsing data from: %s", file_path)

        with open(file_path) as sdp_file:
            for annotated_sentence, directed_arc_indices, arc_tags in lazy_parse(sdp_file.read()):
                # If there are no arc indices, skip this instance.
                if not directed_arc_indices:
                    continue
                tokens = [word["form"] for word in annotated_sentence]
                pos_tags = [word["pos"] for word in annotated_sentence]
                yield self.text_to_instance(annotated_sentence, directed_arc_indices, arc_tags)

    @overrides
    def text_to_instance(self, # type: ignore
                         annotated_sentence,
                         arc_indices: List[Tuple[int, int]] = None,
                         arc_tags: List[str] = None) -> Instance:
        # pylint: disable=arguments-differ
        SDP_graph = SDPGraph(annotated_sentence, arc_indices, arc_tags)
        list_data = SDP_graph.get_list_data(START_SYMBOL, END_SYMBOL, self._word_splitter)
        
        fields = {}
        # These four fields are used for seq2seq model and target side self copy
        fields["src_tokens"] = TextField(
            tokens=[Token(x) for x in list_data["src_tokens"]],
            token_indexers={k: v for k, v in self._token_indexers.items() if 'encoder' in k}
        )

        if list_data['src_token_ids'] is not None:
            fields['src_token_ids'] = ArrayField(list_data['src_token_ids'])
            self._number_bert_ids += len(list_data['src_token_ids'])
            self._number_bert_oov_ids += len(
                [bert_id for bert_id in list_data['src_token_ids'] if bert_id == 100])

        if list_data['src_token_subword_index'] is not None:
            fields['src_token_subword_index'] = ArrayField(
                list_data['src_token_subword_index'])

        fields["src_must_copy_tags"] = SequenceLabelField(
            labels=list_data["src_must_copy_tags"],
            sequence_field=fields["src_tokens"],
            label_namespace="must_copy_tags"
        )

        fields["tgt_tokens"] = TextField(
            tokens=[Token(x) for x in list_data["tgt_tokens"]],
            token_indexers={k: v for k, v in self._token_indexers.items() if 'decoder' in k}
        )

        if list_data["src_pos_tags"] is not None:
            fields["src_pos_tags"] = SequenceLabelField(
                labels=list_data["src_pos_tags"],
                sequence_field=fields["src_tokens"],
                label_namespace="pos_tags"
            )

        if list_data["tgt_pos_tags"] is not None:
            fields["tgt_pos_tags"] = SequenceLabelField(
                labels=list_data["tgt_pos_tags"],
                sequence_field=fields["tgt_tokens"],
                label_namespace="pos_tags"
            )

            self._number_pos_tags += len(list_data['tgt_pos_tags'])
            self._number_non_oov_pos_tags += len(
                [tag for tag in list_data['tgt_pos_tags'] if tag != '@@UNKNOWN@@'])

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

        if self._evaluation:
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

            fields["tag_lut"] = MetadataField(
                dict(pos=list_data["pos_tag_lut"])
            )

            fields["source_copy_invalid_ids"] = MetadataField(
                list_data['src_copy_invalid_ids']
            )

            fields["sdp"] = MetadataField(
               annotated_sentence 
            )

        return Instance(fields)
