from typing import Iterable, Iterator, Callable, Dict
import logging
import json 
import os
import sys
import pdb
from glob import glob 
from overrides import overrides

from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer
from allennlp.data.tokenizers import Tokenizer, Token
from allennlp.data.fields import Field, TextField, ArrayField, SequenceLabelField, MetadataField, AdjacencyField
from allennlp.data.instance import Instance
from allennlp.common.util import START_SYMBOL, END_SYMBOL


from miso.data.fields.continuous_label_field import ContinuousLabelField
from miso.data.dataset_readers.calflow_parsing.calflow_graph import CalFlowGraph
from miso.data.tokenizers import AMRBertTokenizer, AMRXLMRobertaTokenizer, MisoTokenizer

from task_oriented_dialogue_as_dataflow_synthesis.src.dataflow.core.io import load_jsonl_file
from task_oriented_dialogue_as_dataflow_synthesis.src.dataflow.core.dialogue import Dialogue, Turn, TurnId


logger = logging.getLogger(__name__)  # pylint: disable=invalid-name

@DatasetReader.register("smcalflow")
class CalFlowDatasetReader(DatasetReader):
    '''
    Dataset reader for CalFlow data
    '''
    def __init__(self,
                 source_token_indexers: Dict[str, TokenIndexer],
                 target_token_indexers: Dict[str, TokenIndexer],
                 generation_token_indexers: Dict[str, TokenIndexer],
                 tokenizer: MisoTokenizer = None, #AMRTransformerTokenizer,
                 evaluation: bool = False,
                 line_limit: int = None,
                 lazy: bool = False,
                 ) -> None:

        super().__init__(lazy=lazy)
        self.line_limit = line_limit

        self._source_token_indexers = source_token_indexers
        self._target_token_indexers = target_token_indexers
        self._generation_token_indexers = generation_token_indexers
        self._edge_type_indexers = {"edge_types": SingleIdTokenIndexer(namespace="edge_types")}
        self._tokenizer = tokenizer
        self._num_subtokens = 0
        self._num_subtoken_oovs = 0

        self.eval = evaluation

        self._number_bert_ids = 0
        self._number_bert_oov_ids = 0
        self._number_non_oov_pos_tags = 0
        self._number_pos_tags = 0
    
        self.over_len = 0

    def report_coverage(self):
        if self._number_bert_ids != 0:
            logger.info('BERT OOV  rate: {0:.4f} ({1}/{2})'.format(
                self._number_bert_oov_ids / self._number_bert_ids,
                self._number_bert_oov_ids, self._number_bert_ids
            ))
        if self._number_non_oov_pos_tags != 0:
            logger.info('POS tag coverage: {0:.4f} ({1}/{2})'.format(
                self._number_non_oov_pos_tags / self._number_pos_tags,
                self._number_non_oov_pos_tags, self._number_pos_tags
            ))

    def set_evaluation(self):
        self.eval = True
    
    @overrides
    def _read(self, path: str) -> Iterable[Instance]:

        logger.info("Reading calflow data from: %s", path)
        skipped = 0
        for dialogue in load_jsonl_file(path, Dialogue):
            for turn in dialogue.turns:
                pdb.set_trace() 
                t2i = self.text_to_instance(turn)
                if t2i is None:
                    skipped += 1
                    continue
                if self.line_limit is not None:
                    if i > self.line_limit:
                        break
                yield t2i


    @overrides
    def text_to_instance(self, graph, do_print=False) -> Instance:
        """
        Does bulk of work converting a graph to an Instance of Fields 
        """
        # pylint: disable=arguments-differ

        fields: Dict[str, Field] = {}

        max_tgt_length = None if self.eval else 60
        d = CalFlowGraph(graph)
        list_data = d.get_list_data(
             bos=START_SYMBOL, 
             eos=END_SYMBOL, 
             bert_tokenizer = self._tokenizer, 
             max_tgt_length = max_tgt_length) 
        if list_data is None:
            return None

        # These four fields are used for seq2seq model and target side self copy
        fields["source_tokens"] = TextField(
            tokens=[Token(x) for x in list_data["src_tokens"]],
            token_indexers=self._source_token_indexers
        )

        if list_data['src_token_ids'] is not None:
            fields['source_subtoken_ids'] = ArrayField(list_data['src_token_ids'])
            self._number_bert_ids += len(list_data['src_token_ids'])
            self._number_bert_oov_ids += len(
                [bert_id for bert_id in list_data['src_token_ids'] if bert_id == 100])

        if list_data['src_token_subword_index'] is not None:
            fields['source_token_recovery_matrix'] = ArrayField(list_data['src_token_subword_index'])

        # Target-side input.
        # (exclude the last one <EOS>.)
        fields["target_tokens"] = TextField(
            tokens=[Token(x) for x in list_data["tgt_tokens"][:-1]],
            token_indexers=self._target_token_indexers
        )

        if len(list_data['tgt_tokens']) > 60:
            self.over_len += 1

        

        fields["source_pos_tags"] = SequenceLabelField(
            labels=list_data["src_pos_tags"],
            sequence_field=fields["source_tokens"],
            label_namespace="pos_tags"
        )

        fields["target_node_indices"] = SequenceLabelField(
            labels=list_data["tgt_indices"][:-1],
            sequence_field=fields["target_tokens"],
            label_namespace="node_indices",
        )

        # Target-side output.
        # Include <BOS> here because we want it in the generation vocabulary such that
        # at the inference starting stage, <BOS> can be correctly initialized.
        fields["generation_outputs"] = TextField(
            tokens=[Token(x) for x in list_data["tgt_tokens_to_generate"]],
            token_indexers=self._generation_token_indexers
        )

        fields["target_copy_indices"] = SequenceLabelField(
            labels=list_data["tgt_copy_indices"],
            sequence_field=fields["generation_outputs"],
            label_namespace="target_copy_indices",
        )

        fields["target_attention_map"] = AdjacencyField(  # TODO: replace it with ArrayField.
            indices=list_data["tgt_copy_map"],
            sequence_field=fields["generation_outputs"],
            padding_value=0
        )

        # These two fields for source copy

        fields["source_copy_indices"] = SequenceLabelField(
            labels=list_data["src_copy_indices"],
            sequence_field=fields["generation_outputs"],
            label_namespace="source_copy_indices",
        )

        fields["source_attention_map"] = AdjacencyField(  # TODO: replace it with ArrayField.
            indices=list_data["src_copy_map"],
            sequence_field=TextField(
                [Token(x) for x in list_data["src_copy_vocab"].get_special_tok_list() + list_data["src_tokens"]], None
            ),
            padding_value=0
        )
        #print(list_data['src_copy_indices']) 
        #print(list_data['src_copy_map']) 

        #print(f'over textfield {[Token(x) for x in list_data["src_copy_vocab"].get_special_tok_list() + list_data["src_tokens"]]}') 

        #print(fields["source_copy_indices"]) 
        #print(fields["source_attention_map"]) 
        #sys.exit()


        # These two fields are used in biaffine parser
        fields["edge_types"] = TextField(
            tokens=[Token(x) for x in list_data["head_tags"]],
            token_indexers=self._edge_type_indexers
        )

        fields["edge_heads"] = SequenceLabelField(
            labels=list_data["head_indices"],
            sequence_field=fields["edge_types"],
            label_namespace="edge_heads"
        )

        if list_data.get('node_mask', None) is not None:
            # Valid nodes are 1; pads are 0.
            fields['valid_node_mask'] = ArrayField(list_data['node_mask'])

        if list_data.get('edge_mask', None) is not None:
            # A matrix of shape [num_nodes, num_nodes] where entry (i, j) is 1
            # if and only if (1) j < i and (2) j is not an antecedent of i.
            # TODO: try to remove the second constrain.
            fields['edge_head_mask'] = ArrayField(list_data['edge_mask'])

        # node attributes 
        #print(f"tgt attr {len(list_data['tgt_attributes'])}")
        #print(list_data['tgt_attributes'])
        #print(f"target tokens {len(fields['target_tokens'])}")
        #print(fields['target_tokens'])

        # this field is actually needed for scoring later
        fields["graph"] = MetadataField(
            list_data['arbor_graph'])


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

        fields["node_name_list"] = MetadataField(list_data['node_name_list'])
        fields["target_dynamic_vocab"] = MetadataField(dict())

        fields["instance_meta"] = MetadataField(dict(
            pos_tag_lut=list_data["pos_tag_lut"],
            source_dynamic_vocab=list_data["src_copy_vocab"],
            target_token_indexers=self._target_token_indexers,
        ))


        return Instance(fields)


