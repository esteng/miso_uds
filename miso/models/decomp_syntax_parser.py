from typing import List, Dict, Tuple, Any
import logging
from collections import OrderedDict

import subprocess
import math
from overrides import overrides
import torch

from allennlp.data import Token, Instance, Vocabulary
from allennlp.data.fields import TextField
from allennlp.data.dataset import Batch
from allennlp.models import Model
from allennlp.modules import TextFieldEmbedder, Embedding, Seq2SeqEncoder
from allennlp.nn.util import get_text_field_mask
from allennlp.data.vocabulary import DEFAULT_PADDING_TOKEN, DEFAULT_OOV_TOKEN
from allennlp.common.util import START_SYMBOL, END_SYMBOL
from allennlp.training.metrics import AttachmentScores

from miso.models.transduction_base import Transduction
from miso.modules.seq2seq_encoders import Seq2SeqBertEncoder, BaseBertWrapper
from miso.modules.decoders import RNNDecoder
from miso.modules.generators import ExtendedPointerGenerator
from miso.modules.parsers import DeepTreeParser, DecompTreeParser, DeepBiaffineParser
from miso.modules.label_smoothing import LabelSmoothing
from miso.modules.decoders.attribute_decoder import NodeAttributeDecoder 
from miso.modules.decoders.edge_decoder import EdgeAttributeDecoder 
from miso.metrics.decomp_metrics import DecompAttrMetrics
from miso.nn.beam_search import BeamSearch
from miso.data.dataset_readers.decomp_parsing.ontology import NODE_ONTOLOGY, EDGE_ONTOLOGY
from miso.metrics.pearson_r import pearson_r
from miso.models.decomp_parser import DecompParser 

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name

@Model.register("decomp_syntax_parser")
class DecompSyntaxParser(DecompParser):

    def __init__(self,
                 vocab: Vocabulary,
                 # source-side
                 bert_encoder: BaseBertWrapper,
                 encoder_token_embedder: TextFieldEmbedder,
                 encoder_pos_embedding: Embedding,
                 encoder: Seq2SeqEncoder,
                 # target-side
                 decoder_token_embedder: TextFieldEmbedder,
                 decoder_node_index_embedding: Embedding,
                 decoder_pos_embedding: Embedding,
                 decoder: RNNDecoder,
                 extended_pointer_generator: ExtendedPointerGenerator,
                 tree_parser: DecompTreeParser,
                 node_attribute_module: NodeAttributeDecoder,
                 edge_attribute_module: EdgeAttributeDecoder,
                 # misc
                 label_smoothing: LabelSmoothing,
                 target_output_namespace: str,
                 pos_tag_namespace: str,
                 edge_type_namespace: str,
                 biaffine_parser: DeepTreeParser = None,
                 dropout: float = 0.0,
                 beam_size: int = 5,
                 max_decoding_steps: int = 50,
                 eps: float = 1e-20,
                 ) -> None:

        super(DecompSyntaxParser, self).__init__(
                                 vocab=vocab,
                                 # source-side
                                 bert_encoder=bert_encoder,
                                 encoder_token_embedder=encoder_token_embedder,
                                 encoder_pos_embedding=encoder_pos_embedding,
                                 encoder=encoder,
                                 # target-side
                                 decoder_token_embedder=decoder_token_embedder,
                                 decoder_node_index_embedding=decoder_node_index_embedding,
                                 decoder_pos_embedding=decoder_pos_embedding,
                                 decoder=decoder,
                                 extended_pointer_generator=extended_pointer_generator,
                                 tree_parser=tree_parser,
                                 node_attribute_module=node_attribute_module,
                                 edge_attribute_module=edge_attribute_module,
                                 # misc
                                 label_smoothing=label_smoothing,
                                 target_output_namespace=target_output_namespace,
                                 pos_tag_namespace=pos_tag_namespace,
                                 edge_type_namespace=edge_type_namespace,
                                 dropout=dropout,
                                 beam_size=beam_size,
                                 max_decoding_steps=max_decoding_steps,
                                 eps=eps)
        
        self.biaffine_parser = biaffine_parser
        self._syntax_metrics = AttachmentScores()
        self.syntax_las = 0.0 
        self.syntax_uas = 0.0 

    @overrides
    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        node_pred_metrics = self._node_pred_metrics.get_metric(reset)
        edge_pred_metrics = self._edge_pred_metrics.get_metric(reset)
        decomp_metrics = self._decomp_metrics.get_metric(reset) 
        syntax_metrics = self._syntax_metrics.get_metric(reset)

        metrics = OrderedDict(
            ppl=node_pred_metrics["ppl"],
            node_pred=node_pred_metrics["accuracy"] * 100,
            generate=node_pred_metrics["generate"] * 100,
            src_copy=node_pred_metrics["src_copy"] * 100,
            tgt_copy=node_pred_metrics["tgt_copy"] * 100,
            node_pearson=decomp_metrics["node_pearson_r"],
            edge_pearson=decomp_metrics["edge_pearson_r"],
            pearson=decomp_metrics["pearson_r"],
            uas=edge_pred_metrics["UAS"] * 100,
            las=edge_pred_metrics["LAS"] * 100,
            syn_uas=syntax_metrics["UAS"] * 100,
            syn_las=syntax_metrics["LAS"] * 100,
        )
        metrics["s_f1"] = self.val_s_f1
        metrics["syn_las"] = self.syntax_las
        metrics["syn_uas"] = self.syntax_uas
        return metrics

    def _update_syntax_scores(self):
        scores = self._syntax_metrics.get_metric(reset=True)
        self.syntax_las = scores["LAS"] * 100
        self.syntax_uas = scores["UAS"] * 100

    def _compute_biaffine_loss(self, biaffine_outputs, inputs):
        edge_prediction_loss = self._compute_edge_prediction_loss(
                                biaffine_outputs['edge_head_ll'],
                                biaffine_outputs['edge_type_ll'],
                                biaffine_outputs['edge_heads'],
                                biaffine_outputs['edge_types'],
                                inputs['syn_edge_heads'],
                                inputs['syn_edge_types']['syn_edge_types'],
                                inputs['syn_valid_node_mask'],
                                syntax=True)
        return edge_prediction_loss['loss']

    def _parse_syntax(self,
                      encoder_outputs: torch.Tensor,
                      edge_head_mask: torch.Tensor,
                      edge_heads: torch.Tensor = None) -> Dict:

        parser_outputs = self.biaffine_parser(
                                query=encoder_outputs,
                                key=encoder_outputs,
                                edge_head_mask=edge_head_mask,
                                gold_edge_heads=edge_heads
                            )

        return parser_outputs


    @overrides
    def _training_forward(self, inputs: Dict) -> Dict[str, torch.Tensor]:
        encoding_outputs = self._encode(
            tokens=inputs["source_tokens"],
            pos_tags=inputs["source_pos_tags"],
            subtoken_ids=inputs["source_subtoken_ids"],
            token_recovery_matrix=inputs["source_token_recovery_matrix"],
            mask=inputs["source_mask"]
        )
        
        # if we're doing encoder-side 
        if "syn_tokens_str" in inputs.keys():
            #arc_logits, label_logits = self.biaffine_parser(encoding_outputs['encoder_outputs']) 
            biaffine_outputs = self._parse_syntax(encoding_outputs['encoder_outputs'],
                                            inputs["syn_edge_head_mask"],
                                            inputs["syn_edge_heads"])



            biaffine_loss = self._compute_biaffine_loss(biaffine_outputs,
                                                        inputs)

            self._update_syntax_scores()


        else:
            biaffine_loss = 0.0

        decoding_outputs = self._decode(
            tokens=inputs["target_tokens"],
            node_indices=inputs["target_node_indices"],
            pos_tags=inputs["target_pos_tags"],
            encoder_outputs=encoding_outputs["encoder_outputs"],
            hidden_states=encoding_outputs["final_states"],
            mask=inputs["source_mask"]
        )

        node_prediction_outputs = self._extended_pointer_generator(
            inputs=decoding_outputs["attentional_tensors"],
            source_attention_weights=decoding_outputs["source_attention_weights"],
            target_attention_weights=decoding_outputs["target_attention_weights"],
            source_attention_map=inputs["source_attention_map"],
            target_attention_map=inputs["target_attention_map"]
        )

        # compute node attributes
        node_attribute_outputs = self._node_attribute_predict(
            decoding_outputs["rnn_outputs"][:,:-1,:],
            inputs["node_attribute_truth"],
            inputs["node_attribute_mask"]
        )

        edge_prediction_outputs = self._parse(
            rnn_outputs=decoding_outputs["rnn_outputs"],
            edge_head_mask=inputs["edge_head_mask"],
            edge_heads=inputs["edge_heads"]
        )

        edge_attribute_outputs = self._edge_attribute_predict(
                edge_prediction_outputs["edge_type_query"],
                edge_prediction_outputs["edge_type_key"],
                edge_prediction_outputs["edge_heads"],
                inputs["edge_attribute_truth"],
                inputs["edge_attribute_mask"]
                )

        node_pred_loss = self._compute_node_prediction_loss(
            prob_dist=node_prediction_outputs["hybrid_prob_dist"],
            generation_outputs=inputs["generation_outputs"],
            source_copy_indices=inputs["source_copy_indices"],
            target_copy_indices=inputs["target_copy_indices"],
            source_dynamic_vocab_size=inputs["source_dynamic_vocab_size"],
            source_attention_weights=decoding_outputs["source_attention_weights"],
            coverage_history=decoding_outputs["coverage_history"]
        )
        edge_pred_loss = self._compute_edge_prediction_loss(
            edge_head_ll=edge_prediction_outputs["edge_head_ll"],
            edge_type_ll=edge_prediction_outputs["edge_type_ll"],
            pred_edge_heads=edge_prediction_outputs["edge_heads"],
            pred_edge_types=edge_prediction_outputs["edge_types"],
            gold_edge_heads=inputs["edge_heads"],
            gold_edge_types=inputs["edge_types"],
            valid_node_mask=inputs["valid_node_mask"]
        )

        loss = node_pred_loss["loss_per_node"] + edge_pred_loss["loss_per_node"] + \
               node_attribute_outputs['loss'] + edge_attribute_outputs['loss'] + \
               biaffine_loss
        #loss = biaffine_loss

        # compute combined pearson 
        self._decomp_metrics(None, None, None, None, "both")

        return dict(loss=loss, 
                    node_attributes = node_attribute_outputs['pred_dict']['pred_attributes'],
                    edge_attributes = edge_attribute_outputs['pred_dict']['pred_attributes'])


    
