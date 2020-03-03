from typing import List, Dict, Tuple
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

from miso.models.transduction_base import Transduction
from miso.modules.seq2seq_encoders import Seq2SeqBertEncoder
from miso.modules.decoders import RNNDecoder
from miso.modules.generators import ExtendedPointerGenerator
from miso.modules.parsers import DeepTreeParser
from miso.modules.label_smoothing import LabelSmoothing
from miso.modules.decoders.attribute_decoder import NodeAttributeDecoder 
from miso.nn.beam_search import BeamSearch
from miso.data.dataset_readers.decomp_parsing.ontology import NODE_ONTOLOGY, EDGE_ONTOLOGY
from miso.metrics.pearson_r import pearson_r
# The following imports are added for mimick testing.
#from miso.predictors.predictor import Predictor
#from miso.commands.predict import _PredictManager
#from miso.commands.s_score import Scorer, compute_args, ComputeTup
#from miso.losses import LossFunctionDict

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


@Model.register("decomp_parser")
class DecompParser(Transduction):

    def __init__(self,
                 vocab: Vocabulary,
                 # source-side
                 bert_encoder: Seq2SeqBertEncoder,
                 encoder_token_embedder: TextFieldEmbedder,
                 encoder_pos_embedding: Embedding,
                 encoder_anonymization_embedding: Embedding,
                 encoder: Seq2SeqEncoder,
                 # target-side
                 decoder_token_embedder: TextFieldEmbedder,
                 decoder_node_index_embedding: Embedding,
                 decoder_pos_embedding: Embedding,
                 decoder: RNNDecoder,
                 extended_pointer_generator: ExtendedPointerGenerator,
                 tree_parser: DeepTreeParser,
                 # misc
                 label_smoothing: LabelSmoothing,
                 target_output_namespace: str,
                 pos_tag_namespace: str,
                 edge_type_namespace: str,
                 dropout: float = 0.0,
                 beam_size: int = 5,
                 max_decoding_steps: int = 50,
                 eps: float = 1e-20,
                 ) -> None:
        super().__init__(vocab=vocab,
                         # source-side
                         bert_encoder=bert_encoder,
                         encoder_token_embedder=encoder_token_embedder,
                         encoder=encoder,
                         # target-side
                         decoder_token_embedder=decoder_token_embedder,
                         decoder_node_index_embedding=decoder_node_index_embedding,
                         decoder=decoder,
                         extended_pointer_generator=extended_pointer_generator,
                         tree_parser=tree_parser,
                         # misc
                         label_smoothing=label_smoothing,
                         target_output_namespace=target_output_namespace,
                         dropout=dropout,
                         eps=eps)

        # source-side
        self._encoder_pos_embedding = encoder_pos_embedding

        # target-side
        self._decoder_pos_embedding = decoder_pos_embedding

        # metrics
        self.val_s_f1 = .0
        self.val_s_precision = .0
        self.val_s_recall = .0

        self._beam_size = beam_size
        self._max_decoding_steps = max_decoding_steps

        # dynamic initialization
        self._target_output_namespace = target_output_namespace
        self._pos_tag_namespace = pos_tag_namespace
        self._edge_type_namespace = edge_type_namespace
        self._vocab_size = self.vocab.get_vocab_size(target_output_namespace)
        self._vocab_pad_index = self.vocab.get_token_index(DEFAULT_PADDING_TOKEN, target_output_namespace)
        self._vocab_bos_index = self.vocab.get_token_index(START_SYMBOL, target_output_namespace)
        self._vocab_eos_index = self.vocab.get_token_index(END_SYMBOL, target_output_namespace)
        self._extended_pointer_generator.reset_vocab_linear(
            vocab_size=vocab.get_vocab_size(target_output_namespace),
            vocab_pad_index=self._vocab_pad_index
        )
        self._tree_parser.reset_edge_type_bilinear(num_labels=vocab.get_vocab_size(edge_type_namespace))
        self._label_smoothing.reset_parameters(pad_index=self._vocab_pad_index)
        self._beam_search = BeamSearch(self._vocab_eos_index, self._max_decoding_steps, self._beam_size)

    @overrides
    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        node_pred_metrics = self._node_pred_metrics.get_metric(reset)
        edge_pred_metrics = self._edge_pred_metrics.get_metric(reset)
        metrics = OrderedDict(
            ppl=node_pred_metrics["ppl"],
            node_pred=node_pred_metrics["accuracy"] * 100,
            generate=node_pred_metrics["generate"] * 100,
            src_copy=node_pred_metrics["src_copy"] * 100,
            tgt_copy=node_pred_metrics["tgt_copy"] * 100,
            node_pearson=node_pred_metrics["pearson_r"],
            edge_pearson=edge_pred_metrics["pearson_r"],
            uas=edge_pred_metrics["UAS"] * 100,
            las=edge_pred_metrics["LAS"] * 100,
        )
        metrics["s_f1"] = self.val_s_f1
        return metrics

    def forward(self, **raw_inputs: Dict) -> Dict:
        inputs = self._prepare_inputs(raw_inputs)
        if self.training:
            return self._training_forward(inputs)
        else:
            return self._test_forward(inputs)

    def _prepare_inputs(self, raw_inputs):
        inputs = raw_inputs.copy()

        inputs["source_mask"] = get_text_field_mask(raw_inputs["source_tokens"])

        source_subtoken_ids = raw_inputs.get("source_subtoken_ids", None)
        if source_subtoken_ids is None:
            inputs["source_subtoken_ids"] = None
        else:
            inputs["source_subtoken_ids"] = source_subtoken_ids.long()
        source_token_recovery_matrix = raw_inputs.get("source_token_recovery_matrix", None)
        if source_token_recovery_matrix is None:
            inputs["source_token_recovery_matrix"] = None
        else:
            inputs["source_token_recovery_matrix"] = source_token_recovery_matrix.long()

        # Exclude <BOS>.
        inputs["generation_outputs"] = raw_inputs["generation_outputs"]["generation_tokens"][:, 1:]
        inputs["source_copy_indices"] = raw_inputs["source_copy_indices"][:, 1:]
        inputs["target_copy_indices"] = raw_inputs["target_copy_indices"][:, 1:]


        # [batch, target_seq_length, target_seq_length + 1(sentinel)]
        inputs["target_attention_map"] = raw_inputs["target_attention_map"][:, 1:]  # exclude UNK
        # [batch, 1(unk) + source_seq_length, dynamic_vocab_size]
        # Exclude unk and the last pad.
        inputs["source_attention_map"] = raw_inputs["source_attention_map"][:, 1:-1]

        inputs["source_dynamic_vocab_size"] = inputs["source_attention_map"].size(2)

        inputs["edge_types"] = raw_inputs["edge_types"]["edge_types"]

        # self._pprint(inputs)
        node_attribute_stack = raw_inputs['target_attributes']
        node_attribute_values = node_attribute_stack[0,:,:,:].squeeze(0)
        node_attribute_mask = node_attribute_stack[1,:,:,:].squeeze(0)
        edge_attribute_stack = raw_inputs['edge_attributes']
        edge_attribute_values = edge_attribute_stack[0,:,:,:].squeeze(0)
        edge_attribute_mask = edge_attribute_stack[1,:,:,:].squeeze(0)

        inputs.update(dict(
                # like decoder_token_inputs
                node_attribute_truth = node_attribute_values[:,1:-1,:],
                node_attribute_mask = node_attribute_mask[:,1:-1,:],
                edge_attribute_truth = edge_attribute_values[:,1:-1,:],
                edge_attribute_mask = edge_attribute_mask[:,1:-1,:]
        ))

        return inputs





        return encoder_inputs, decoder_inputs, generator_inputs, parser_inputs, attribute_inputs

    @overrides
    def _training_forward(self, inputs: Dict) -> Dict[str, torch.Tensor]:
        encoding_outputs = self._encode(
            tokens=inputs["source_tokens"],
            pos_tags=inputs["source_pos_tags"],
            subtoken_ids=inputs["source_subtoken_ids"],
            token_recovery_matrix=inputs["source_token_recovery_matrix"],
            mask=inputs["source_mask"]
        )
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

        logger.info(decoder_outputs.keys())
        logger.info(encoder_outputs.keys())
        sys.exit()
        # compute node attributes
        node_attributes = self._node_attribute_module(
            decoder_outputs[""],
            encoder_outputs[""])

        edge_prediction_outputs = self._parse(
            rnn_outputs=decoding_outputs["rnn_outputs"],
            edge_head_mask=inputs["edge_head_mask"],
            edge_heads=inputs["edge_heads"]
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
        loss = node_pred_loss["loss_per_node"] + edge_pred_loss["loss_per_node"]
        return dict(loss=loss)


