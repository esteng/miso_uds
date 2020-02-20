from typing import Dict, Tuple
import logging

from overrides import overrides
import torch
from allennlp.data import Vocabulary
from allennlp.models import Model
from allennlp.modules import TextFieldEmbedder, Embedding, InputVariationalDropout, Seq2SeqEncoder
from allennlp.training.metrics import AttachmentScores
from allennlp.data.vocabulary import DEFAULT_PADDING_TOKEN
from allennlp.common.util import START_SYMBOL, END_SYMBOL

from miso.modules.seq2seq_encoders import Seq2SeqBertEncoder
from miso.modules.decoders import RNNDecoder
from miso.modules.generators import ExtendedPointerGenerator
from miso.modules.parsers import DeepTreeParser
from miso.metrics import ExtendedPointerGeneratorMetrics

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


@Model.register("transductive_parser")
class TransductiveParser(Model):

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
                 target_output_namespace: str,
                 edge_type_namespace: str,
                 dropout: float = 0.0,
                 beam_size: int = 5,
                 max_decoding_length: int = 50,
                 eps: float = 1e-20,
                 ) -> None:
        super().__init__(vocab=vocab)
        # source-side
        self._bert_encoder = bert_encoder
        self._encoder_token_embedder = encoder_token_embedder
        self._encoder_pos_embedding = encoder_pos_embedding
        self._encoder_anonymization_embedding = encoder_anonymization_embedding
        self._encoder = encoder

        # target-side
        self._decoder_token_embedder = decoder_token_embedder
        self._decoder_node_index_embedding = decoder_node_index_embedding
        self._decoder_pos_embedding = decoder_pos_embedding
        self._decoder = decoder
        self._extended_pointer_generator = extended_pointer_generator
        self._tree_parser = tree_parser

        # metrics
        self._node_pred_metrics = ExtendedPointerGeneratorMetrics()
        self._edge_pred_metrics = AttachmentScores()

        self._dropout = InputVariationalDropout(p=dropout)
        self._beam_size = beam_size
        self._max_decoding_length = max_decoding_length
        self._eps = eps

        # dynamic initialization
        self._target_output_namespace = target_output_namespace
        self._edge_type_namespace = edge_type_namespace
        self._vocab_size = self.vocab.get_vocab_size(target_output_namespace)
        self._vocab_pad_index = self.vocab.get_token_index(DEFAULT_PADDING_TOKEN, target_output_namespace)
        self._vocab_bos_index = self.vocab.get_token_index(START_SYMBOL, target_output_namespace)
        self._extended_pointer_generator.reset_vocab_linear(
            vocab_size=vocab.get_vocab_size(target_output_namespace),
            vocab_pad_index=self._vocab_pad_index
        )
        self._tree_parser.reset_edge_type_bilinear(num_labels=vocab.get_vocab_size(edge_type_namespace))

    @overrides
    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        metrics = dict()
        node_pred_metrics = self._node_pred_metrics.get_metric(reset)
        edge_pred_metrics = self._edge_pred_metrics.get_metric(reset)
        metrics.update(node_pred_metrics)
        metrics.update(edge_pred_metrics)
        return metrics

    def _pprint(self, inputs: Dict, index: int = 1) -> None:
        logger.info("==== Source-side Input ====")
        logger.info("source tokens:")
        source_tokens = inputs["source_tokens"]["source_tokens"][index].tolist()
        logger.info("\t" + " ".join(map(lambda x: self.vocab.get_token_from_index(x, "source_tokens"), source_tokens)))
        logger.info("\t" + str(source_tokens))
        logger.info("source_pos_tags:")
        source_pos_tags = inputs["source_pos_tags"][index].tolist()
        logger.info("\t" + " ".join(map(
            lambda x: self.vocab.get_token_from_index(x, "pos_tags"), source_pos_tags)))
        logger.info("source_anonymiaztion_tags:")
        anonymization_tags = inputs["source_anonymization_tags"][index].tolist()
        logger.info("\t" + str(anonymization_tags))

        logger.info("==== Target-side Input ====")
        logger.info("target tokens:")
        target_tokens = inputs["target_tokens"]["target_tokens"][index].tolist()
        logger.info("\t" + " ".join(map(lambda x: self.vocab.get_token_from_index(x, "target_tokens"), target_tokens)))
        logger.info("\t" + str(target_tokens))
        logger.info("target_pos_tags:")
        target_pos_tags = inputs["target_pos_tags"][index].tolist()
        logger.info("\t" + " ".join(map(
            lambda x: self.vocab.get_token_from_index(x, "pos_tags"), target_pos_tags)))
        logger.info("target_node_indices:")
        node_indices = inputs["target_node_indices"][index].tolist()
        logger.info("\t" + str(node_indices))

        logger.info("==== Output ====")
        logger.info("generation_outputs:")
        generation_tokens = inputs["generation_outputs"]["generation_tokens"][index].tolist()
        logger.info("\t" + " ".join(
            map(lambda x: self.vocab.get_token_from_index(x, "generation_tokens"), generation_tokens)))
        logger.info("\t" + str(generation_tokens))
        logger.info("target_copy_indices:")
        target_copy_indices = inputs["target_copy_indices"][index].tolist()
        logger.info("\t" + str(target_copy_indices))
        logger.info("source_copy_indices:")
        source_copy_indices = inputs["source_copy_indices"][index].tolist()
        logger.info("\t" + str(source_copy_indices))
        logger.info("edge_heads:")
        edge_heads = inputs["edge_heads"][index].tolist()
        logger.info("\t" + str(edge_heads))
        logger.info("edge_types:")
        edge_types = inputs["edge_types"]["edge_types"][index].tolist()
        logger.info("\t" + " ".join(
            map(lambda x: self.vocab.get_token_from_index(x, "edge_types"), edge_types)))

        logger.info("==== Misc ====")
        logger.info("target attention map:")
        target_attention_map = inputs["target_attention_map"][index]
        logger.info(target_attention_map)
        logger.info("source attention map:")
        source_attention_map = inputs["source_attention_map"][index]
        logger.info(source_attention_map)
        logger.info("edge head mask:")
        edge_head_mask = inputs["edge_head_mask"][index]
        logger.info(edge_head_mask)

    def _prepare_inputs(self, raw_inputs: Dict) -> Dict:
        inputs = raw_inputs.copy()
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
        inputs["generation_outputs"] = raw_inputs["generation_outputs"]["tokens"][:, 1:]
        inputs["source_copy_indices"] = raw_inputs["source_copy_indices"][:, 1:]
        inputs["target_copy_indices"] = raw_inputs["target_copy_indices"][:, 1:]

        # [batch, target_seq_length, target_seq_length + 1(sentinel)]
        inputs["target_attention_map"] = raw_inputs["target_attention_map"][:, 1:]  # exclude BOS
        # [batch, 1(unk) + source_seq_length, dynamic_vocab_size]
        # Exclude unk and the last pad.
        inputs["source_attention_map"] = raw_inputs["source_attention_map"][:, 1:-1]

        inputs["source_dynamic_vocab_size"] = inputs["source_attention_map"].size(2)

        self._pprint(inputs)

        return inputs

    @overrides
    def forward(self, **raw_inputs: Dict) -> Dict[str, torch.Tensor]:
        inputs = self._prepare_inputs(raw_inputs)
        if True:  # self.training:
            return self._training_forward(inputs)

    def _compute_edge_prediction_loss(self,
                                      edge_head_ll: torch.Tensor,
                                      edge_type_ll: torch.Tensor,
                                      pred_edge_heads: torch.Tensor,
                                      pred_edge_types: torch.Tensor,
                                      gold_edge_heads: torch.Tensor,
                                      gold_edge_types: torch.Tensor,
                                      valid_node_mask: torch.Tensor) -> Dict:
        """
        Compute the edge prediction loss.

        :param edge_head_ll: [batch_size, target_length, target_length + 1 (for sentinel)].
        :param edge_type_ll: [batch_size, target_length, num_labels].
        :param pred_edge_heads: [batch_size, target_length].
        :param pred_edge_types: [batch_size, target_length].
        :param gold_edge_heads: [batch_size, target_length].
        :param gold_edge_types: [batch_size, target_length].
        :param valid_node_mask: [batch_size, target_length].
        """
        # Index the log-likelihood (ll) of gold edge heads and types.
        batch_size, target_length, _ = edge_head_ll.size()
        batch_indices = torch.arange(0, batch_size).view(batch_size, 1).type_as(gold_edge_heads)
        node_indices = torch.arange(0, target_length).view(1, target_length)\
            .expand(batch_size, target_length).type_as(gold_edge_heads)
        gold_edge_head_ll = edge_head_ll[batch_indices, node_indices, gold_edge_heads]
        gold_edge_type_ll = edge_type_ll[batch_indices, node_indices, gold_edge_types]
        # Set the ll of invalid nodes to 0.
        num_nodes = valid_node_mask.sum().float()
        valid_node_mask = valid_node_mask.byte()
        gold_edge_head_ll.masked_fill_(~valid_node_mask, 0)
        gold_edge_type_ll.masked_fill_(~valid_node_mask, 0)
        # Negative log-likelihood.
        loss = -(gold_edge_head_ll.sum() + gold_edge_type_ll.sum())
        # Update metrics.
        self._edge_pred_metrics(
            predicted_indices=pred_edge_heads,
            predicted_labels=pred_edge_types,
            gold_indices=gold_edge_heads,
            gold_labels=gold_edge_types,
            mask=valid_node_mask
        )

        return dict(
            loss=loss,
            num_nodes=num_nodes
        )

    def _compute_node_prediction_loss(self,
                                      prob_dist: torch.Tensor,
                                      generation_outputs: torch.Tensor,
                                      source_copy_indices: torch.Tensor,
                                      target_copy_indices: torch.Tensor,
                                      source_dynamic_vocab_size: int,
                                      source_attention_weights: torch.Tensor = None,
                                      coverage_history: torch.Tensor = None) -> Dict:
        """
        Compute the node prediction loss based on the final hybrid probability distribution.

        :param prob_dist: probability distribution,
            [batch_size, target_length, vocab_size + source_dynamic_vocab_size + target_dynamic_vocab_size].
        :param generation_outputs: generated node indices in the pre-defined vocabulary,
            [batch_size, target_length].
        :param source_copy_indices:  source-side copied node indices in the source dynamic vocabulary,
            [batch_size, target_length].
        :param target_copy_indices:  target-side copied node indices in the source dynamic vocabulary,
            [batch_size, target_length].
        :param source_dynamic_vocab_size: int.
        :param source_attention_weights: None or [batch_size, target_length, source_length].
        :param coverage_history: None or a tensor recording the source-side coverage history.
            [batch_size, target_length, source_length].
        """
        _, prediction = prob_dist.max(2)
        batch_size, target_length = prediction.size()
        not_pad_mask = generation_outputs.ne(self._vocab_pad_index)
        num_nodes = not_pad_mask.sum()

        # Priority: target_copy > source_copy > generation
        # Prepare mask.
        valid_target_copy_mask = target_copy_indices.ne(0) & not_pad_mask  # 0 for sentinel.
        valid_source_copy_mask = ~valid_target_copy_mask & not_pad_mask \
                                 & source_copy_indices.ne(1) & source_copy_indices.ne(0)  # 1 for unk; 0 for pad.
        valid_generation_mask = ~(valid_target_copy_mask | valid_source_copy_mask) & not_pad_mask
        # Prepare hybrid targets.
        _target_copy_indices = (target_copy_indices + self._vocab_size + source_dynamic_vocab_size) \
                               * valid_target_copy_mask.long()
        _source_copy_indices = (source_copy_indices + self._vocab_size) * valid_source_copy_mask.long()
        _generation_outputs = generation_outputs * valid_generation_mask.long()
        hybrid_targets = _target_copy_indices + _source_copy_indices + _generation_outputs

        # Compute loss.
        log_prob_dist = (prob_dist.view(batch_size * target_length, -1) + self._eps).log()
        flat_hybrid_targets = hybrid_targets.view(batch_size * target_length)
        loss = self.label_smoothing(log_prob_dist, flat_hybrid_targets)
        # Coverage loss.
        if coverage_history is not None:
            coverage_loss = torch.sum(torch.min(coverage_history, source_attention_weights), 2)
            coverage_loss = (coverage_loss * not_pad_mask.float()).sum()
            loss = loss + coverage_loss
        # Update metric stats.
        self._node_pred_metrics(
            loss=loss,
            prediction=prediction,
            generation_outputs=_generation_outputs,
            valid_generation_mask=valid_generation_mask,
            source_copy_indices=_source_copy_indices,
            valid_source_copy_mask=valid_source_copy_mask,
            target_copy_indices=_target_copy_indices,
            valid_target_copy_mask=valid_target_copy_mask
        )

        return dict(
            loss=loss,
            num_nodes=num_nodes
        )

    def _decode(self,
                tokens: Dict[str, torch.Tensor],
                node_indices: torch.Tensor,
                pos_tags: torch.Tensor,
                encoder_outputs: torch.Tensor,
                hidden_states: Tuple[torch.Tensor, torch.Tensor],
                mask: torch.Tensor) -> Dict:
        # [batch, num_tokens, embedding_size]
        decoder_inputs = torch.cat([
            self._decoder_token_embedder(tokens),
            self._decoder_node_index_embedding(node_indices),
            self._decoder_pos_embedding(pos_tags)
        ], dim=2)
        decoder_inputs = self._dropout(decoder_inputs)

        decoder_outputs = self._decoder(
            inputs=decoder_inputs,
            source_memory_bank=encoder_outputs,
            source_mask=mask,
            hidden_state=hidden_states
        )

        return decoder_outputs

    def _encode(self,
                tokens: Dict[str, torch.Tensor],
                pos_tags: torch.Tensor,
                anonymization_tags: torch.Tensor,
                subtoken_ids: torch.Tensor,
                token_recovery_matrix: torch.Tensor,
                mask: torch.Tensor):
        # [batch, num_tokens, embedding_size]
        encoder_inputs = [
            self._encoder_token_embedder(tokens),
            self._encoder_pos_embedding(pos_tags),
            self._encoder_anonymization_embedding(anonymization_tags)
        ]
        if subtoken_ids is not None and self._bert_encoder is not None:
            bert_embeddings = self._bert_encoder(
                input_ids=subtoken_ids,
                attention_mask=subtoken_ids.ne(0),
                output_all_encoded_layers=False,
                token_recovery_matrix=token_recovery_matrix
            )
            encoder_inputs += [bert_embeddings]
        encoder_inputs = torch.cat(encoder_inputs, 2)
        encoder_inputs = self._dropout(encoder_inputs)

        # [batch, num_tokens, encoder_output_size]
        encoder_outputs = self._encoder(encoder_inputs, mask)
        encoder_outputs = self._dropout(encoder_outputs)
        # A tuple of (state, memory) with shape [num_layers, batch, encoder_output_size]
        encoder_final_states = self.encoder.get_final_states()
        self.encoder.reset_states()

        return dict(
            encoder_outputs=encoder_outputs,
            final_states=encoder_final_states
        )

    def _parse(self,
               rnn_outputs: torch.Tensor,
               edge_head_mask: torch.Tensor,
               edge_heads: torch.Tensor) -> Dict:
        """
        Based on the vector representation for each node, predict its head and edge type.
        :param rnn_outputs: vector representations of nodes, including <BOS>.
            [batch_size, target_length + 1, hidden_vector_dim].
        :param edge_head_mask: mask used in the edge head search.
            [batch_size, target_length, target_length].
        :param edge_heads: gold head indices, [batch_size, target_length]
        """
        # Exclude <BOS>.
        # <EOS> is already excluded in ``_prepare_inputs''.
        rnn_outputs = self._dropout(rnn_outputs[:, 1:])
        parser_outputs = self._tree_parser(
            query=rnn_outputs,
            key=rnn_outputs,
            edge_head_mask=edge_head_mask,
            gold_edge_heads=edge_heads
        )
        return parser_outputs

    def _training_forward(self,
                          inputs: Dict) -> Dict[str, torch.Tensor]:
        encoding_outputs = self._encode(
            tokens=inputs["source_tokens"],
            pos_tags=inputs["source_pos_tags"],
            anonymization_tags=inputs["source_anonymization_tags"],
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
        edge_prediction_outputs = self._parse(
            rnn_outputs=decoding_outputs["rnn_outputs"],
            edge_head_mask=inputs["edge_mask"],
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
        loss = node_pred_loss["loss"] + edge_pred_loss["loss"]
        return dict(loss=loss)
