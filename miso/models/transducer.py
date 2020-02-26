from typing import List, Dict, Tuple
import logging
from collections import OrderedDict

from overrides import overrides
import torch
from allennlp.data import Token, Instance, Vocabulary
from allennlp.data.fields import TextField
from allennlp.data.dataset import Batch
from allennlp.models import Model
from allennlp.modules import TextFieldEmbedder, Embedding, InputVariationalDropout, Seq2SeqEncoder
from allennlp.training.metrics import AttachmentScores
from allennlp.nn.util import get_text_field_mask
from allennlp.data.vocabulary import DEFAULT_PADDING_TOKEN, DEFAULT_OOV_TOKEN
from allennlp.common.util import START_SYMBOL, END_SYMBOL

from miso.modules.seq2seq_encoders import Seq2SeqBertEncoder
from miso.modules.decoders import RNNDecoder
from miso.modules.generators import ExtendedPointerGenerator
from miso.modules.parsers import DeepTreeParser
from miso.modules.label_smoothing import LabelSmoothing
from miso.nn.beam_search import BeamSearch
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
                 label_smoothing: LabelSmoothing,
                 target_output_namespace: str,
                 pos_tag_namespace: str,
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

        self._label_smoothing = label_smoothing
        self._dropout = InputVariationalDropout(p=dropout)
        self._beam_size = beam_size
        self._max_decoding_length = max_decoding_length
        self._eps = eps

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
        self._beam_search = BeamSearch(self._vocab_eos_index, self._max_decoding_length, self._beam_size)

    @overrides
    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        node_pred_metrics = self._node_pred_metrics.get_metric(reset)
        edge_pred_metrics = self._edge_pred_metrics.get_metric(reset)
        return OrderedDict(
            ppl=node_pred_metrics["ppl"],
            node_pred=node_pred_metrics["accuracy"] * 100,
            generate=node_pred_metrics["generate"] * 100,
            src_copy=node_pred_metrics["src_copy"] * 100,
            tgt_copy=node_pred_metrics["tgt_copy"] * 100,
            gen_freq=node_pred_metrics["gen_freq"] * 100,
            src_freq=node_pred_metrics["src_freq"] * 100,
            tgt_freq=node_pred_metrics["tgt_freq"] * 100,
            uas=edge_pred_metrics["UAS"] * 100,
            las=edge_pred_metrics["LAS"] * 100,
        )

    def _pprint(self, inputs: Dict, index: int = 0) -> None:
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
        generation_tokens = inputs["generation_outputs"][index].tolist()
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
        edge_types = inputs["edge_types"][index].tolist()
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
        """
        Prepare inputs as below:
            source_tokens:          PERSON could help himself
            source_pos_tags:            NP    MD   VB     PP
            source_anonymization_tags:   1     0    0      0
            (source_dynamic_vocab: {0: pad, 1: UNK, 2: PERSON, 3: could, 4: help, 5: himself})

            target_tokens:          <BOS> possible help person PERSON person
            target_pos_tags:          UNK      UNK   VB    UNK     NP    UNK
            target_node_indices:        0        1    2      3      4      3
            (target_dynamic_vocab: {0: UNK, 1: possible, 2: help, 3: person, 4: PERSON})

            generation_outputs:         possible  UNK person  UNK person <EOS>
            source_copy_indices                1    4      1    2      1     1
            target_copy_indices:               0    0      0    0      3     0
            edge_heads:                        0    1      2    3      2     0
            edge_types:                     root arg0   arg0 name   arg1   pad

            source_attention_map:                   p U P c h h
                                       PERSON       0 0 1 0 0 0
                                       could        0 0 0 1 0 0
                                       help         0 0 0 0 1 0
                                       himself      0 0 0 0 0 1

            target_attention_map:                   S p h p P
                                       possible     0 1 0 0 0
                                       help         0 0 1 0 0
                                       person       0 0 0 1 0
                                       Person       0 0 0 0 1
                                       person       0 0 0 1 0
        """
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

        return inputs

    @overrides
    def forward(self, **raw_inputs: Dict) -> Dict:
        inputs = self._prepare_inputs(raw_inputs)
        if self.training:
            return self._training_forward(inputs)
        else:
            return self._test_forward(inputs)

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
        valid_node_mask = valid_node_mask.bool()
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
            num_nodes=num_nodes,
            loss_per_node=loss/num_nodes,
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
        loss = self._label_smoothing(log_prob_dist, flat_hybrid_targets)
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
            num_nodes=num_nodes,
            loss_per_node=loss/num_nodes,
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
        encoder_final_states = self._encoder.get_final_states()
        self._encoder.reset_states()

        return dict(
            encoder_outputs=encoder_outputs,
            final_states=encoder_final_states
        )

    def _parse(self,
               rnn_outputs: torch.Tensor,
               edge_head_mask: torch.Tensor,
               edge_heads: torch.Tensor = None) -> Dict:
        """
        Based on the vector representation for each node, predict its head and edge type.
        :param rnn_outputs: vector representations of nodes, including <BOS>.
            [batch_size, target_length + 1, hidden_vector_dim].
        :param edge_head_mask: mask used in the edge head search.
            [batch_size, target_length, target_length].
        :param edge_heads: None or gold head indices, [batch_size, target_length]
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

    def _prepare_decoding_start_state(self, inputs: Dict, encoding_outputs: Dict[str, torch.Tensor]) \
            -> Tuple[torch.Tensor, Dict[str, torch.Tensor], Dict]:
        batch_size = inputs["source_tokens"]["tokens"].size(0)
        bos = self.vocab.get_token_index(START_SYMBOL, self._target_output_namespace)
        start_predictions = inputs["source_tokens"]["tokens"].new_full((batch_size,), bos)
        start_state = {
            # [batch_size, *]
            "source_memory_bank": encoding_outputs["encoder_outputs"],
            "hidden_state_1": encoding_outputs["final_states"][0].permute(1, 0, 2),
            "hidden_state_2": encoding_outputs["final_states"][1].permute(1, 0, 2),
            "source_mask": inputs["source_mask"],
            "source_attention_map": inputs["source_attention_map"],
            "target_attention_map": inputs["source_attention_map"].new_zeros(
                (batch_size, self._max_decoding_length, self._max_decoding_length + 1))
        }
        auxiliaries = {
            "batch_size": batch_size,
            "last_decoding_step": -1,  # At <BOS>, we set it to -1.
            "source_dynamic_vocab_size": inputs["source_dynamic_vocab_size"],
            "instance_meta": inputs["instance_meta"]
        }
        return start_predictions, start_state, auxiliaries

    def _prepare_next_inputs(self,
                             predictions: torch.Tensor,
                             target_attention_map: torch.Tensor,
                             meta_data: List[Dict],
                             batch_size: int,
                             last_decoding_step: int,
                             source_dynamic_vocab_size: int) -> Dict:
        """
        Read out a group of hybrid predictions. Based on different ways of node prediction,
        find the corresponding token, node index and pos tags. Prepare the tensorized inputs
        for the next decoding step. Update the target attention map, target dynamic vocab, etc.
        :param predictions: [group_size,]
        :param target_attention_map: [group_size, target_length, target_dynamic_vocab_size].
        :param meta_data: meta data for each instance.
        :param batch_size: int.
        :param last_decoding_step: the decoding step starts from 0, so the last decoding step
            starts from -1.
        :param source_dynamic_vocab_size: int.
        """
        # On the default, if a new node is created via either generation or source-side copy,
        # its node index will be last_decoding_step + 1. One shift between the last decoding
        # step and the default node index is because node index 0 is reserved for no target copy.
        # See `_prepare_inputs` for detail.
        default_node_index = last_decoding_step + 1

        def batch_index(instance_i: int) -> int:
            if predictions.size(0) == batch_size * self._beam_size:
                return instance_i // self._beam_size
            else:
                return instance_i

        token_instances = []
        node_indices = torch.zeros_like(predictions)
        pos_tags = torch.zeros_like(predictions)
        for i, index in enumerate(predictions.tolist()):
            instance_meta = meta_data[batch_index(i)]
            pos_tag_lut = instance_meta["pos_tag_lut"]
            target_dynamic_vocab = instance_meta["target_dynamic_vocab"]
            # Generation.
            if index < self._vocab_size:
                token = self.vocab.get_token_from_index(index, self._target_output_namespace)
                node_index = default_node_index
                pos_tag = pos_tag_lut.get(token, DEFAULT_OOV_TOKEN)
            # Source-side copy.
            elif self._vocab_size <= index < self._vocab_size + source_dynamic_vocab_size:
                index -= self._vocab_size
                source_dynamic_vocab = instance_meta["source_dynamic_vocab"]
                token = source_dynamic_vocab.get_token_from_idx(index)
                node_index = default_node_index
                pos_tag = pos_tag_lut.get(token, DEFAULT_OOV_TOKEN)
            # Target-side copy.
            else:
                index -= (self._vocab_size + source_dynamic_vocab_size)
                token = target_dynamic_vocab[index]
                node_index = index
                pos_tag = pos_tag_lut.get(token, DEFAULT_OOV_TOKEN)

            target_token = TextField([Token(token)], instance_meta["target_token_indexers"])
            token_instances.append(Instance({"target_tokens": target_token}))
            node_indices[i] = node_index
            pos_tags[i] = self.vocab.get_token_index(pos_tag, self._pos_tag_namespace)
            if last_decoding_step != -1:  # At <BOS>, we set the last decoding step to -1.
                target_attention_map[i, last_decoding_step, node_index] = 1
                target_dynamic_vocab[node_index] = token

        # Covert tokens to tensors.
        batch = Batch(token_instances)
        batch.index_instances(self.vocab)
        padding_lengths = batch.get_padding_lengths()
        tokens = {}
        for key, tensor in batch.as_tensor_dict(padding_lengths)["target_tokens"].items():
            tokens[key] = tensor.type_as(predictions)

        return dict(
            tokens=tokens,
            # [group_size, 1]
            node_indices=node_indices.unsqueeze(1),
            pos_tags=pos_tags.unsqueeze(1),
        )

    def _read_node_predictions(self,
                               predictions: torch.Tensor,
                               meta_data: List[Dict],
                               source_dynamic_vocab_size: int):
        """
        :param predictions: [batch_size, max_steps].
        :return:
            node_predictions: [batch_size, max_steps].
            edge_head_mask: [batch_size, max_steps, max_steps].
        """
        batch_size, max_steps = predictions.size()
        node_predictions = []
        edge_head_mask = predictions.new_ones((batch_size, max_steps, max_steps))
        edge_head_mask = torch.tril(edge_head_mask, diagonal=-1).long()
        for i in range(batch_size):
            nodes = []
            instance_meta = meta_data[i]
            source_dynamic_vocab = instance_meta["source_dynamic_vocab"]
            target_dynamic_vocab = instance_meta["target_dynamic_vocab"]
            prediction_list = predictions[i].tolist()
            for j, index in enumerate(prediction_list):
                if index == self._vocab_eos_index:
                    break
                if index < self._vocab_size:
                    node = self.vocab.get_token_from_index(index, self._target_output_namespace)
                elif self._vocab_size <= index < self._vocab_size + source_dynamic_vocab_size:
                    node = source_dynamic_vocab.get_token_from_idx(index - self._vocab_size)
                else:
                    node = target_dynamic_vocab[index - self._vocab_size - source_dynamic_vocab_size]
                    for k, antecedent in enumerate(prediction_list[:j]):
                        if index == antecedent:
                            edge_head_mask[i, j, k] = 0
                nodes.append(node)
            node_predictions.append(nodes)
        return node_predictions, edge_head_mask

    def _take_one_step_node_prediction(self,
                                       last_predictions: torch.Tensor,
                                       state: Dict[str, torch.Tensor],
                                       auxiliaries: Dict) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        inputs = self._prepare_next_inputs(
            predictions=last_predictions,
            target_attention_map=state["target_attention_map"],
            meta_data=auxiliaries["instance_meta"],
            batch_size=auxiliaries["batch_size"],
            last_decoding_step=auxiliaries["last_decoding_step"],
            source_dynamic_vocab_size=auxiliaries["source_dynamic_vocab_size"]
        )

        decoder_inputs = torch.cat([
            self._decoder_token_embedder(inputs["tokens"]),
            self._decoder_node_index_embedding(inputs["node_indices"]),
            self._decoder_pos_embedding(inputs["pos_tags"])
        ], dim=2)
        hidden_states = (
            # [num_layers, batch_size, hidden_vector_dim]
            state["hidden_state_1"].permute(1, 0, 2),
            state["hidden_state_2"].permute(1, 0, 2),
        )
        decoding_outputs = self._decoder.one_step_forward(
            input_tensor=decoder_inputs,
            source_memory_bank=state["source_memory_bank"],
            source_mask=state["source_mask"],
            target_memory_bank=state.get("target_memory_bank", None),
            decoding_step=auxiliaries["last_decoding_step"] + 1,
            total_decoding_steps=self._max_decoding_length,
            input_feed=state.get("input_feed", None),
            hidden_state=hidden_states,
            coverage=state.get("coverage", None)
        )
        state["input_feed"] = decoding_outputs["attentional_tensor"]
        state["coverage"] = decoding_outputs["coverage"]
        state["hidden_state_1"] = decoding_outputs["hidden_state"][0].permute(1, 0, 2)
        state["hidden_state_2"] = decoding_outputs["hidden_state"][1].permute(1, 0, 2)
        state["rnn_output"] = decoding_outputs["rnn_output"]
        if state.get("target_memory_bank", None) is None:
            state["target_memory_bank"] = decoding_outputs["attentional_tensor"]
        else:
            state["target_memory_bank"] = torch.cat(
                [state["target_memory_bank"], decoding_outputs["attentional_tensor"]], 1
            )

        node_prediction_outputs = self._extended_pointer_generator(
            inputs=decoding_outputs["attentional_tensor"],
            source_attention_weights=decoding_outputs["source_attention_weights"],
            target_attention_weights=decoding_outputs["target_attention_weights"],
            source_attention_map=state["source_attention_map"],
            target_attention_map=state["target_attention_map"]
        )
        log_probs = (node_prediction_outputs["hybrid_prob_dist"] + self._eps).log()

        auxiliaries["last_decoding_step"] += 1

        return log_probs, state

    def _test_forward(self, inputs: Dict) -> Dict:
        encoding_outputs = self._encode(
            tokens=inputs["source_tokens"],
            pos_tags=inputs["source_pos_tags"],
            anonymization_tags=inputs["source_anonymization_tags"],
            subtoken_ids=inputs["source_subtoken_ids"],
            token_recovery_matrix=inputs["source_token_recovery_matrix"],
            mask=inputs["source_mask"]
        )

        start_predictions, start_state, auxiliaries = self._prepare_decoding_start_state(inputs, encoding_outputs)
        # all_predictions: [batch_size, beam_size, max_steps]
        # rnn_outputs: [batch_size, beam_size, max_steps, hidden_vector_dim]
        # log_probs: [batch_size, beam_size]
        all_predictions, rnn_outputs, log_probs = self._beam_search.search(
            start_predictions=start_predictions,
            start_state=start_state,
            step=lambda x, y: self._take_one_step_test_forward(x, y, auxiliaries),
            tracked_state_name="rnn_output"
        )
        node_predictions, edge_head_mask = self._read_node_predictions(
            # Remove the last one because we can't get the RNN state for the last one.
            predictions=all_predictions[:, 0, :-1],
            meta_data=inputs["instance_meta"],
            source_dynamic_vocab_size=inputs["source_dynamic_vocab_size"]
        )

        edge_predictions = self._parse(
            # Remove the first RNN state because it represents <BOS>.
            rnn_outputs=rnn_outputs[:, 0],
            edge_head_mask=edge_head_mask
        )

        edge_head_predictions, edge_type_predictions = self._read_edge_predictions(edge_predictions)

        return dict(
            node_predictions=node_predictions,
            edge_head_predictions=edge_head_predictions,
            edge_type_predictions=edge_type_predictions
        )

    def _training_forward(self, inputs: Dict) -> Dict[str, torch.Tensor]:
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
