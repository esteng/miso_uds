from typing import Dict, Tuple
import logging

from overrides import overrides
import torch
from allennlp.data import Vocabulary
from allennlp.models import Model
from allennlp.modules import TextFieldEmbedder, Embedding, InputVariationalDropout, Seq2SeqEncoder
from allennlp.training.metrics import Metric
from allennlp.data.vocabulary import DEFAULT_PADDING_TOKEN
from allennlp.common.util import START_SYMBOL, END_SYMBOL

from miso.modules.seq2seq_encoders import Seq2SeqBertEncoder
from miso.modules.decoders import RNNDecoder
from miso.modules.generators import ExtendedPointerGenerator
from miso.modules.parsers import DeepTreeParser

from miso.modules.seq2vec_encoders.cnn_encoder import CnnEncoder
from miso.modules.seq2seq_encoders.pytorch_seq2seq_wrapper import PytorchSeq2SeqWrapper
from miso.modules.stacked_bilstm import StackedBidirectionalLstm
from miso.modules.stacked_lstm import StackedLstm
from miso.modules.attention_layers.global_attention import GlobalAttention
from miso.modules.attention import DotProductAttention
from miso.modules.attention import MLPAttention
from miso.modules.attention import BiaffineAttention
from miso.modules.decoders.pointer_generator import PointerGenerator
from miso.utils.nn import get_text_field_mask
from miso.utils.string import find_similar_token
# The following imports are added for mimick testing.
from miso.data.dataset_builder import load_dataset_reader
from miso.predictors.predictor import Predictor
from miso.commands.predict import _PredictManager
import subprocess

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


def character_tensor_from_token_tensor(
        token_tensor,
        vocab,
        character_tokenizer,
        namespace=dict(tokens="decoder_token_ids", characters="decoder_token_characters")
):
    token_str = [vocab.get_token_from_index(i, namespace["tokens"]) for i in token_tensor.view(-1).tolist()]
    max_char_len = max([len(token) for token in token_str])
    indices = []
    for token in token_str:
        token_indices = [vocab.get_token_index(vocab._padding_token) for _ in range(max_char_len)]
        for char_i, character in enumerate(character_tokenizer.tokenize(token)):
            index = vocab.get_token_index(character.text, namespace["characters"])
            token_indices[char_i] = index
        indices.append(token_indices)

    return torch.tensor(indices).view(token_tensor.size(0), token_tensor.size(1), -1).type_as(token_tensor)


@Model.register("transductive_parser")
class TransductiveParser(Model):

    def __init__(self,
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
                 # metrics:
                 node_pred_metric: Metric,
                 edge_pred_metric: Metric,
                 # misc
                 vocab: Vocabulary,
                 target_output_namespace: str,
                 edge_type_namespace: str,
                 dropout: float,
                 beam_size: int,
                 max_decoding_length: int,
                 eps: float = 1e-20,
                 ) -> None:
        super().__init__()
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
        self._node_pred_metric = node_pred_metric
        self._edge_pred_metric = edge_pred_metric

        self._vocab = vocab
        self._dropout = InputVariationalDropout(p=dropout)
        self._beam_size = beam_size
        self._max_decoding_length = max_decoding_length
        self._eps = eps

        # dynamic initialization
        self._target_output_namespace = target_output_namespace
        self._edge_type_namespace = edge_type_namespace
        self._vocab_size = self._vocab.get_vocab_size(target_output_namespace)
        self._vocab_pad_index = self._vocab.get_token_index(DEFAULT_PADDING_TOKEN, target_output_namespace)
        self._vocab_bos_index = self._vocab.get_token_index(START_SYMBOL, target_output_namespace)
        self._extended_pointer_generator.reset_vocab_linear(
            vocab_size=vocab.get_vocab_size(target_output_namespace),
            vocab_pad_index=self._vocab_pad_index
        )
        self._tree_parser.reset_edge_label_bilinear(num_labels=vocab.get_vocab_size(edge_type_namespace))

    def get_metrics(self, reset: bool = False, mimick_test: bool = False):
        metrics = dict()
        if mimick_test and self.test_config:
            metrics = self.mimick_test()
        generator_metrics = self.generator.metrics.get_metric(reset)
        tree_decoder_metrics = self.tree_decoder.metrics.get_metric(reset)
        metrics.update(generator_metrics)
        metrics.update(tree_decoder_metrics)
        if 'F1' not in metrics:
            metrics['F1'] = metrics['all_acc']
        return metrics

    def mimick_test(self):
        word_splitter = None
        if self.use_bert:
            word_splitter = self.test_config.get('word_splitter', None)
        dataset_reader = load_dataset_reader('AMR', word_splitter=word_splitter)
        dataset_reader.set_evaluation()
        predictor = Predictor.by_name('STOG')(self, dataset_reader)
        manager = _PredictManager(
            predictor,
            self.test_config['data'],
            self.test_config['prediction'],
            self.test_config['batch_size'],
            False,
            True,
            0
        )
        try:
            logger.info('Mimicking test...')
            manager.run()
        except Exception as e:
            logger.info('Exception threw out when running the manager.')
            logger.error(e, exc_info=True)
            return {}
        try:
            logger.info('Computing the Smatch score...')
            result = subprocess.check_output([
                self.test_config['eval_script'],
                self.test_config['smatch_dir'],
                self.test_config['data'],
                self.test_config['prediction']
            ]).decode().split()
            result = list(map(float, result))
            return dict(PREC=result[0] * 100, REC=result[1] * 100, F1=result[2] * 100)
        except Exception as e:
            logger.info('Exception threw out when computing smatch.')
            logger.error(e, exc_info=True)
            return {}

    def _prepare_inputs(self, raw_inputs: Dict) -> Dict:
        # [batch_size, source_seq_length]
        source_tokens = raw_inputs["source_tokens"]
        source_pos_tags = raw_inputs["source_pos_tags"]
        source_anonymization_tags = raw_inputs["source_anonymization_tags"]
        source_mask = get_text_field_mask(source_tokens)
        source_subtoken_ids = raw_inputs.get("source_subtoken_ids", None)
        if source_subtoken_ids is not None:
            source_subtoken_ids = source_subtoken_ids.long()
        source_token_recovery_matrix = raw_inputs.get("source_token_recovery_matrix", None)
        if source_token_recovery_matrix is not None:
            source_token_recovery_matrix = source_token_recovery_matrix.long()

        # [batch_size, target_seq_length]
        target_tokens = raw_inputs["target_tokens"]
        target_pos_tags = raw_inputs["target_pos_tags"]
        target_node_indices = raw_inputs["target_node_indices"]

        # [batch_size, target_seq_length]
        generation_outputs = raw_inputs["generation_outputs"]["tokens"][:, 1:]
        source_copy_indices = raw_inputs["source_copy_indices"][:, 1:]
        target_copy_indices = raw_inputs["target_copy_indices"][:, 1:]

        # [batch, target_seq_length, target_seq_length + 1(sentinel)]
        target_attention_map = raw_inputs["target_attention_map"][:, 1:]  # exclude BOS
        # [batch, 1(unk) + source_seq_length, dynamic_vocab_size]
        # Exclude unk and the last pad.
        source_attention_map = raw_inputs["source_attention_map"][:, 1:-1]

        edge_heads = raw_inputs["edge_heads"]
        edge_types = raw_inputs["edge_types"]
        node_mask = raw_inputs['node_mask']
        edge_mask = raw_inputs['edge_mask']

        inputs = raw_inputs[:]
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

        return inputs

    @overrides
    def forward(self, **raw_inputs: Dict) -> Dict[str, torch.Tensor]:
        inputs = self._prepare_inputs(raw_inputs)
        if self.training:
            return self._training_forward(inputs)

        encoder_outputs = self.encode(
            encoder_inputs['bert_token'],
            encoder_inputs['token_subword_index'],
            encoder_inputs['token'],
            encoder_inputs['pos_tag'],
            encoder_inputs['anonym_indicator'],
            encoder_inputs['char'],
            encoder_inputs['mask']
        )

        if for_training:
            decoder_outputs = self.decode_for_training(
                decoder_inputs['token'],
                decoder_inputs['pos_tag'],
                decoder_inputs['char'],
                decoder_inputs['coref'],
                encoder_outputs['memory_bank'],
                encoder_inputs['mask'],
                encoder_outputs['final_states'],
            )

            generator_output = self.generator(
                decoder_outputs['memory_bank'],
                decoder_outputs['copy_attentions'],
                generator_inputs['copy_attention_maps'],
                decoder_outputs['coref_attentions'],
                generator_inputs['coref_attention_maps']
            )

            generator_loss_output = self.generator.compute_loss(
                generator_output['probs'],
                generator_output['predictions'],
                generator_inputs['vocab_targets'],
                generator_inputs['copy_targets'],
                generator_output['source_dynamic_vocab_size'],
                generator_inputs['coref_targets'],
                generator_output['target_dynamic_vocab_size'],
                decoder_outputs['coverage_records'],
                decoder_outputs['copy_attentions']
            )

            tree_decoder_outputs = self.tree_decode(
                decoder_outputs['rnn_memory_bank'],
                parser_inputs['edge_heads'],
                parser_inputs['edge_labels'],
                parser_inputs['edge_mask'],
                parser_inputs['node_mask']
            )

            return dict(
                loss=generator_loss_output['loss'] + tree_decoder_outputs['loss'],
                token_loss=generator_loss_output['total_loss'],
                edge_loss=tree_decoder_outputs['total_loss'],
                num_tokens=generator_loss_output['num_tokens'],
                num_nodes=tree_decoder_outputs['num_instances']
            )

        else:

            invalid_indexes = dict(
                source_copy=batch.get('source_copy_invalid_ids', None),
                vocab=[set(self.punctuation_ids) for _ in range(len(batch['tag_lut']))]
            )

            return dict(
                encoder_memory_bank=encoder_outputs['memory_bank'],
                encoder_mask=encoder_inputs['mask'],
                encoder_final_states=encoder_outputs['final_states'],
                copy_attention_maps=generator_inputs['copy_attention_maps'],
                copy_vocabs=batch['src_copy_vocab'],
                tag_luts=batch['tag_lut'],
                invalid_indexes=invalid_indexes
            )

    def _compute_node_prediction_loss(self,
                                      prob_dist: torch.Tensor,
                                      generation_outputs: torch.Tensor,
                                      source_copy_indices: torch.Tensor,
                                      target_copy_indices: torch.Tensor,
                                      source_dynamic_vocab_size: int,
                                      source_attention_weights: torch.Tensor = None,
                                      coverage_history: torch.Tensor = None):
        """
        Compute the node prediction loss based on the final hybrid probability distribtuion.

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
        self._node_pred_metric(
            loss=loss,
            prediction=prediction,
            generation_outputs=_generation_outputs,
            valid_generation_mask=valid_generation_mask,
            source_copy_indices=_source_copy_indices,
            valid_source_copy_mask=valid_source_copy_mask,
            target_copy_indices=_target_copy_indices,
            valid_target_copy_mask=valid_target_copy_mask
        )

        return {"prediction": prediction, "loss": loss, "num_nodes": num_nodes}

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
        if subtoken_ids is not None:
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

    def _parse(self,
               rnn_outputs: torch.Tensor,
               mask: torch.Tensor,
               edge_heads: torch.Tensor) -> Dict:
        # Exclude <BOS>.
        rnn_outputs = self._dropout(rnn_outputs[:, 1:])
        parser_outputs = self._tree_parser(
            query=rnn_outputs,
            key=rnn_outputs,
            mask=mask,
            gold_edge_head=edge_heads
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
            mask=inputs["edge_mask"],
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
        return {}

    def tree_decode(self, memory_bank, edge_heads, edge_labels, edge_mask, node_mask):
        # Exclude the BOS symbol.
        memory_bank = memory_bank[:, 1:]
        memory_bank = self.decoder_embedding_dropout(memory_bank)
        return self.tree_decoder.get_loss(
            memory_bank, memory_bank, edge_heads, edge_labels, edge_mask, node_mask)

    def _get_encoder_char_cnn_output(self, chars):
        # [batch, num_tokens, num_chars, embedding_size]
        char_embeddings = self.encoder_char_embedding(chars)
        batch_size, num_tokens, num_chars, _ = char_embeddings.size()
        char_embeddings = char_embeddings.view(batch_size * num_tokens, num_chars, -1)
        char_cnn_output = self.encoder_char_cnn(char_embeddings, None)
        char_cnn_output = char_cnn_output.view(batch_size, num_tokens, -1)
        return char_cnn_output

    def _get_decoder_char_cnn_output(self, chars):
        # [batch, num_tokens, num_chars, embedding_size]
        char_embeddings = self.decoder_char_embedding(chars)
        batch_size, num_tokens, num_chars, _ = char_embeddings.size()
        char_embeddings = char_embeddings.view(batch_size * num_tokens, num_chars, -1)
        char_cnn_output = self.decoder_char_cnn(char_embeddings, None)
        char_cnn_output = char_cnn_output.view(batch_size, num_tokens, -1)
        return char_cnn_output

    def decode(self, input_dict):
        memory_bank = input_dict['encoder_memory_bank']
        mask = input_dict['encoder_mask']
        states = input_dict['encoder_final_states']
        copy_attention_maps = input_dict['copy_attention_maps']
        copy_vocabs = input_dict['copy_vocabs']
        tag_luts = input_dict['tag_luts']
        invalid_indexes = input_dict['invalid_indexes']

        if self.beam_size == 0:
            generator_outputs = self.decode_with_pointer_generator(
                memory_bank, mask, states, copy_attention_maps, copy_vocabs, tag_luts, invalid_indexes)
        else:
            generator_outputs = self.beam_search_with_pointer_generator(
                memory_bank, mask, states, copy_attention_maps, copy_vocabs, tag_luts, invalid_indexes)

        return dict(
            nodes=generator_outputs['predictions'],
            heads=generator_outputs['edge_heads'],
            head_labels=generator_outputs['edge_labels'],
            corefs=generator_outputs['coref_indexes'],
        )

    def beam_search_with_pointer_generator(
            self, memory_bank, mask, states, copy_attention_maps, copy_vocabs, tag_luts, invalid_indices):
        batch_size = memory_bank.size(0)
        beam_size = self.beam_size

        #  new_order is used to replicate tensors for different beam
        new_order = torch.arange(batch_size).view(-1, 1).repeat(1, beam_size).view(-1).type_as(mask)

        # special token indices
        bos_token = self.vocab.get_token_index(START_SYMBOL, "decoder_token_ids")
        eos_token = self.vocab.get_token_index(END_SYMBOL, "decoder_token_ids")
        pad_token = self.vocab.get_token_index(DEFAULT_PADDING_TOKEN, "decoder_token_ids")

        bucket = [[] for i in range(batch_size)]
        bucket_max_score = [-1e8 for i in range(batch_size)]

        def flatten(tensor):
            sizes = list(tensor.size())
            assert len(sizes) >= 2
            assert sizes[0] == batch_size and sizes[1] == beam_size

            if len(sizes) == 2:
                new_sizes = [sizes[0] * sizes[1], 1]
            else:
                new_sizes = [sizes[0] * sizes[1]] + sizes[2:]

            return tensor.contiguous().view(new_sizes)

        def fold(tensor):
            sizes = list(tensor.size())
            new_sizes = [batch_size, beam_size]

            if len(sizes) >= 2:
                new_sizes = [batch_size, beam_size] + sizes[1:]

            return tensor.view(new_sizes)

        def beam_select_2d(input, indices):
            # input [batch_size, beam_size, ......]
            # indices [batch_size, beam_size]
            input_size = list(input.size())
            indices_size = list(indices.size())
            assert len(indices_size) == 2
            assert len(input_size) >= 2
            assert input_size[0] == indices_size[0]
            assert input_size[1] == indices_size[1]

            return input.view(
                [input_size[0] * input_size[1]] + input_size[2:]
            ).index_select(
                0,
                (
                        torch.arange(
                            indices_size[0]
                        ).unsqueeze(1).expand_as(indices).type_as(indices) * indices_size[1] + indices
                ).view(-1)
            ).view(input_size)

        def beam_select_1d(input, indices):
            input_size = list(input.size())
            indices_size = list(indices.size())
            assert len(indices_size) == 2
            assert len(input_size) > 1
            assert input_size[0] == indices_size[0] * indices_size[1]

            return input.index_select(
                0,
                (
                        torch.arange(
                            indices_size[0]
                        ).unsqueeze(1).expand_as(indices).type_as(indices) * indices_size[1] + indices
                ).view(-1)
            ).view(input_size)

        def update_tensor_buff(key, step, beam_indices, tensor, select_input=True):
            if step == 0 and beam_buffer[key] is None:
                beam_buffer[key] = tensor.new_zeros(
                    batch_size,
                    beam_size,
                    self.max_decode_length,
                    tensor.size(-1)
                )

            if select_input:
                beam_buffer[key][:, :, step] = fold(tensor.squeeze(1))
                beam_buffer[key] = beam_select_2d(beam_buffer[key], beam_indices)
            else:
                beam_buffer[key] = beam_select_2d(beam_buffer[key], beam_indices)
                beam_buffer[key][:, :, step] = fold(tensor.squeeze(1))

        def get_decoder_input(tokens, pos_tags, corefs):
            token_embeddings = self.decoder_token_embedding(tokens)
            pos_tag_embeddings = self.decoder_pos_embedding(pos_tags)
            coref_embeddings = self.decoder_coref_embedding(corefs)

            if self.use_char_cnn:
                # TODO: get chars from tokens.
                # [batch_size, 1, num_chars]
                chars = character_tensor_from_token_tensor(
                    tokens,
                    self.vocab,
                    self.character_tokenizer
                )
                if chars.size(-1) < 3:
                    chars = torch.cat(
                        (
                            chars,
                            chars.new_zeros(
                                (
                                    chars.size(0),
                                    chars.size(1),
                                    3 - chars.size(2)
                                )
                            )
                        ),
                        2
                    )

                char_cnn_output = self._get_decoder_char_cnn_output(chars)
                decoder_inputs = torch.cat(
                    [token_embeddings, pos_tag_embeddings,
                     coref_embeddings, char_cnn_output], 2)
            else:
                decoder_inputs = torch.cat(
                    [token_embeddings, pos_tag_embeddings, coref_embeddings], 2)

            return self.decoder_embedding_dropout(decoder_inputs)

        def repeat_list_item(input_list, n):
            new_list = []
            for item in input_list:
                new_list += [item] * n
            return new_list

        beam_buffer = {}
        beam_buffer["predictions"] = mask.new_full(
            (batch_size, beam_size, self.max_decode_length),
            pad_token
        )

        beam_buffer["edge_labels"] = mask.new_full(
            (batch_size, beam_size, self.max_decode_length),
            pad_token
        )

        beam_buffer["edge_heads"] = mask.new_full(
            (batch_size, beam_size, self.max_decode_length),
            pad_token
        )

        beam_buffer["coref_indexes"] = memory_bank.new_zeros(
            batch_size,
            beam_size,
            self.max_decode_length
        )

        beam_buffer["decoder_mask"] = memory_bank.new_ones(
            batch_size,
            beam_size,
            self.max_decode_length
        )

        beam_buffer["decoder_inputs"] = None
        beam_buffer["decoder_memory_bank"] = None
        beam_buffer["decoder_rnn_memory_bank"] = None

        # beam_buffer["source_attentions"] = None
        # beam_buffer["copy_attentions"] = []
        # beam_buffer["coref_attentions"] = []

        beam_buffer["scores"] = memory_bank.new_zeros(batch_size, beam_size, 1)
        beam_buffer["scores"][:, 1:] = -float(1e8)

        # inter media variables
        variables = {}

        variables["input_tokens"] = beam_buffer["predictions"].new_full(
            (batch_size * beam_size, 1),
            bos_token
        )

        variables["pos_tags"] = mask.new_full(
            (batch_size * beam_size, 1),
            self.vocab.get_token_index(DEFAULT_OOV_TOKEN, "pos_tags")
        )

        variables["corefs"] = mask.new_zeros(batch_size * beam_size, 1)

        variables["input_feed"] = memory_bank.new_zeros(
            batch_size * beam_size,
            1,
            self.decoder.rnn_cell.hidden_size
        )

        variables["coref_inputs"] = []
        variables["states"] = [item.index_select(1, new_order) for item in states]

        variables["prev_tokens"] = mask.new_full(
            (batch_size * beam_size, 1), bos_token)

        # A sparse indicator matrix mapping each node to its index in the dynamic vocab.
        # Here the maximum size of the dynamic vocab is just max_decode_length.
        variables["coref_attention_maps"] = memory_bank.new_zeros(
            batch_size * beam_size, self.max_decode_length, self.max_decode_length + 1
        )
        # A matrix D where the element D_{ij} is for instance i the real vocab index of
        # the generated node at the decoding step `i'.
        variables["coref_vocab_maps"] = mask.new_zeros(batch_size * beam_size, self.max_decode_length + 1)

        variables["coverage"] = None
        if self.use_coverage:
            variables["coverage"] = memory_bank.new_zeros(batch_size * beam_size, 1, memory_bank.size(1))

        for key in invalid_indices.keys():
            invalid_indices[key] = repeat_list_item(invalid_indices[key], beam_size)

        for step in range(self.max_decode_length):  # one extra step for EOS marker
            # 1. Decoder inputs
            # decoder_inputs : [ batch_size * beam_size, model_dim]
            decoder_inputs = get_decoder_input(
                variables["input_tokens"],
                variables["pos_tags"],
                variables["corefs"]
            )

            # 2. Decode one stepi.
            _rnn_outputs, _ = self.decoder.one_step_rnn_forward(
                decoder_inputs,
                variables["states"],
                variables["input_feed"]
            )

            if step != 0:
                queries = _rnn_outputs
                if step == 1:
                    keys, edge_mask = None, None
                else:
                    keys = flatten(beam_buffer["decoder_rnn_memory_bank"])[
                           :, 1:step, :
                           ]
                    coref_indexes_flatten = flatten(
                        beam_buffer["coref_indexes"]
                    )
                    coref_history = \
                        coref_indexes_flatten[:, :step - 1]
                    coref_current = \
                        coref_indexes_flatten[:, step - 1].unsqueeze(1)
                    edge_mask = coref_history.ne(coref_current).unsqueeze(1)
                heads, labels = self.tree_decoder(queries, keys, edge_mask)

                update_tensor_buff(
                    "edge_heads", step, beam_indices, heads, False
                )
                update_tensor_buff(
                    "edge_labels", step, beam_indices, labels, False
                )
                # edge_heads += [heads]
                # edge_labels += [labels]

            decoder_output_dict = self.decoder.one_step_forward(
                decoder_inputs,
                memory_bank.index_select(0, new_order),
                mask.index_select(0, new_order),
                variables["states"],
                variables["input_feed"],
                None,
                None,
                variables["coref_inputs"],
                variables["coverage"],
                step,
                1
            )

            _decoder_outputs = decoder_output_dict['decoder_output']
            _rnn_outputs = decoder_output_dict['rnn_output']
            _copy_attentions = decoder_output_dict['source_copy_attention']
            _coref_attentions = decoder_output_dict['target_copy_attention']
            states = decoder_output_dict['rnn_hidden_state']
            input_feed = decoder_output_dict['input_feed']
            coverage = decoder_output_dict['coverage']

            # coverage_records = decoder_output_dict['coverage_records']

            # 3. Run pointer/generator.instance.fields['src_copy_vocab'].metadata
            if step == 0:
                _coref_attention_maps = variables["coref_attention_maps"][:, :step + 1]
            else:
                _coref_attention_maps = variables["coref_attention_maps"][:, :step]

            generator_output = self.generator(
                _decoder_outputs,
                _copy_attentions,
                copy_attention_maps.index_select(0, new_order),
                _coref_attentions,
                _coref_attention_maps,
                invalid_indices
            )

            # new word probs
            word_lprobs = fold(torch.log(1e-8 + generator_output['probs'].squeeze(1)))

            if self.use_coverage:
                coverage_loss = torch.sum(
                    torch.min(coverage, _copy_attentions),
                    dim=2
                )
            else:
                coverage_loss = word_lprobs.new_zeros(batch_size, beam_size, 1)

            new_all_scores = \
                word_lprobs \
                + beam_buffer["scores"].expand_as(word_lprobs) \
                - coverage_loss.view(batch_size, beam_size, 1).expand_as(word_lprobs)

            # top beam_size hypos
            # new_hypo_indices : [batch_size, beam_size * 2]
            new_hypo_scores, new_hypo_indices = torch.topk(
                new_all_scores.view(batch_size, -1).contiguous(),
                beam_size * 2,
                dim=-1
            )

            new_token_indices = torch.fmod(new_hypo_indices, word_lprobs.size(-1))

            eos_token_mask = new_token_indices.eq(eos_token)

            eos_beam_indices_offset = torch.div(
                new_hypo_indices,
                word_lprobs.size(-1)
            )[:, :beam_size] + new_order.view(batch_size, beam_size) * beam_size

            eos_beam_indices_offset = eos_beam_indices_offset.masked_select(eos_token_mask[:, :beam_size])

            if eos_beam_indices_offset.numel() > 0:
                for index in eos_beam_indices_offset.tolist():
                    eos_batch_idx = int(index / beam_size)
                    eos_beam_idx = index % beam_size
                    hypo_score = float(new_hypo_scores[eos_batch_idx, eos_beam_idx]) / (step + 1)
                    if step > 0 and hypo_score > bucket_max_score[eos_batch_idx] and eos_beam_idx == 0:
                        bucket_max_score[eos_batch_idx] = hypo_score
                        bucket[eos_batch_idx] += [
                            {
                                key: tensor[eos_batch_idx, eos_beam_idx].unsqueeze(0) for key, tensor in
                            beam_buffer.items()
                            }
                        ]
                        # bucket[eos_batch_idx][-1]['decoder_inputs'][0, step] = decoder_inputs[index, 0]
                        # bucket[eos_batch_idx][-1]['decoder_rnn_memory_bank'][0, step] = _rnn_outputs[index, 0]
                        # bucket[eos_batch_idx][-1]['decoder_memory_bank'][0, step] = _decoder_outputs[index, 0]
                        # bucket[eos_batch_idx][-1]['decoder_mask'][0, step] = 1

                eos_token_mask = eos_token_mask.type_as(new_hypo_scores)
                active_hypo_scores, active_sort_indices = torch.sort(
                    (1 - eos_token_mask) * new_hypo_scores + eos_token_mask * - float(1e8),
                    descending=True
                )

                active_sort_indices_offset = active_sort_indices \
                                             + 2 * beam_size * torch.arange(
                    batch_size
                ).unsqueeze(1).expand_as(active_sort_indices).type_as(active_sort_indices)
                active_hypo_indices = new_hypo_indices.view(batch_size * beam_size * 2)[
                    active_sort_indices_offset.view(batch_size * beam_size * 2)
                ].view(batch_size, -1)

                new_hypo_scores = active_hypo_scores
                new_hypo_indices = active_hypo_indices
                new_token_indices = torch.fmod(new_hypo_indices, word_lprobs.size(-1))

            new_hypo_indices = new_hypo_indices[:, :beam_size]
            new_hypo_scores = new_hypo_scores[:, :beam_size]
            new_token_indices = new_token_indices[:, :beam_size]

            # find out which beam the new hypo came from and what is the new token
            beam_indices = torch.div(new_hypo_indices, word_lprobs.size(-1))
            if step == 0:
                decoder_mask_input = []
            else:

                decoder_mask_input = beam_select_2d(
                    beam_buffer["decoder_mask"],
                    beam_indices
                ).view(batch_size * beam_size, -1)[:, :step].split(1, 1)

            variables["coref_attention_maps"] = beam_select_1d(variables["coref_attention_maps"], beam_indices)
            variables["coref_vocab_maps"] = beam_select_1d(variables["coref_vocab_maps"], beam_indices)

            input_tokens, _predictions, pos_tags, corefs, _mask = self._update_maps_and_get_next_input(
                step,
                flatten(new_token_indices).squeeze(1),
                generator_output['source_dynamic_vocab_size'],
                variables["coref_attention_maps"],
                variables["coref_vocab_maps"],
                repeat_list_item(copy_vocabs, beam_size),
                decoder_mask_input,
                repeat_list_item(tag_luts, beam_size),
                invalid_indices
            )

            beam_buffer["scores"] = new_hypo_scores.unsqueeze(2)

            update_tensor_buff(
                "decoder_inputs", step, beam_indices, decoder_inputs
            )
            update_tensor_buff(
                "decoder_memory_bank", step, beam_indices, _decoder_outputs
            )
            update_tensor_buff(
                "decoder_rnn_memory_bank", step, beam_indices, _rnn_outputs
            )

            # update_tensor_buff("source_attentions", step, _source_attentions)
            # update_tensor_buff("copy_attentions", step, _copy_attentions)
            # update_tensor_buff("coref_attentions", step, _coref_attentions)

            update_tensor_buff(
                "predictions", step, beam_indices, _predictions, False
            )
            update_tensor_buff(
                "coref_indexes", step, beam_indices, corefs, False
            )
            update_tensor_buff(
                "decoder_mask", step, beam_indices, _mask, False
            )

            variables["input_tokens"] = input_tokens
            variables["pos_tags"] = pos_tags
            variables["corefs"] = corefs

            variables["states"] = [
                state.index_select(1, new_order * beam_size + beam_indices.view(-1)) for state in states]
            variables["input_feed"] = beam_select_1d(input_feed, beam_indices)
            variables["coref_inputs"].append(_decoder_outputs)
            variables["coref_inputs"] = list(
                beam_select_1d(
                    torch.cat(variables["coref_inputs"], 1),
                    beam_indices
                ).split(1, 1)
            )
            if self.use_coverage:
                variables["coverage"] = beam_select_1d(coverage, beam_indices)
            else:
                variables["coverage"] = None

        for batch_idx, item in enumerate(bucket):
            if len(item) == 0:
                bucket[batch_idx].append(
                    {
                        key: tensor[batch_idx, 0].unsqueeze(0) for key, tensor in beam_buffer.items()
                    }
                )

        return_dict = {}

        for key in bucket[0][-1].keys():
            return_dict[key] = torch.cat(
                [hypos[-1][key] for hypos in bucket],
                dim=0
            )

        # return_dict["decoder_inputs"] = return_dict["decoder_inputs"][:, 1:]
        # return_dict["decoder_memory_bank"] = return_dict["decoder_memory_bank"][:, 1:]
        # return_dict["decoder_rnn_memory_bank"] = return_dict["decoder_rnn_memory_bank"][:, 1:]

        # return_dict["decoder_mask"] = 1 - return_dict["decoder_mask"]

        return_dict["predictions"] = return_dict["predictions"][:, :-1]
        return_dict["predictions"][return_dict["predictions"] == pad_token] = eos_token
        return_dict["edge_heads"] = return_dict["edge_heads"][:, 1:]
        return_dict["edge_labels"] = return_dict["edge_labels"][:, 1:]
        return_dict["coref_indexes"] = return_dict["coref_indexes"][:, :-1]
        return_dict["decoder_mask"] = return_dict["predictions"] != eos_token  # return_dict["decoder_mask"][:, :-1]
        # return_dict["scores"] = torch.div(return_dict["scores"], return_dict["decoder_mask"].sum(1, keepdim=True).type_as(return_dict["scores"]))

        return return_dict

    def decode_with_pointer_generator(
            self, memory_bank, mask, states, copy_attention_maps, copy_vocabs,
            tag_luts, invalid_indexes):
        # [batch_size, 1]
        batch_size = memory_bank.size(0)
        # Input
        tokens = torch.ones(batch_size, 1) * self.vocab.get_token_index(
            START_SYMBOL, "decoder_token_ids")
        pos_tags = torch.ones(batch_size, 1) * self.vocab.get_token_index(
            DEFAULT_OOV_TOKEN, "pos_tags")
        tokens = tokens.type_as(mask).long()
        pos_tags = pos_tags.type_as(tokens)
        corefs = torch.zeros(batch_size, 1).type_as(mask).long()
        # Output
        rnn_outputs = []
        copy_attentions = []
        coref_attentions = []
        predictions = []
        coref_indexes = []
        decoder_mask = []
        edge_heads = []
        edge_labels = []
        # Internal input
        input_feed = memory_bank.new_zeros(batch_size, 1, self.decoder.rnn_cell.hidden_size)
        coref_inputs = []
        # A sparse indicator matrix mapping each node to its index in the dynamic vocab.
        # Here the maximum size of the dynamic vocab is just max_decode_length.
        coref_attention_maps = torch.zeros(
            batch_size,
            self.max_decode_length,
            self.max_decode_length + 1).type_as(memory_bank)
        # A matrix D where the element D_{ij} is for instance i the real vocab index of
        # the generated node at the decoding step `i'.
        coref_vocab_maps = torch.zeros(
            batch_size,
            self.max_decode_length + 1).type_as(mask).long()
        coverage = None
        if self.use_coverage:
            coverage = memory_bank.new_zeros(batch_size, 1, memory_bank.size(1))

        for step_i in range(self.max_decode_length):
            # 1. Get the decoder inputs.
            token_embeddings = self.decoder_token_embedding(tokens)
            pos_tag_embeddings = self.decoder_pos_embedding(pos_tags)
            coref_embeddings = self.decoder_coref_embedding(corefs)
            if self.use_char_cnn:
                # TODO: get chars from tokens.
                # [batch_size, 1, num_chars]
                chars = character_tensor_from_token_tensor(
                    tokens,
                    self.vocab,
                    self.character_tokenizer
                )

                char_cnn_output = self._get_decoder_char_cnn_output(chars)
                decoder_inputs = torch.cat(
                    [token_embeddings, pos_tag_embeddings,
                     coref_embeddings, char_cnn_output], 2)
            else:
                decoder_inputs = torch.cat(
                    [token_embeddings, pos_tag_embeddings, coref_embeddings], 2)

            decoder_inputs = self.decoder_embedding_dropout(decoder_inputs)

            # 2. Run tree decoder.
            _rnn_outputs, _ = self.decoder.one_step_rnn_forward(
                decoder_inputs, states, input_feed)

            if step_i != 0:
                queries = _rnn_outputs
                if step_i == 1:
                    keys, edge_mask = None, None
                else:
                    keys = torch.cat(rnn_outputs[1:], dim=1)
                    coref_history = torch.cat(coref_indexes[:-1], dim=1)
                    coref_current = coref_indexes[-1]
                    edge_mask = coref_history.ne(coref_current).unsqueeze(1)
                heads, labels = self.tree_decoder(queries, keys, edge_mask)
                edge_heads += [heads]
                edge_labels += [labels]

            # 3. Decode one step.
            decoder_output_dict = self.decoder.one_step_forward(
                decoder_inputs, memory_bank, mask, states, input_feed,
                None, None, coref_inputs, coverage, step_i, 1)
            _decoder_outputs = decoder_output_dict['decoder_output']
            _rnn_outputs = decoder_output_dict['rnn_output']
            _copy_attentions = decoder_output_dict['source_copy_attention']
            _coref_attentions = decoder_output_dict['target_copy_attention']
            states = decoder_output_dict['rnn_hidden_state']
            input_feed = decoder_output_dict['input_feed']
            coverage = decoder_output_dict['coverage']

            # 4. Run pointer/generator.
            if step_i == 0:
                _coref_attention_maps = coref_attention_maps[:, :step_i + 1]
            else:
                _coref_attention_maps = coref_attention_maps[:, :step_i]

            generator_output = self.generator(
                _decoder_outputs, _copy_attentions, copy_attention_maps,
                _coref_attentions, _coref_attention_maps, invalid_indexes)
            _predictions = generator_output['predictions']

            # 5. Update maps and get the next token input.
            tokens, _predictions, pos_tags, corefs, _mask = self._update_maps_and_get_next_input(
                step_i,
                generator_output['predictions'].squeeze(1),
                generator_output['source_dynamic_vocab_size'],
                coref_attention_maps,
                coref_vocab_maps,
                copy_vocabs,
                decoder_mask,
                tag_luts,
                invalid_indexes
            )

            # 6. Update variables.
            rnn_outputs += [_rnn_outputs]
            coref_inputs += [_decoder_outputs]

            copy_attentions += [_copy_attentions]
            coref_attentions += [_coref_attentions]

            predictions += [_predictions]
            # Add the coref info for the next input.
            coref_indexes += [corefs]
            # Add the mask for the next input.
            decoder_mask += [_mask]

        # 7. Do the following chunking for the tree decoding input.
        # Exclude coref/mask for EOS.
        # TODO: Answer "What if the last one is not EOS?"
        predictions = torch.cat(predictions[:-1], dim=1)
        coref_indexes = torch.cat(coref_indexes[:-1], dim=1)
        decoder_mask = 1 - torch.cat(decoder_mask[:-1], dim=1)
        edge_heads = torch.cat(edge_heads, dim=1)
        edge_labels = torch.cat(edge_labels, dim=1)

        return dict(
            # [batch_size, max_decode_length]
            predictions=predictions,
            coref_indexes=coref_indexes,
            decoder_mask=decoder_mask,
            edge_heads=edge_heads,
            edge_labels=edge_labels,
            # [batch_size, max_decode_length, encoder_length]
            copy_attentions=copy_attentions,
            coref_attentions=coref_attentions
        )

    def _update_maps_and_get_next_input(
            self, step, predictions, copy_vocab_size, coref_attention_maps, coref_vocab_maps,
            copy_vocabs, masks, tag_luts, invalid_indexes):
        """Dynamically update/build the maps needed for copying.

        :param step: the decoding step, int.
        :param predictions: [batch_size]
        :param copy_vocab_size: int.
        :param coref_attention_maps: [batch_size, max_decode_length, max_decode_length]
        :param coref_vocab_maps:  [batch_size, max_decode_length]
        :param copy_vocabs: a list of dynamic vocabs.
        :param masks: a list of [batch_size] tensors indicating whether EOS has been generated.
            if EOS has has been generated, then the mask is `1`.
        :param tag_luts: a dict mapping key to a list of dicts mapping a source token to a POS tag.
        :param invalid_indexes: a dict storing invalid indexes for copying and generation.
        :return:
        """
        vocab_size = self.generator.vocab_size
        batch_size = predictions.size(0)

        batch_index = torch.arange(0, batch_size).type_as(predictions)
        step_index = torch.full_like(predictions, step)

        gen_mask = predictions.lt(vocab_size)
        copy_mask = predictions.ge(vocab_size).mul(predictions.lt(vocab_size + copy_vocab_size))
        coref_mask = predictions.ge(vocab_size + copy_vocab_size)

        # 1. Update coref_attention_maps
        # Get the coref index.
        coref_index = (predictions - vocab_size - copy_vocab_size)
        # Fill the place where copy didn't happen with the current step,
        # which means that the node doesn't refer to any precedent, it refers to itself.
        coref_index.masked_fill_(1 - coref_mask, step + 1)

        coref_attention_maps[batch_index, step_index, coref_index] = 1

        # 2. Compute the next input.
        # coref_predictions have the dynamic vocabulary index, and OOVs are set to zero.
        coref_predictions = (predictions - vocab_size - copy_vocab_size) * coref_mask.long()
        # Get the actual coreferred token's index in the gen vocab.
        coref_predictions = coref_vocab_maps.gather(1, coref_predictions.unsqueeze(1)).squeeze(1)

        # If a token is copied from the source side, we look up its index in the gen vocab.
        copy_predictions = (predictions - vocab_size) * copy_mask.long()
        pos_tags = torch.full_like(predictions, self.vocab.get_token_index(DEFAULT_OOV_TOKEN, 'pos_tags'))
        for i, index in enumerate(copy_predictions.tolist()):
            copied_token = copy_vocabs[i].get_token_from_idx(index)
            if index != 0:
                pos_tags[i] = self.vocab.get_token_index(
                    tag_luts[i]['pos'][copied_token], 'pos_tags')
                if False:  # is_abstract_token(copied_token):
                    invalid_indexes['source_copy'][i].add(index)
            copy_predictions[i] = self.vocab.get_token_index(copied_token, 'decoder_token_ids')

        for i, index in enumerate(
                (predictions * gen_mask.long() + coref_predictions * coref_mask.long()).tolist()):
            if index != 0:
                token = self.vocab.get_token_from_index(index, 'decoder_token_ids')
                src_token = find_similar_token(token, list(tag_luts[i]['pos'].keys()))
                if src_token is not None:
                    pos_tags[i] = self.vocab.get_token_index(
                        tag_luts[i]['pos'][src_token], 'pos_tag')
                if False:  # is_abstract_token(token):
                    invalid_indexes['vocab'][i].add(index)

        next_input = coref_predictions * coref_mask.long() + \
                     copy_predictions * copy_mask.long() + \
                     predictions * gen_mask.long()

        # 3. Update dynamic_vocab_maps
        # Here we update D_{step} to the index in the standard vocab.
        coref_vocab_maps[batch_index, step_index + 1] = next_input

        # 4. Get the coref-resolved predictions.
        coref_resolved_preds = coref_predictions * coref_mask.long() + predictions * (1 - coref_mask).long()

        # 5. Get the mask for the current generation.
        has_eos = torch.zeros_like(gen_mask)
        if len(masks) != 0:
            has_eos = torch.cat(masks, 1).long().sum(1).gt(0)
        mask = next_input.eq(self.vocab.get_token_index(END_SYMBOL, 'decoder_token_ids')) | has_eos

        return (next_input.unsqueeze(1),
                coref_resolved_preds.unsqueeze(1),
                pos_tags.unsqueeze(1),
                coref_index.unsqueeze(1),
                mask.unsqueeze(1))

    @classmethod
    def from_params(cls, vocab, params):
        logger.info('Building the ISTOG Model...')

        # Encoder
        encoder_input_size = 0
        bert_encoder = None
        if params.get('use_bert', False):
            bert_encoder = Seq2SeqBertEncoder.from_pretrained(params['bert']['pretrained_model_dir'])
            encoder_input_size += params['bert']['hidden_size']
            for p in bert_encoder.parameters():
                p.requires_grad = False

        encoder_token_embedding = Embedding.from_params(vocab, params['encoder_token_embedding'])
        encoder_input_size += params['encoder_token_embedding']['embedding_dim']
        encoder_pos_embedding = Embedding.from_params(vocab, params['encoder_pos_embedding'])
        encoder_input_size += params['encoder_pos_embedding']['embedding_dim']

        encoder_anonym_indicator_embedding = None
        if params.get('use_anonym_indicator', False):
            encoder_anonym_indicator_embedding = Embedding.from_params(
                vocab, params['encoder_anonym_indicator_embedding'])
            encoder_input_size += params['encoder_anonym_indicator_embedding']['embedding_dim']

        if params['use_char_cnn']:
            encoder_char_embedding = Embedding.from_params(vocab, params['encoder_char_embedding'])
            encoder_char_cnn = CnnEncoder(
                embedding_dim=params['encoder_char_cnn']['embedding_dim'],
                num_filters=params['encoder_char_cnn']['num_filters'],
                ngram_filter_sizes=params['encoder_char_cnn']['ngram_filter_sizes'],
                conv_layer_activation=torch.tanh
            )
            encoder_input_size += params['encoder_char_cnn']['num_filters']
        else:
            encoder_char_embedding = None
            encoder_char_cnn = None

        encoder_embedding_dropout = InputVariationalDropout(p=params['encoder_token_embedding']['dropout'])

        params['encoder']['input_size'] = encoder_input_size
        encoder = PytorchSeq2SeqWrapper(
            module=StackedBidirectionalLstm.from_params(params['encoder']),
            stateful=True
        )
        encoder_output_dropout = InputVariationalDropout(p=params['encoder']['dropout'])

        # Decoder
        decoder_input_size = params['decoder_token_embedding']['embedding_dim']
        decoder_input_size += params['decoder_coref_embedding']['embedding_dim']
        decoder_input_size += params['decoder_pos_embedding']['embedding_dim']
        decoder_token_embedding = Embedding.from_params(vocab, params['decoder_token_embedding'])
        decoder_coref_embedding = Embedding.from_params(vocab, params['decoder_coref_embedding'])
        decoder_pos_embedding = Embedding.from_params(vocab, params['decoder_pos_embedding'])
        if params['use_char_cnn']:
            decoder_char_embedding = Embedding.from_params(vocab, params['decoder_char_embedding'])
            decoder_char_cnn = CnnEncoder(
                embedding_dim=params['decoder_char_cnn']['embedding_dim'],
                num_filters=params['decoder_char_cnn']['num_filters'],
                ngram_filter_sizes=params['decoder_char_cnn']['ngram_filter_sizes'],
                conv_layer_activation=torch.tanh
            )
            decoder_input_size += params['decoder_char_cnn']['num_filters']
        else:
            decoder_char_embedding = None
            decoder_char_cnn = None

        decoder_embedding_dropout = InputVariationalDropout(p=params['decoder_token_embedding']['dropout'])

        # Source attention
        if params['source_attention']['attention_function'] == 'mlp':
            source_attention = MLPAttention(
                decoder_hidden_size=params['decoder']['hidden_size'],
                encoder_hidden_size=params['encoder']['hidden_size'] * 2,
                attention_hidden_size=params['decoder']['hidden_size'],
                coverage=params['source_attention'].get('coverage', False)
            )
        else:
            source_attention = DotProductAttention(
                decoder_hidden_size=params['decoder']['hidden_size'],
                encoder_hidden_size=params['encoder']['hidden_size'] * 2,
                share_linear=params['source_attention'].get('share_linear', False)
            )

        source_attention_layer = GlobalAttention(
            decoder_hidden_size=params['decoder']['hidden_size'],
            encoder_hidden_size=params['encoder']['hidden_size'] * 2,
            attention=source_attention
        )

        # Coref attention
        if params['coref_attention']['attention_function'] == 'mlp':
            coref_attention = MLPAttention(
                decoder_hidden_size=params['decoder']['hidden_size'],
                encoder_hidden_size=params['decoder']['hidden_size'],
                attention_hidden_size=params['decoder']['hidden_size'],
                coverage=params['coref_attention'].get('coverage', False),
                use_concat=params['coref_attention'].get('use_concat', False)
            )
        elif params['coref_attention']['attention_function'] == 'biaffine':
            coref_attention = BiaffineAttention(
                input_size_decoder=params['decoder']['hidden_size'],
                input_size_encoder=params['encoder']['hidden_size'] * 2,
                hidden_size=params['coref_attention']['hidden_size']
            )
        else:
            coref_attention = DotProductAttention(
                decoder_hidden_size=params['decoder']['hidden_size'],
                encoder_hidden_size=params['decoder']['hidden_size'],
                share_linear=params['coref_attention'].get('share_linear', True)
            )

        coref_attention_layer = GlobalAttention(
            decoder_hidden_size=params['decoder']['hidden_size'],
            encoder_hidden_size=params['decoder']['hidden_size'],
            attention=coref_attention
        )

        decoder_input_size += params['decoder']['hidden_size']
        params['decoder']['input_size'] = decoder_input_size
        decoder = InputFeedRNNDecoder(
            rnn_cell=StackedLstm.from_params(params['decoder']),
            attention_layer=source_attention_layer,
            coref_attention_layer=coref_attention_layer,
            # TODO: modify the dropout so that the dropout mask is unchanged across the steps.
            dropout=InputVariationalDropout(p=params['decoder']['dropout']),
            use_coverage=params['use_coverage']
        )

        if params.get('use_aux_encoder', False):
            aux_encoder = PytorchSeq2SeqWrapper(
                module=StackedBidirectionalLstm.from_params(params['aux_encoder']),
                stateful=True
            )
            aux_encoder_output_dropout = InputVariationalDropout(
                p=params['aux_encoder']['dropout'])
        else:
            aux_encoder = None
            aux_encoder_output_dropout = None

        switch_input_size = params['encoder']['hidden_size'] * 2
        generator = PointerGenerator(
            input_size=params['decoder']['hidden_size'],
            switch_input_size=switch_input_size,
            vocab_size=vocab.get_vocab_size('decoder_token_ids'),
            force_copy=params['generator'].get('force_copy', True),
            # TODO: Set the following indices.
            vocab_pad_idx=0
        )

        tree_decoder = DeepBiaffineTreeDecoder.from_params(vocab, params['tree_decoder'])

        # Vocab
        punctuation_ids = []
        oov_id = vocab.get_token_index(DEFAULT_OOV_TOKEN, 'decoder_token_ids')
        for c in ',.?!:;"\'-(){}[]':
            c_id = vocab.get_token_index(c, 'decoder_token_ids')
            if c_id != oov_id:
                punctuation_ids.append(c_id)

        logger.info('encoder_token: %d' % vocab.get_vocab_size('encoder_token_ids'))
        logger.info('encoder_chars: %d' % vocab.get_vocab_size('encoder_token_characters'))
        logger.info('decoder_token: %d' % vocab.get_vocab_size('decoder_token_ids'))
        logger.info('decoder_chars: %d' % vocab.get_vocab_size('decoder_token_characters'))

        return cls(
            vocab=vocab,
            punctuation_ids=punctuation_ids,
            use_anonym_indicator=params.get('use_anonym_indicator', False),
            use_char_cnn=params['use_char_cnn'],
            use_coverage=params['use_coverage'],
            use_aux_encoder=params.get('use_aux_encoder', False),
            use_bert=params.get('use_bert', False),
            max_decode_length=params.get('max_decode_length', 50),
            bert_encoder=bert_encoder,
            encoder_token_embedding=encoder_token_embedding,
            encoder_pos_embedding=encoder_pos_embedding,
            encoder_anonym_indicator_embedding=encoder_anonym_indicator_embedding,
            encoder_char_embedding=encoder_char_embedding,
            encoder_char_cnn=encoder_char_cnn,
            encoder_embedding_dropout=encoder_embedding_dropout,
            encoder=encoder,
            encoder_output_dropout=encoder_output_dropout,
            decoder_token_embedding=decoder_token_embedding,
            decoder_coref_embedding=decoder_coref_embedding,
            decoder_pos_embedding=decoder_pos_embedding,
            decoder_char_cnn=decoder_char_cnn,
            decoder_char_embedding=decoder_char_embedding,
            decoder_embedding_dropout=decoder_embedding_dropout,
            decoder=decoder,
            aux_encoder=aux_encoder,
            aux_encoder_output_dropout=aux_encoder_output_dropout,
            generator=generator,
            tree_decoder=tree_decoder,
            test_config=params.get('mimick_test', None)
        )
