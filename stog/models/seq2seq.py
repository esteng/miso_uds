import torch

from stog.models.model import Model
from stog.utils.logging import init_logger
from stog.modules.token_embedders.embedding import Embedding
from stog.modules.seq2vec_encoders.cnn_encoder import CnnEncoder
from stog.modules.seq2seq_encoders.pytorch_seq2seq_wrapper import PytorchSeq2SeqWrapper
from stog.modules.stacked_bilstm import StackedBidirectionalLstm
from stog.modules.stacked_lstm import StackedLstm
from stog.modules.decoders.rnn_decoder import InputFeedRNNDecoder
from stog.modules.attention_layers.global_attention import GlobalAttention
from stog.modules.attention.dot_production_attention import DotProductAttention
from stog.modules.attention_layers.self_copy_attention import SelfCopyAttention
from stog.modules.input_variational_dropout import InputVariationalDropout
from stog.modules.decoders.generator import Generator
from stog.modules.decoders.copy_generator import CopyGenerator
from stog.utils.nn import get_text_field_mask
from stog.utils.string import START_SYMBOL
from stog.data.tokenizers.character_tokenizer import CharacterTokenizer

logger = init_logger()

def character_tensor_from_token_tensor(token_tensor,
                                      vocab,
                                      character_tokenizer,
                                      namespace={
                                          "tokens" : "decoder_token_ids",
                                          "characters" : "decoder_token_characters"
                                      }):
        #import pdb;pdb.set_trace()
        token_str = [vocab.get_token_from_index(i, namespace["tokens"]) for i in token_tensor.view(-1).tolist()]
        max_char_len = max([len(token) for token in token_str])
        indices = []
        for token in token_str:
            token_indices = [vocab.get_token_index(vocab._padding_token) for _ in range(max_char_len)]
            for char_i, character in enumerate(character_tokenizer.tokenize(token)):
                index = vocab.get_token_index(character.text, namespace["characters"])
                token_indices[char_i] = index
            indices.append(token_indices)

        return torch.tensor(indices).view(token_tensor.size(0), 1, -1).type_as(token_tensor)
    


class Seq2Seq(Model):

    def __init__(self,
                 vocab,
                 use_char_cnn,
                 use_self_copy,
                 max_decode_length,
                 # Encoder
                 encoder_token_embedding,
                 encoder_char_embedding,
                 encoder_char_cnn,
                 encoder_embedding_dropout,
                 encoder,
                 encoder_output_dropout,
                 # Decoder
                 decoder_token_embedding,
                 decoder_char_embedding,
                 decoder_char_cnn,
                 decoder_embedding_dropout,
                 decoder,
                 # Self-copy Mechanism
                 self_copy_attention,
                 # Generator
                 generator):
        super(Seq2Seq, self).__init__()

        self.vocab = vocab
        self.use_char_cnn = use_char_cnn
        self.use_self_copy = use_self_copy
        self.max_decode_length = max_decode_length

        self.encoder_token_embedding = encoder_token_embedding
        self.encoder_char_embedding = encoder_char_embedding
        self.encoder_char_cnn = encoder_char_cnn
        self.encoder_embedding_dropout = encoder_embedding_dropout
        self.encoder = encoder
        self.encoder_output_dropout = encoder_output_dropout

        self.decoder_token_embedding = decoder_token_embedding
        self.decoder_char_embedding = decoder_char_embedding
        self.decoder_char_cnn = decoder_char_cnn
        self.decoder_embedding_dropout = decoder_embedding_dropout
        self.decoder = decoder

        self.self_copy_attention = self_copy_attention

        self.generator = generator

        self.beam_size = 1

    def set_beam_size(self, beam_size):
        self.beam_size = beam_size

    def set_decoder_token_indexers(self, token_indexers):
        self.decoder_token_indexers = token_indexers
        self.character_tokenizer = CharacterTokenizer()

    def get_metrics(self, reset: bool = False):
        return self.generator.metrics.get_metric(reset)

    def forward(self, batch, for_training=False):
        # TODO: Xutai
        # [batch, num_tokens]
        encoder_token_inputs = batch['src_tokens']['encoder_tokens']
        # [batch, num_tokens, num_chars]
        encoder_char_inputs = batch['src_tokens']['encoder_characters']
        # [batch, num_tokens]
        encoder_mask = get_text_field_mask(batch['src_tokens'])

        try:
            has_decoder_inputs = True
            # [batch, num_tokens]
            decoder_token_inputs = batch['amr_tokens']['decoder_tokens'][:, :-1].contiguous()
            # [batch, num_tokens, num_chars]
            decoder_char_inputs = batch['amr_tokens']['decoder_characters'][:, :-1].contiguous()
            # [batch, num_tokens]
            targets = batch['amr_tokens']['decoder_tokens'][:, 1:].contiguous()

            vocab_targets = None
            copy_targets = None
            copy_attention_maps = None

            # import pdb;pdb.set_trace()
        except:
            has_decoder_inputs = False

        encoder_memory_bank, encoder_final_states = self.encode(
            encoder_token_inputs,
            encoder_char_inputs,
            encoder_mask
        )

        if for_training and has_decoder_inputs:
            decoder_memory_bank, source_attentions, decoder_final_states = self.decode_for_training(
                decoder_token_inputs,
                decoder_char_inputs,
                encoder_memory_bank,
                encoder_mask,
                encoder_final_states
            )

            attentions = dict(
                source=source_attentions,
                copy=[]
            )

            if self.use_self_copy:
                copy_attentions = self.self_copy_attention(decoder_memory_bank, decoder_memory_bank)
                _generator_output = self.generator(decoder_memory_bank, copy_attentions, copy_attention_maps)
                generator_output = self.generator.compute_loss(
                    _generator_output['loss'],
                    _generator_output['predictions'],
                    vocab_targets,
                    copy_targets
                )
                attentions['copy'] = copy_attentions

            else:
                generator_output = self.generator.compute_loss(decoder_memory_bank, targets)

            return dict(
                loss=generator_output['loss'],
                predictions=generator_output['predictions'],
                attentions=attentions,
                decoder_final_states=decoder_final_states
            )
        else:
            return dict(
                encoder_memory_bank=encoder_memory_bank,
                encoder_mask=encoder_mask,
                encoder_final_states=encoder_final_states,
            )

    def encode(self, tokens, chars, mask):
        # [batch, num_tokens, embedding_size]
        token_embeddings = self.encoder_token_embedding(tokens)
        if self.use_char_cnn:
            char_cnn_output = self._get_encoder_char_cnn_output(chars)
            encoder_inputs = torch.cat([token_embeddings, char_cnn_output], 2)
        else:
            encoder_inputs = token_embeddings

        encoder_inputs = self.encoder_embedding_dropout(encoder_inputs)

        # [batch, num_tokens, encoder_output_size]
        encoder_outputs = self.encoder(encoder_inputs, mask)
        encoder_outputs = self.encoder_output_dropout(encoder_outputs)

        # A tuple of (state, memory) with shape [num_layers, batch, encoder_output_size]
        encoder_final_states = self.encoder._states
        self.encoder.reset_states()

        return encoder_outputs, encoder_final_states

    def decode_for_training(self, tokens, chars, memory_bank, mask, states):
        # [batch, num_tokens, embedding_size]
        token_embeddings = self.decoder_token_embedding(tokens)
        if self.use_char_cnn:
            char_cnn_output = self._get_decoder_char_cnn_output(chars)
            decoder_inputs = torch.cat([token_embeddings, char_cnn_output], 2)
        else:
            decoder_inputs = token_embeddings

        decoder_inputs = self.decoder_embedding_dropout(decoder_inputs)

        decoder_outputs, attentions, decoder_final_states, _ = \
            self.decoder(decoder_inputs, memory_bank, mask, states)
        return decoder_outputs, attentions, decoder_final_states

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

        if self.beam_size == 1:
            return self.greedy_decode(memory_bank, mask, states)
        else:
            raise NotImplementedError

    def greedy_decode(self, memory_bank, mask, states):
        # TODO: convert START_SYMBOL to a batch of 'START_SYMBOL' indices.
        # [batch_size, 1]
        batch_size = memory_bank.size(0)
        tokens = torch.ones(batch_size, 1) \
                * self.vocab.get_token_index(START_SYMBOL, "decoder_token_ids")
        tokens = tokens.type_as(mask)

        input_feed = None
        source_attentions = []
        copy_attentions = []
        decoder_outputs = []
        predictions = []
        for step_i in range(self.max_decode_length):
            # Get embeddings.
            token_embeddings = self.decoder_token_embedding(tokens)
            if self.use_char_cnn:
                # TODO: get chars from tokens.
                # [batch_size, 1, num_chars]
                chars = character_tensor_from_token_tensor(
                    tokens,
                    self.vocab,
                    self.character_tokenizer
                )
                # [batch, num_tokens, embedding_size]
                char_embeddings = self.decoder_char_embedding(chars)
                batch_size, num_tokens, num_chars, _ = char_embeddings.size()
                char_embeddings = char_embeddings.view(batch_size * num_tokens, num_chars, -1)
                char_cnn_output = self.decoder_char_cnn(char_embeddings, None)
                char_cnn_output = char_cnn_output.view(batch_size, num_tokens, -1)

                char_cnn_output = self._get_decoder_char_cnn_output(chars)
                decoder_inputs = torch.cat([token_embeddings, char_cnn_output], 2)
            else:
                decoder_inputs = token_embeddings
            decoder_inputs = self.decoder_embedding_dropout(decoder_inputs)

            # Decode one step.
            _decoder_outputs, _source_attentions, states, input_feed = \
                self.decoder(decoder_inputs, memory_bank, mask, states, input_feed)

            generator_output = self.generator(_decoder_outputs)

            # Update decoder outputs.
            source_attentions += _source_attentions
            decoder_outputs += _decoder_outputs

            # Generate.
            _predictions = generator_output['predictions']
            predictions.append(_predictions)

            tokens = _predictions

        return dict(
            # [batch_size, max_decode_length]
            predictions=torch.cat(predictions, dim=1),
            # [batch_size, max_decode_length, encoder_length]
            std_attentions=torch.cat(source_attentions, dim=1) if len(std_attentions) != 0 else None,
            copy_attentions=torch.cat(copy_attentions, dim=1) if len(copy_attentions) != 0 else None
        )

    def greedy_decode_with_copy(self, memory_bank, mask, states):
        # TODO: convert START_SYMBOL to a batch of 'START_SYMBOL' indices.
        # [batch_size, 1]
        batch_size = memory_bank.size(0)
        tokens = None

        input_feed = None
        source_attentions = []
        copy_attentions = []
        decoder_outputs = []
        predictions = []

        # A sparse indicator matrix mapping each node to its index in the dynamic vocab.
        # Here the maximum size of the dynamic vocab is just max_decode_length.
        attention_maps = torch.zeros(batch_size, self.max_decode_length, self.max_decode_length)
        # A diagonal matrix D where the element D_{i,i} is the real vocab index that the index `i'
        # in the dynamic vocab should be mapped to.
        # With D and the index `i' represented by an one-hot vector h, the real vocab index can
        # be computed by the matrix product h * D.
        dynamic_vocab_maps = torch.zeros(batch_size, self.max_decode_length, self.max_decode_length).long()

        for step_i in range(self.max_decode_length):
            # Get embeddings.
            token_embeddings = self.decoder_token_embedding(tokens)
            if self.use_char_cnn:
                # TODO: get chars from tokens.
                # [batch_size, 1, num_chars]
                chars = None

                char_cnn_output = self._get_decoder_char_cnn_output(chars)
                decoder_inputs = torch.cat([token_embeddings, char_cnn_output], 2)
            else:
                decoder_inputs = token_embeddings
            decoder_inputs = self.decoder_embedding_dropout(decoder_inputs)

            # Decode one step.
            _decoder_outputs, _source_attentions, states, input_feed = \
                self.decoder(decoder_inputs, memory_bank, mask, states, input_feed)

            if step_i == 0:
                # Dummy copy attention for the first step will never be chosen.
                _copy_attentions = _decoder_outputs.new_ones(batch_size, 1, 1)
                _attention_maps = _decoder_outputs.new_zeros(batch_size, 1, 1)
            else:
                decoder_outputs_by_far = torch.cat(decoder_outputs, dim=1)
                _copy_attentions = self.self_copy_attention(_decoder_outputs, decoder_outputs_by_far)
                _attention_maps = attention_maps[:, :step_i]

            # Generate.
            generator_output = self.generator(_decoder_outputs, _copy_attentions, _attention_maps)
            _predictions = generator_output['predictions']

            # Update decoder outputs.
            copy_attentions += _copy_attentions
            source_attentions += _source_attentions
            decoder_outputs += _decoder_outputs

            tokens = self._update_maps_and_get_next_input(
                step_i, _predictions.squeeze(1), attention_maps, dynamic_vocab_maps)

            predictions += tokens.unsqueeze(1)

        return dict(
            # [batch_size, max_decode_length]
            predictions=torch.cat(predictions, dim=1),
            # [batch_size, max_decode_length, encoder_length]
            std_attentions=torch.cat(std_attentions, dim=1) if len(std_attentions) != 0 else None,
            #copy_attentions=torch.cat(copy_attentions, dim=1) if len(copy_attentions) != 0 else None
        )

    def _update_maps_and_get_next_input(self, step, predictions, attention_maps, dynamic_vocab_maps):
        """Dynamically update/build the maps needed for copying.

        :param step: the decoding step, int.
        :param predictions: [batch_size]
        :param attention_maps: [batch_size, max_decode_length, max_decode_length]
        :param dynamic_vocab_maps:  [batch_size, max_decode_length, max_decode_length]
        :return:
        """
        vocab_size = self.generator.vocab_size
        batch_size = predictions.size(0)

        batch_index = torch.arange(0, batch_size).long()
        step_index = torch.tensor([step] * batch_size).long()
        vocab_oov_mask = predictions.ge(vocab_size)

        # 1. Update attention_maps
        dynamic_index = (predictions - vocab_size)
        # OOVs of the dynamic vocabulary are filled with `step',
        # which means a new index `step' is added to the dynamic vocab
        # for those generated nodes.
        # dynamic_index means where the nodes at this step should be mapped to.
        dynamic_index.masked_fill_(1 - vocab_oov_mask, step)

        attention_maps[batch_index, step_index, dynamic_index] = 1

        # 2. Update dynamic_vocab_maps
        # vocab_predictions have the standard vocabulary index, and OOVs are set to zero.
        vocab_predictions = predictions * (1 - vocab_oov_mask)

        # If the index in vocab_predictions is not zero, it means a new node has been generated.
        # The index of this new node in the dynamic vocab is `step'.
        # So the map is `step' -> the index in the standard vocab.
        # Here we update D_{step, step} to the index in the standard vocab.
        dynamic_vocab_maps[batch_index, step_index, step_index] = vocab_predictions

        # 3. Compute the next input.
        # copy_predictions have the dynamic vocabulary index, and OOVs are set to zero.
        copy_predictions = (predictions - vocab_size) * vocab_oov_mask
        # Convert the dynamic vocab index to one-hot vector.
        copy_prediction_one_hots = torch.zeros(batch_size, self.max_decode_length)
        copy_prediction_one_hots.scatter_(1, copy_predictions, 1)
        # [batch_size, 1, max_decode_length]
        copy_prediction_one_hots = copy_prediction_one_hots.unsqueeze(1)
        # Convert the dynamic vocab index to the standard vocab index.
        # [batch_size]: the next input index in the standard vocab for copied predictions.
        next_input = torch.bmm(copy_prediction_one_hots, dynamic_vocab_maps).sum(dim=(1, 2))
        # Merge it with generated predictions.
        next_input = next_input * vocab_oov_mask + vocab_predictions

        return next_input

    @classmethod
    def from_params(cls, vocab, params):
        logger.info('Building the Seq2Seq Model...')

        # Encoder
        encoder_token_embedding = Embedding.from_params(vocab, params['encoder_token_embedding'])
        if params['use_char_cnn']:
            encoder_char_embedding = Embedding.from_params(vocab, params['encoder_char_embedding'])
            encoder_char_cnn = CnnEncoder(
                embedding_dim=params['encoder_char_cnn']['embedding_dim'],
                num_filters=params['encoder_char_cnn']['num_filters'],
                ngram_filter_sizes=params['encoder_char_cnn']['ngram_filter_sizes'],
                conv_layer_activation=torch.tanh
            )
        else:
            encoder_char_embedding = None
            encoder_char_cnn = None

        encoder_embedding_dropout = InputVariationalDropout(p=params['encoder_token_embedding']['dropout'])

        encoder = PytorchSeq2SeqWrapper(
            module=StackedBidirectionalLstm.from_params(params['encoder']),
            stateful=True
        )
        encoder_output_dropout = InputVariationalDropout(p=params['encoder']['dropout'])

        # Decoder
        decoder_token_embedding = Embedding.from_params(vocab, params['decoder_token_embedding'])
        if params['use_char_cnn']:
            decoder_char_embedding = Embedding.from_params(vocab, params['decoder_char_embedding'])
            decoder_char_cnn = CnnEncoder(
                embedding_dim=params['decoder_char_cnn']['embedding_dim'],
                num_filters=params['decoder_char_cnn']['num_filters'],
                ngram_filter_sizes=params['decoder_char_cnn']['ngram_filter_sizes'],
                conv_layer_activation=torch.tanh
            )
        else:
            decoder_char_embedding = None
            decoder_char_cnn = None

        decoder_embedding_dropout = InputVariationalDropout(p=params['decoder_token_embedding']['dropout'])

        attention = DotProductAttention(
            decoder_hidden_size=params['decoder']['hidden_size'],
            encoder_hidden_size=params['encoder']['hidden_size'] * 2,
            add_linear=params['attention'].get('add_linear', True)
        )
        attention_layer = GlobalAttention(
            decoder_hidden_size=params['decoder']['hidden_size'],
            encoder_hidden_size=params['encoder']['hidden_size'] * 2,
            attention=attention
        )
        decoder = InputFeedRNNDecoder(
            rnn_cell=StackedLstm.from_params(params['decoder']),
            attention_layer=attention_layer,
            # TODO: modify the dropout so that the dropout mask is unchanged across the steps.
            dropout=InputVariationalDropout(p=params['decoder']['dropout'])
        )

        if params['use_self_copy']:
            attention_module = DotProductAttention(
                decoder_hidden_size=params['decoder']['hidden_size'],
                encoder_hidden_size=params['decoder']['hidden_size'],
                add_linear=params['self_copy_attention'].get('add_linear', True)
            )
            self_copy_attention = SelfCopyAttention(
                attention=attention_module
            )
            generator = CopyGenerator(
                input_size=params['decoder']['hidden_size'],
                vocab_size=vocab.get_vocab_size('decoder_token_ids'),
                force_copy=params['self_copy'].get('force_copy', True),
                # TODO: Set the following indices.
                vocab_pad_idx=0,
                vocab_oov_idx=0,
                copy_oov_idx=0
            )

        else:
            self_copy_attention = None
            # Generator
            params['generator']['vocab_size'] = vocab.get_vocab_size('decoder_token_ids')
            params['generator']['pad_idx'] = 0
            generator = Generator.from_params(params['generator'])

        logger.info('encoder_token: %d' %vocab.get_vocab_size('encoder_token_ids'))
        logger.info('encoder_chars: %d' %vocab.get_vocab_size('encoder_token_characters'))
        logger.info('decoder_token: %d' %vocab.get_vocab_size('decoder_token_ids'))
        logger.info('decoder_chars: %d' %vocab.get_vocab_size('decoder_token_characters'))

        return cls(
            vocab=vocab,
            use_char_cnn=params['use_char_cnn'],
            use_self_copy=params['use_self_copy'],
            max_decode_length=params.get('max_decode_length', 50),
            encoder_token_embedding=encoder_token_embedding,
            encoder_char_embedding=encoder_char_embedding,
            encoder_char_cnn=encoder_char_cnn,
            encoder_embedding_dropout=encoder_embedding_dropout,
            encoder=encoder,
            encoder_output_dropout=encoder_output_dropout,
            decoder_token_embedding=decoder_token_embedding,
            decoder_char_cnn=decoder_char_cnn,
            decoder_char_embedding=decoder_char_embedding,
            decoder_embedding_dropout=decoder_embedding_dropout,
            decoder=decoder,
            self_copy_attention=self_copy_attention,
            generator=generator
        )
