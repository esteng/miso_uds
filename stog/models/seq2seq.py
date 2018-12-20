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

    def print_batch_details(self, batch, batch_idx):
        print(batch["amr"][batch_idx])
        print()

        print("Source tokens:")
        print([(i, x) for i, x in enumerate(batch["src_tokens_str"][batch_idx])])
        print()

        print('Source copy vocab')
        print(batch["src_copy_vocab"][batch_idx])
        print()

        print('Source map')
        print(batch["src_copy_map"][batch_idx].int())
        print()

        print("Target tokens")
        print([(i, x) for i, x in enumerate(batch["tgt_tokens_str"][batch_idx])])
        print()

        print('Source copy indices')
        print([(i, x) for i, x in enumerate(batch["src_copy_indices"][batch_idx].tolist())])


        print('Target copy indices')
        print([(i, x) for i, x in enumerate(batch["tgt_copy_indices"][batch_idx].tolist())])


    def forward(self, batch, for_training=False):

        # [batch, num_tokens]
        encoder_token_inputs = batch['src_tokens']['encoder_tokens']
        # [batch, num_tokens, num_chars]
        encoder_char_inputs = batch['src_tokens']['encoder_characters']
        # [batch, num_tokens]
        encoder_mask = get_text_field_mask(batch['src_tokens'])

        try:
            has_decoder_inputs = True
            # [batch, num_tokens]
            decoder_token_inputs = batch['tgt_tokens']['decoder_tokens'][:, :-1].contiguous()
            # [batch, num_tokens, num_chars]
            decoder_char_inputs = batch['tgt_tokens']['decoder_characters'][:, :-1].contiguous()
            # [batch, num_tokens]
            targets = batch['tgt_tokens']['decoder_tokens'][:, 1:].contiguous()

            vocab_targets = targets

            copy_targets = batch["tgt_copy_indices"][:, 1:]
            copy_attention_maps = batch['tgt_copy_map'][:, 1:]

            # TODO: use these two tensors for source side copy
            src_copy_targets = batch["src_copy_indices"]
            src_copy_attention_maps = batch['src_copy_map']

        except:
            has_decoder_inputs = False

        encoder_memory_bank, encoder_final_states = self.encode(
            encoder_token_inputs,
            encoder_char_inputs,
            encoder_mask
        )

        if for_training and has_decoder_inputs:
            decoder_memory_bank, aug_decoder_memory_bank, source_attentions, decoder_final_states = self.decode_for_training(
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
                copy_attentions = self.self_copy_attention(aug_decoder_memory_bank, aug_decoder_memory_bank)
                _generator_output = self.generator(decoder_memory_bank, aug_decoder_memory_bank, copy_attentions, copy_attention_maps)
                generator_output = self.generator.compute_loss(
                    _generator_output['probs'],
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

        decoder_outputs, augmented_decoder_outputs, attentions, decoder_final_states, _ = \
            self.decoder(decoder_inputs, memory_bank, mask, states)
        return decoder_outputs, augmented_decoder_outputs, attentions, decoder_final_states

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
            if self.use_self_copy:
                return self.greedy_decode_with_copy(memory_bank, mask, states)
            else:
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
            _decoder_outputs, _aug_decoder_outputs, _source_attentions, states, input_feed = \
                self.decoder(decoder_inputs, memory_bank, mask, states, input_feed)

            generator_output = self.generator(_decoder_outputs)

            # Update decoder outputs.
            source_attentions += _source_attentions
            decoder_outputs += [_decoder_outputs]

            # Generate.
            _predictions = generator_output['predictions']
            predictions.append(_predictions)

            tokens = _predictions

        return dict(
            # [batch_size, max_decode_length]
            predictions=torch.cat(predictions, dim=1),
            # [batch_size, max_decode_length, encoder_length]
            std_attentions=torch.cat(source_attentions, dim=1) if len(source_attentions) != 0 else None,
        )

    def greedy_decode_with_copy(self, memory_bank, mask, states):
        # [batch_size, 1]
        batch_size = memory_bank.size(0)
        tokens = torch.ones(batch_size, 1) \
                 * self.vocab.get_token_index(START_SYMBOL, "decoder_token_ids")
        tokens = tokens.type_as(mask).long()

        input_feed = None
        source_attentions = []
        copy_attentions = []
        aug_decoder_outputs = []
        predictions = []
        copy_indexes = []

        # A sparse indicator matrix mapping each node to its index in the dynamic vocab.
        # Here the maximum size of the dynamic vocab is just max_decode_length.
        attention_maps = torch.zeros(batch_size, self.max_decode_length, self.max_decode_length).type_as(memory_bank)
        # A diagonal matrix D where the element D_{i} is the real vocab index of the generated
        # node at the decoding step `i'.
        dynamic_vocab_maps = torch.zeros(batch_size, self.max_decode_length).type_as(mask).long()

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

                char_cnn_output = self._get_decoder_char_cnn_output(chars)
                decoder_inputs = torch.cat([token_embeddings, char_cnn_output], 2)
            else:
                decoder_inputs = token_embeddings
            decoder_inputs = self.decoder_embedding_dropout(decoder_inputs)

            # Decode one step.
            _decoder_outputs, _aug_decoder_outputs, _source_attentions, states, input_feed = \
                self.decoder(decoder_inputs, memory_bank, mask, states, input_feed)

            if step_i == 0:
                # Dummy copy attention for the first step will never be chosen.
                _copy_attentions = _decoder_outputs.new_ones(batch_size, 1, 1).type_as(memory_bank)
                _attention_maps = _decoder_outputs.new_zeros(batch_size, 1, 1).type_as(memory_bank)
            else:
                decoder_outputs_by_far = torch.cat(aug_decoder_outputs, dim=1)
                _copy_attentions = self.self_copy_attention(_aug_decoder_outputs, decoder_outputs_by_far)
                _attention_maps = attention_maps[:, :step_i]

            # Generate.
            generator_output = self.generator(_decoder_outputs, _aug_decoder_outputs, _copy_attentions, _attention_maps)
            _predictions = generator_output['predictions']

            # Update decoder outputs.
            copy_attentions += [_copy_attentions]
            source_attentions += _source_attentions
            aug_decoder_outputs += [_aug_decoder_outputs]

            tokens, copy_index = self._update_maps_and_get_next_input(
                step_i, _predictions.squeeze(1), attention_maps, dynamic_vocab_maps)
            tokens = tokens.unsqueeze(1)

            predictions += [tokens]
            copy_indexes += [copy_index.unsqueeze(1)]

        predictions = torch.cat(predictions, dim=1)
        copy_indexes = torch.cat(copy_indexes, dim=1)
        source_attentions = torch.cat(source_attentions, dim=1)
        #copy_attentions = torch.cat(copy_attentions, dim=1)

        return dict(
            # [batch_size, max_decode_length]
            predictions=predictions,
            copy_indexes=copy_indexes,
            # [batch_size, max_decode_length, encoder_length]
            source_attentions=source_attentions,
            copy_attentions=copy_attentions
        )

    def _update_maps_and_get_next_input(self, step, predictions, attention_maps, dynamic_vocab_maps):
        """Dynamically update/build the maps needed for copying.

        :param step: the decoding step, int.
        :param predictions: [batch_size]
        :param attention_maps: [batch_size, max_decode_length, max_decode_length]
        :param dynamic_vocab_maps:  [batch_size, max_decode_length]
        :return:
        """
        vocab_size = self.generator.vocab_size
        batch_size = predictions.size(0)

        batch_index = torch.arange(0, batch_size).type_as(dynamic_vocab_maps).long()
        step_index = torch.tensor([step] * batch_size).type_as(dynamic_vocab_maps).long()
        copy_mask = predictions.ge(vocab_size)
        generate_mask = 1 - copy_mask

        # 1. Update attention_maps
        # Get the copy index.
        copy_index = (predictions - vocab_size)
        # Fill the place where copy didn't happen with the current step,
        # which means that the node doesn't refer to any precedent, it refers to itself.
        copy_index.masked_fill_(generate_mask, step)

        attention_maps[batch_index, step_index, copy_index] = 1

        # 2. Compute the next input.
        # copy_predictions have the dynamic vocabulary index, and OOVs are set to zero.
        copy_predictions = (predictions - vocab_size) * copy_mask.long()
        next_input = dynamic_vocab_maps.gather(1, copy_predictions.unsqueeze(1)).squeeze(1)
        # Merge it with generated predictions.
        next_input = next_input * copy_mask.long() + predictions * generate_mask.long()

        # 3. Update dynamic_vocab_maps
        # Here we update D_{step} to the index in the standard vocab.
        dynamic_vocab_maps[batch_index, step_index] = next_input

        return next_input, copy_index

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

        if params.get('use_self_copy', False):
            attention_module = DotProductAttention(
                decoder_hidden_size=params['decoder']['hidden_size'] * 2,
                encoder_hidden_size=params['decoder']['hidden_size'] * 2,
                add_linear=params['self_copy_attention'].get('add_linear', True)
            )
            self_copy_attention = SelfCopyAttention(
                attention=attention_module
            )
            generator = CopyGenerator(
                input_size=params['decoder']['hidden_size'],
                vocab_size=vocab.get_vocab_size('decoder_token_ids'),
                force_copy=params['generator'].get('force_copy', True),
                # TODO: Set the following indices.
                vocab_pad_idx=0
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
            use_self_copy=params.get('use_self_copy', False),
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
