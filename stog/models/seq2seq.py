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
from stog.modules.input_variational_dropout import InputVariationalDropout
from stog.modules.decoders.generator import Generator
from stog.utils.nn import get_text_field_mask

logger = init_logger()


class Seq2Seq(Model):

    def __init__(self,
                 use_char_cnn,
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
                 # Generator
                 generator):
        super(Seq2Seq, self).__init__()

        self.use_char_cnn = use_char_cnn
        
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

        self.generator = generator

    def get_metrics(self, reset: bool = False):
        return self.generator.metrics.get_metric(reset)

    def forward(self, batch, for_training=True):
        # TODO: Xutai
        # [batch, num_tokens]
        encoder_token_inputs = batch['src_tokens']['encoder_tokens']
        # [batch, num_tokens, num_chars]
        encoder_char_inputs = batch['src_tokens']['encoder_characters']
        # [batch, num_tokens]
        encoder_mask = get_text_field_mask(batch['src_tokens'])
        # [batch, num_tokens]
        decoder_token_inputs = batch['amr_tokens']['decoder_tokens'][:, :-1].contiguous()
        # [batch, num_tokens, num_chars]
        decoder_char_inputs = batch['amr_tokens']['decoder_characters'][:, :-1].contiguous()
        # [batch, num_tokens]
        targets = batch['amr_tokens']['decoder_tokens'][:, 1:].contiguous()

        # import pdb;pdb.set_trace()

        encoder_memory_bank, encoder_final_states = self.encode(
            encoder_token_inputs,
            encoder_char_inputs,
            encoder_mask
        )
        decoder_memory_bank, attentions, decoder_final_states = self.decode_for_training(
            decoder_token_inputs,
            decoder_char_inputs,
            encoder_memory_bank,
            encoder_mask,
            encoder_final_states
        )

        generator_output = self.generator.compute_loss(decoder_memory_bank, targets)

        return dict(
            loss=generator_output['loss'],
            predictions=generator_output['predictions'],
            attentions=attentions,
            decoder_final_states=decoder_final_states
        )

    def encode(self, tokens, chars, mask):
        # [batch, num_tokens, embedding_size]
        token_embeddings = self.encoder_token_embedding(tokens)
        if self.use_char_cnn:
            # [batch, num_tokens, num_chars, embedding_size]
            char_embeddings = self.encoder_char_embedding(chars)
            batch_size, num_tokens, num_chars, _ = char_embeddings.size()
            char_embeddings = char_embeddings.view(batch_size * num_tokens, num_chars, -1)
            # TODO: add mask?
            char_cnn_output = self.encoder_char_cnn(char_embeddings, None)
            char_cnn_output = char_cnn_output.view(batch_size, num_tokens, -1)

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
            # [batch, num_tokens, embedding_size]
            char_embeddings = self.decoder_char_embedding(chars)
            batch_size, num_tokens, num_chars, _ = char_embeddings.size()
            char_embeddings = char_embeddings.view(batch_size * num_tokens, num_chars, -1)
            # TODO: add mask?
            char_cnn_output = self.decoder_char_cnn(char_embeddings, None)
            char_cnn_output = char_cnn_output.view(batch_size, num_tokens, -1)

            decoder_inputs = torch.cat([token_embeddings, char_cnn_output], 2)
        else:
            decoder_inputs = token_embeddings

        decoder_inputs = self.decoder_embedding_dropout(decoder_inputs)

        decoder_outputs, attentions, decoder_final_states = self.decoder(decoder_inputs, memory_bank, mask, states)
        return decoder_outputs, attentions, decoder_final_states

    @classmethod
    def from_params(cls, vocab, recover, params):
        logger.info('Building the Seq2Seq Model...')

        # Encoder
        encoder_token_embedding = Embedding.from_params(vocab, recover, params['encoder_token_embedding'])
        if params['use_char_cnn']:
            encoder_char_embedding = Embedding.from_params(vocab, recover, params['encoder_char_embedding'])
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
        decoder_token_embedding = Embedding.from_params(vocab, recover, params['decoder_token_embedding'])
        if params['use_char_cnn']:
            decoder_char_embedding = Embedding.from_params(vocab, recover, params['decoder_char_embedding'])
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

        # Generator
        # TODO: Xutai, make sure I set them correctly.
        params['generator']['vocab_size'] = vocab.get_vocab_size('decoder_token_ids')
        params['generator']['pad_idx'] = 0
        generator = Generator.from_params(params['generator'])

        logger.info('encoder_token: %d' %vocab.get_vocab_size('encoder_token_ids'))
        logger.info('encoder_chars: %d' %vocab.get_vocab_size('encoder_token_characters'))
        logger.info('decoder_token: %d' %vocab.get_vocab_size('decoder_token_ids'))
        logger.info('decoder_chars: %d' %vocab.get_vocab_size('decoder_token_characters'))

        return cls(
            use_char_cnn=params['use_char_cnn'],
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
            generator=generator
        )
