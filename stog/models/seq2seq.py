import torch

from stog.models.model import Model
from stog.utils.logging import init_logger
from stog.modules.token_embedders.embedding import Embedding
from stog.modules.seq2seq_encoders.pytorch_seq2seq_wrapper import PytorchSeq2SeqWrapper
from stog.modules.stacked_bilstm import StackedBidirectionalLstm
from stog.modules.stacked_lstm import StackedLstm
from stog.modules.decoders.rnn_decoder import InputFeedRNNDecoder
from stog.modules.attention_layers.global_attention import GlobalAttention
from stog.modules.attention.dot_production_attention import DotProductAttention
from stog.modules.input_variational_dropout import InputVariationalDropout
from stog.modules.decoders.generator import Generator

logger = init_logger()


class Seq2Seq(Model):

    def __init__(self,
                 # Encoder
                 encoder_token_embedding,
                 encoder_char_embedding,
                 encoder_embedding_dropout,
                 encoder,
                 encoder_output_dropout,
                 # Decoder
                 decoder_token_embedding,
                 decoder_char_embedding,
                 decoder_embedding_dropout,
                 decoder,
                 # Generator
                 generator):

        self.encoder_token_embedding = encoder_token_embedding
        self.encoder_char_embedding = encoder_char_embedding
        self.encoder_embedding_dropout = encoder_embedding_dropout
        self.encoder = encoder
        self.encoder_output_dropout = encoder_output_dropout

        self.decoder_token_embedding = decoder_token_embedding
        self.decoder_char_embedding = decoder_char_embedding
        self.decoder_embedding_dropout = decoder_embedding_dropout
        self.decoder = decoder

        self.generator = generator

    def get_metrics(self, reset: bool = False):
        return self.generator.metrics.get_metric(reset)

    def forward(self, batch, for_training=True):
        # [batch, num_tokens]
        encoder_token_inputs = batch['encoder_token_inputs']
        # [batch, num_tokens, num_chars]
        encoder_char_inputs = batch['encoder_char_inputs']
        # [batch, num_tokens]
        encoder_mask = batch['encoder_mask']
        # [batch, num_tokens]
        decoder_token_inputs = batch['decoder_token_inputs']
        # [batch, num_tokens, num_chars]
        decoder_char_inputs = batch['decoder_char_inputs']
        # [batch, num_tokens]
        targets = batch['targets']

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
        # [batch, num_tokens, embedding_size]
        char_embeddings = self.encoder_char_embedding(chars)
        encoder_inputs = torch.cat([token_embeddings, char_embeddings], 2)
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
        # [batch, num_tokens, embedding_size]
        char_embeddings = self.decoder_char_embedding(chars)
        decoder_inputs = torch.cat([token_embeddings, char_embeddings], 2)
        decoder_inputs = self.decoder_embedding_dropout(decoder_inputs)

        decoder_outputs, attentions, decoder_final_states = self.decoder(decoder_inputs, memory_bank, mask, states)
        return decoder_outputs, attentions, decoder_final_states

    @classmethod
    def from_params(cls, vocab, recover, params):
        logger.info('Building the Seq2Seq Model...')

        # Encoder
        encoder_token_embedding = Embedding.from_params(vocab, recover, params['encoder_token_embedding'])
        encoder_char_embedding = Embedding.from_params(vocab, recover, params['encoder_char_embedding'])
        encoder_embedding_dropout = InputVariationalDropout(p=params['encoder_token_embedding']['dropout'])

        encoder = PytorchSeq2SeqWrapper(
            module=StackedBidirectionalLstm.from_params(params['encoder']),
            stateful=True
        )
        encoder_output_dropout = InputVariationalDropout(p=params['encoder']['dropout'])

        # Decoder
        decoder_token_embedding = Embedding.from_params(vocab, recover, params['encoder_token_embeddings'])
        decoder_char_embedding = Embedding.from_params(vocab, recover, params['encoder_char_embeddings'])
        decoder_embedding_dropout = InputVariationalDropout(p=params['decoder_token_embedding']['dropout'])

        attention = DotProductAttention(
            decoder_hidden_size=params['decoder']['hidden_size'],
            encoder_hidden_size=params['encoder']['hidden_size'],
            add_linear=params['attention'].get('add_linear', True)
        )
        attention_layer = GlobalAttention(
            decoder_hidden_size=params['decoder']['hidden_size'],
            encoder_hidden_size=params['encoder']['hidden_size'],
            attention=attention
        )
        decoder = InputFeedRNNDecoder(
            rnn_cell=StackedLstm.from_params(params['decoder']),
            attention_layer=attention_layer,
            # TODO: modify the dropout so that the dropout mask is unchanged across the steps.
            dropout=InputVariationalDropout(p=params['decoder']['dropout'])
        )

        # Generator
        generator = Generator.from_params(params['generator'])

        return cls(
            encoder_token_embedding=encoder_token_embedding,
            encoder_char_embedding=encoder_char_embedding,
            encoder_embedding_dropout=encoder_embedding_dropout,
            encoder=encoder,
            encoder_output_dropout=encoder_output_dropout,
            decoder_token_embedding=decoder_token_embedding,
            decoder_char_embedding=decoder_char_embedding,
            decoder_embedding_dropout=decoder_embedding_dropout,
            decoder=decoder,
            generator=generator
        )
