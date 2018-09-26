import torch

from stog.modules.embeddings import Embedding
from stog.modules.seq2seq_encoders import PytorchSeq2SeqWrapper
from stog.modules.stacked_bilstm import StackedBidirectionalLstm
from stog.modules.attention import BiaffineAttention
from stog.modules.linear import BiLinear


class DeepBiaffineParser(torch.nn.Module):
    """
    Adopted from NeuroNLP2:
        https://github.com/XuezheMax/NeuroNLP2/blob/master/neuronlp2/models/parsing.py

    Deep Biaffine Attention Parser was originally used in dependency parsing.
    See https://arxiv.org/abs/1611.01734
    """

    def __init__(
            self,
            # Embedding
            num_token_embeddings,
            token_embedding_dim,
            token_embedding_weight,
            num_char_embeddings,
            char_embedding_dim,
            char_embedding_weight,
            embedding_dropout_rate,
            hidden_state_dropout_rate,
            # Character CNN
            use_char_conv,
            num_filters,
            kernel_size,
            # Encoder
            encoder_input_size,
            encoder_hidden_size,
            num_encoder_layers,
            encoder_dropout_rate,
            # Attention
            edge_hidden_size,
            # Edge type classifier
            type_hidden_size,
            num_labels

    ):
        super(DeepBiaffineParser, self).__init__()
        self.num_token_embeddings = num_char_embeddings
        self.token_embedding_dim = token_embedding_dim
        self.token_embedding_weight = token_embedding_weight
        self.num_char_embeddings = num_char_embeddings
        self.char_embedding_dim = char_embedding_dim
        self.char_embedding_weight = char_embedding_weight
        self.embedding_dropout_rate = embedding_dropout_rate
        self.hidden_state_dropout_rate = hidden_state_dropout_rate
        self.use_char_conv = use_char_conv
        self.num_filters = num_filters
        self.kernel_size = kernel_size
        self.encoder_input_size = encoder_input_size
        self.encoder_hidden_size = encoder_hidden_size
        self.num_encoder_layers = num_encoder_layers
        self.encoder_dropout = encoder_dropout_rate
        self.edge_hidden_size = edge_hidden_size
        self.type_hidden_size = type_hidden_size
        self.num_labels = num_labels


        self.token_embedding = Embedding(
            num_token_embeddings,
            token_embedding_dim,
            token_embedding_weight
        )
        self.char_embedding = Embedding(
            num_char_embeddings,
            char_embedding_dim,
            char_embedding_weight
        )

        self.embedding_dropout = torch.nn.Dropout2d(p=embedding_dropout_rate)
        self.hidden_state_dropout = torch.nn.Dropout2d(p=hidden_state_dropout_rate)

        self.char_conv = None
        if use_char_conv:
            self.char_conv = torch.nn.Conv1d(
                char_embedding_dim,
                num_filters,
                kernel_size,
                padding=kernel_size - 1
            )

        self.encoder = PytorchSeq2SeqWrapper(StackedBidirectionalLstm(
            encoder_input_size,
            encoder_hidden_size,
            num_encoder_layers,
            encoder_dropout_rate
        ))

        encoder_output_size = encoder_input_size * 2
        # Linear transformation for edge headers.
        self.edge_h = torch.nn.Linear(encoder_input_size, edge_hidden_size)
        # Linear transformation for edge modifiers.
        self.edge_m = torch.nn.Linear(encoder_input_size, edge_hidden_size)

        self.attention = BiaffineAttention(edge_hidden_size, edge_hidden_size)

        # Linear transformation for type headers.
        self.type_h = torch.nn.Linear(encoder_output_size, type_hidden_size)
        # Linear transformation for type modifiers.
        self.type_m = torch.nn.Linear(encoder_output_size, type_hidden_size)

        self.bilinear = BiLinear(type_hidden_size, type_hidden_size, num_labels)

    def forward(self, input_token, input_char, mask):
        """
        :param input_token: [batch, token_length]
        :param input_char:  [batch, token_length, char_length]
        :param mask: [batch, token_length]
        :return:
        """
        # Output: [batch, length, token_dim]
        token = self.token_embedding(input_token)
        token = self.embedding_dropout(token)

        input = token
        if self.use_char_conv:
            # Output: [batch, length, char_length, char_dim]
            char = self.char_embedding(input_char)
            char_size = char.size()
            # First transform to [batch*length, char_length, char_dim]
            # Then transpose to [batch*length, char_dim, char_length]
            char = char.view(char_size[0] * char_size[1], char_size[2], char_size[3]).transpose(1, 2)
            # Put into CNN [batch*length, char_filters, char_length]
            # Then MaxPooling [batch*length, char_filters]
            char, _ = self.char_conv(char).max(dim=2)
            # Squash and reshape to [batch, length, char_filters]
            char = torch.tanh(char).view(char_size[0], char_size[1], -1)
            # Apply dropout on input
            char = self.embedding_dropout(char)
            # concatenate word and char [batch, length, word_dim+char_filter]
            input = torch.cat([input, char], dim=2)

        encoder_output, h = self.encoder(input, mask)


