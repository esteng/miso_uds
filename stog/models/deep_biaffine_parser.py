import numpy as np
import torch
import torch.nn.functional as F

from .model import  Model
from stog.modules.embedding import Embedding
from stog.modules.seq2seq_encoders import PytorchSeq2SeqWrapper
from stog.modules.stacked_bilstm import StackedBidirectionalLstm
from stog.modules.attention import BiaffineAttention
from stog.modules.linear import BiLinear
from stog.metrics import UnlabeledAttachScore as UAS
from stog.algorithms import maximum_spanning_tree as MST


class DeepBiaffineParser(Model, torch.nn.Module):
    """
    Adopted from NeuroNLP2:
        https://github.com/XuezheMax/NeuroNLP2/blob/master/neuronlp2/models/parsing.py

    Deep Biaffine Attention Parser was originally used in dependency parsing.
    See https://arxiv.org/abs/1611.01734
    """

    # TODO: change it to -np.inf?
    minus_inf = -1e8

    def __init__(
            self,
            # Embedding
            num_token_embeddings,
            token_embedding_dim,
            num_char_embeddings,
            char_embedding_dim,
            embedding_dropout_rate,
            hidden_state_dropout_rate,
            # Character CNN
            use_char_conv,
            num_filters,
            kernel_size,
            # Encoder
            encoder_hidden_size,
            num_encoder_layers,
            encoder_dropout_rate,
            # Attention
            edge_hidden_size,
            # Edge type classifier
            type_hidden_size,
            num_labels,
            # Decode
            decode_type

    ):
        super(DeepBiaffineParser, self).__init__()
        self.num_token_embeddings = num_char_embeddings
        self.token_embedding_dim = token_embedding_dim
        self.num_char_embeddings = num_char_embeddings
        self.char_embedding_dim = char_embedding_dim
        self.embedding_dropout_rate = embedding_dropout_rate
        self.hidden_state_dropout_rate = hidden_state_dropout_rate
        self.use_char_conv = use_char_conv
        self.num_filters = num_filters
        self.kernel_size = kernel_size
        self.encoder_input_size = token_embedding_dim + num_filters if use_char_conv else token_embedding_dim
        self.encoder_hidden_size = encoder_hidden_size
        self.num_encoder_layers = num_encoder_layers
        self.encoder_dropout = encoder_dropout_rate
        self.edge_hidden_size = edge_hidden_size
        self.type_hidden_size = type_hidden_size
        self.num_labels = num_labels
        self.decode_type = decode_type


        self.token_embedding = Embedding(
            num_token_embeddings,
            token_embedding_dim
        )
        self.char_embedding = Embedding(
            num_char_embeddings,
            char_embedding_dim
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
            self.encoder_input_size,
            encoder_hidden_size,
            num_encoder_layers,
            encoder_dropout_rate
        ))

        encoder_output_size = self.encoder_hidden_size * 2
        # Linear transformation for edge headers.
        self.edge_h = torch.nn.Linear(encoder_output_size, edge_hidden_size)
        # Linear transformation for edge modifiers.
        self.edge_m = torch.nn.Linear(encoder_output_size, edge_hidden_size)

        self._attention = BiaffineAttention(edge_hidden_size, edge_hidden_size)

        # Comment out because currently we don't consider edge types.
        # Linear transformation for type headers.
        # self.type_h = torch.nn.Linear(encoder_output_size, type_hidden_size)
        # Linear transformation for type modifiers.
        # self.type_m = torch.nn.Linear(encoder_output_size, type_hidden_size)

        self.bilinear = BiLinear(type_hidden_size, type_hidden_size, num_labels)

        # Metrics
        self.accumulated_loss = 0.0
        self.num_accumulated_tokens = 0
        self.uas = UAS()

    def get_metrics(self, reset=False):
        metrics = dict(
            loss=self.accumulated_loss / self.num_accumulated_tokens,
            UAS=self.uas.score
        )
        if reset:
            self.accumulated_loss = 0.0
            self.num_accumulated_tokens = 0
            self.uas.reset()
        return metrics

    def get_regularization_penalty(self):
        return 0.0

    def forward(self, batch, for_training=True,):
        input_token = batch.tokens
        input_char = batch.chars
        headers, mask = batch.headers
        encoder_output = self.encode(input_token, input_char, mask)
        edge = self.mlp(encoder_output)
        edge_headers, edge_modifiers = edge
        edge_scores = self.attention(edge_headers, edge_modifiers, mask)
        edge_log_likelihood = self.compute_edge_log_likelihood(edge_scores, mask)
        num_tokens = mask.sum().item() - mask.size(0)

        if for_training or headers is not None:
            loss = self.compute_loss(edge_log_likelihood, headers)
            #TODO: Metric for graph includes type.
            pred_headers = self.decode(edge_log_likelihood, mask)
            self.uas(pred_headers, headers, mask)
            self.accumulated_loss += loss.item()
            self.num_accumulated_tokens += num_tokens
        else:
            loss = 0.0
            pred_headers = self.decode(edge_log_likelihood, mask)

        return dict(
            pred_headers=pred_headers,
            loss=loss / num_tokens,
        )

    def compute_edge_log_likelihood(self, edge_scores, mask):
        """
        :param edge_scores: [batch, header_length, modifier_length]
        :param mask: [bath, length]
        """
        # Make pad position -inf for log_softmax
        minus_mask = (1 - mask) * self.minus_inf
        edge_scores = edge_scores + minus_mask.unsqueeze(2) + minus_mask.unsqueeze(1)

        # Compute the edge log likelihood.
        # [batch, header_length, modifier_length]
        edge_log_likelihood = F.log_softmax(edge_scores, dim=1)

        # Make pad position 0 for sum of loss
        edge_log_likelihood = edge_log_likelihood * mask.unsqueeze(2) * mask.unsqueeze(1)

        return edge_log_likelihood

    def compute_loss(self, edge_log_likelihood, headers):
        """
        :param edge_log_likelihood: [batch, header_length, modifier_length]
        :param headers: [batch, length] -- header at [i, j] means the header index of token_j at batch_i.
        """
        # Total number of headers to predict (ROOT excluded).
        batch_size, max_len, _ = edge_log_likelihood.size()

        # Create indexing matrix for batch: [batch, 1]
        batch_index = torch.arange(0, batch_size).view(batch_size, 1)
        batch_index = batch_index.type_as(edge_log_likelihood.data).long()
        # Create indexing matrix for modifier: [batch, modifier_length]
        modifier_index = torch.arange(0, max_len).view(1, max_len).expand(batch_size, max_len)
        modifier_index = modifier_index.type_as(edge_log_likelihood.data).long()
        # Index the log likelihood of gold edges (ROOT excluded).
        # Output [batch, length - 1]
        gold_edge_log_likelihood = edge_log_likelihood[batch_index, headers.data, modifier_index][:, 1:]

        return -gold_edge_log_likelihood.sum()

    def decode(self, edge_scores, mask):
        if self.decode_type == 'mst':
            return self.mst_decode(edge_scores, mask)
        else:
            return self.greedy_decode(edge_scores, mask)

    def greedy_decode(self, edge_scores, mask=None):
        # out_arc shape [batch, length, length]
        edge_scores = edge_scores.data
        batch, max_len, _ = edge_scores.size()

        # set diagonal elements to -inf
        edge_scores += torch.diag(edge_scores.new(max_len).fill_(-np.inf))

        # set invalid positions to -inf
        # minus_mask = (1 - mask.data).byte().view(batch, max_len, 1)
        minus_mask = (1 - mask.data).byte().unsqueeze(2)
        edge_scores.masked_fill_(minus_mask, -np.inf)

        # compute naive predictions.
        # predition shape = [batch, length]
        _, headers = edge_scores.max(dim=1)

        return headers.cpu().numpy()

    def mst_decode(self, edge_scores, mask):
        length = mask.sum(dim=1).long().cpu().numpy()
        pred_headers, _ = MST.decode(
            edge_scores.detach().cpu().numpy(),
            length,
            num_leading_symbols=1,
            labeled=False
        )
        return pred_headers

    def encode(self, input_token, input_char, mask):
        """
        Encode input sentence into a list of hidden states by a stacked BiLSTM.

        :param input_token: [batch, token_length]
        :param input_char:  [batch, token_length, char_length]
        :param mask: [batch, token_length]
        :return: [batch, length, hidden_size]
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

        # Output: [batch, length, hidden_size]
        output = self.encoder(input, mask)

        # Apply dropout to certain step?
        output = self.hidden_state_dropout(output.transpose(1, 2)).transpose(1, 2)

        return output

    def mlp(self, input):
        """
        Map contextual representation into specific space (w/ lower dimensionality).

        :param input: [batch, length, encoder_hidden_size]
        :return:
            edge: a tuple of (header, modifier) hidden state with size [batch, length, edge_hidden_size]
            type: a tuple of (header, modifier) hidden state with size [batch, length, type_hidden_size]
        """

        # Output: [batch, length, edge_hidden_size]
        edge_h = F.elu(self.edge_h(input))
        edge_m = F.elu(self.edge_m(input))

        # Output: [batch, length, type_hidden_size]
        # type_h = F.elu(self.type_h(input))
        # type_m = F.elu(self.type_m(input))

        # Apply dropout to certain node?
        # [batch, length * 2, hidden_size]
        edge = torch.cat([edge_h, edge_m], dim=1)
        # type = torch.cat([type_h, type_m], dim=1)
        edge = self.hidden_state_dropout(edge.transpose(1, 2)).transpose(1, 2)
        # type = self.hidden_state_dropout(type.transpose(1, 2)).transpose(1, 2)

        edge_h, edge_m = edge.chunk(2, 1)
        # type_h, type_m = type.chunk(2, 1)

        return edge_h, edge_m

    def attention(self, input_header, input_modifier, mask):
        """
        Compute attention between headers and modifiers.

        :param input_header:  [batch, header_length, hidden_size]
        :param input_modifier: [batch, modifier_length, hidden_size]
        :param mask: [batch, length, hidden_size]
        :return: [batch, header_length, modifier_length]
        """
        output = self._attention(input_header, input_modifier, mask_d=mask, mask_e=mask).squeeze(dim=1)
        return output

    def load_embedding(self, field, file, vocab):
        assert field in ["chars", "tokens"]
        if field == "chars":
            self.char_embedding.load_pretrain_from_file(vocab, file)
        if field == "tokens":
            self.token_embedding.load_pretrain_from_file(vocab, file)
