import numpy as np
import torch
import torch.nn.functional as F

from .model import  Model
from stog.modules.token_embedders import Embedding
from stog.modules.seq2seq_encoders import PytorchSeq2SeqWrapper
from stog.modules.stacked_bilstm import StackedBidirectionalLstm
from stog.modules.attention import BiaffineAttention
from stog.modules.linear import BiLinear
from stog.metrics import AttachmentScores
from stog.algorithms.maximum_spanning_tree import decode_mst_with_coreference, decode_mst
from stog.utils.nn import masked_log_softmax
from stog.utils.nn import get_text_field_mask
from stog.utils.logging import init_logger
logger = init_logger()


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
            # Edge label classifier
            label_hidden_size,
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
        self.label_hidden_size = label_hidden_size
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

        # Hidden representation for ROOT.
        self.head_sentinel = torch.nn.Parameter(torch.randn([1, 1, encoder_output_size]))

        # Linear transformation for edge headers.
        self.edge_h = torch.nn.Linear(encoder_output_size, edge_hidden_size)
        # Linear transformation for edge modifiers.
        self.edge_m = torch.nn.Linear(encoder_output_size, edge_hidden_size)

        self._attention = BiaffineAttention(edge_hidden_size, edge_hidden_size)

        # Comment out because currently we don't consider edge labels.
        # Linear transformation for label headers.
        self.label_h = torch.nn.Linear(encoder_output_size, label_hidden_size)
        # Linear transformation for label modifiers.
        self.label_m = torch.nn.Linear(encoder_output_size, label_hidden_size)

        self.bilinear = BiLinear(label_hidden_size, label_hidden_size, num_labels)

        # Metrics
        self.accumulated_loss = 0.0
        self.accumulated_edge_loss = 0.0
        self.accumulated_label_loss = 0.0
        self.num_accumulated_tokens = 0
        self.metrics = AttachmentScores()

    def get_metrics(self, reset=False):
        metrics = dict(
            loss=self.accumulated_loss / self.num_accumulated_tokens,
            eloss=self.accumulated_edge_loss / self.num_accumulated_tokens,
            lloss=self.accumulated_label_loss / self.num_accumulated_tokens
        )
        metrics.update(self.metrics.get_metric(reset))
        if reset:
            self.accumulated_loss = 0.0
            self.accumulated_edge_loss = 0.0
            self.accumulated_label_loss = 0.0
            self.num_accumulated_tokens = 0
        return metrics

    def get_regularization_penalty(self):
        return 0.0

    def forward(self, batch, for_training=True):
        input_token = batch["amr_tokens"]["tokens"]
        input_char = batch["amr_tokens"]["characters"]
        headers = batch["head_indices"]
        labels = batch["head_tags"]
        coreference = batch.get('coref', None)
        mask = get_text_field_mask(batch["amr_tokens"]).float()
        num_tokens = mask.sum().item()

        encoder_output = self.encode(input_token, input_char, mask)
        encoder_output, headers, labels, mask, coreference = self.add_head_sentinel(
            encoder_output, headers, labels, mask, coreference)

        edge = self.mlp(encoder_output)
        edge_headers, edge_modifiers, label_headers, label_modifiers = edge
        edge_scores = self.attention(edge_headers, edge_modifiers, mask)

        if headers is not None and labels is not None:
            edge_nll, label_nll = self.compute_loss(
                label_headers, label_modifiers, edge_scores, headers, labels, mask)
            loss = edge_nll + label_nll

            self.accumulated_edge_loss += edge_nll.item()
            self.accumulated_label_loss += label_nll.item()
            self.accumulated_loss += loss.item()

            self.num_accumulated_tokens += num_tokens

            predicted_headers, predicted_header_labels = headers, labels
            if not for_training:
                predicted_headers, predicted_header_labels = self.predict(
                    label_headers, label_modifiers, edge_scores, mask, coreference)

                self.metrics(
                    predicted_headers[:, 1:],
                    predicted_header_labels[:, 1:],
                    headers[:, 1:],
                    labels[:, 1:],
                    mask[:, 1:])
        else:

            predicted_headers, predicted_header_labels = self.predict(
                label_headers, label_modifiers, edge_scores, mask, coreference)
            edge_nll, label_nll = self.compute_loss(
                label_headers, label_modifiers, edge_scores, predicted_headers, predicted_header_labels, mask)

            loss = edge_nll + label_nll

        return dict(
            headers=predicted_headers[:, 1:],
            relations=predicted_header_labels[:, 1:],
            mask=mask[:, 1:],
            loss=loss / num_tokens,
            edge_loss=edge_nll / num_tokens,
            label_loss=label_nll / num_tokens
        )

    def add_head_sentinel(self, encoder_output, headers, labels, mask, coreference):
        """
        Add a dummpy ROOT at the beginning of each sequence.
        :param encoder_output: [batch, length, hidden_size]
        :param headers: [batch, length]
        :param labels: [batch, length]
        :param mask: [batch, length]
        :param coreference: [batch, length] or None
        """
        batch_size, _, hidden_size = encoder_output.size()
        head_sentinel = self.head_sentinel.expand([batch_size, 1, hidden_size])
        encoder_output = torch.cat([head_sentinel, encoder_output], 1)
        headers = torch.cat([headers.new_zeros(batch_size, 1), headers], 1)
        labels = torch.cat([labels.new_zeros(batch_size, 1), labels], 1)
        mask = torch.cat([mask.new_ones(batch_size, 1), mask], 1)
        if coreference is not None:
            coreference = torch.cat([coreference.new_zeros(batch_size, 1), coreference], 1)
        return  encoder_output, headers, labels, mask, coreference

    def remove_head_sentinel(self, headers, labels):
        """
        Remove the dummpy ROOT at the beginning of each sequence.
        :param headers: [batch, length + 1]
        :param labels: [batch, length + 1]
        """
        headers = headers[:, 1:]
        labels = labels[:, 1:]
        return  headers, labels


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

    def compute_header_label_logits(self, label_headers, label_modifiers, headers):
        """
        Compute the edge label logits.
        :param label_headers: [batch, length, label_hidden_size]
        :param label_modifiers: [batch, length, label_hidden_size]
        :param headers: [batch, length] -- header at [i, j] means the header index of token_j at batch_i.
        :param mask: [batch, length]
        :return: [batch, length, num_labels]
        """
        batch_size = label_headers.size(0)
        # Create indexing matrix for batch: [batch, 1]
        batch_index = torch.arange(0, batch_size).view(batch_size, 1)
        batch_index = batch_index.type_as(label_headers.data).long()

        # Select the corresponding header representations
        # based on gold/predicted headers.
        # [batch, length, label_hidden_size]
        label_selected_headers = label_headers[batch_index, headers]

        label_selected_headers = label_selected_headers.contiguous()
        label_modifiers = label_modifiers.contiguous()

        # [batch, length, num_labels]
        label_logits = self.bilinear(label_selected_headers, label_modifiers)

        return label_logits

    def compute_loss(self, label_headers, label_modifiers, edge_scores, headers, header_labels, mask):
        """
        :param label_headers: [batch, length, label_hidden_size]
        :param label_modifiers: [batch, length, label_hidden_size]
        :param edge_scores: [batch, header_length, modifier_length]
        :param mask: [batch, length]
        """
        batch_size, max_len, _ = edge_scores.size()
        float_mask = mask.float()

        edge_log_likelihood = masked_log_softmax(edge_scores, mask.unsqueeze(2) + mask.unsqueeze(1), dim=1)

        header_label_logits = self.compute_header_label_logits(label_headers, label_modifiers, headers)
        label_log_likelihood = F.log_softmax(header_label_logits, dim=2)

        # Create indexing matrix for batch: [batch, 1]
        batch_index = torch.arange(0, batch_size).view(batch_size, 1)
        batch_index = batch_index.type_as(edge_log_likelihood.data).long()
        # Create indexing matrix for modifier: [batch, modifier_length]
        modifier_index = torch.arange(0, max_len).view(1, max_len).expand(batch_size, max_len)
        modifier_index = modifier_index.type_as(edge_log_likelihood.data).long()
        # Index the log likelihood of gold edges (ROOT excluded).
        # Output [batch, length - 1]
        gold_edge_log_likelihood = edge_log_likelihood[batch_index, headers.data, modifier_index][:, 1:]
        gold_label_log_likelihood = label_log_likelihood[batch_index, modifier_index, header_labels.data][:, 1:]

        return -gold_edge_log_likelihood.sum(), -gold_label_log_likelihood.sum()

    def predict(self, label_headers, label_modifiers, edge_scores, mask, coreference=None):
        if self.decode_type == 'mst':
            return self.mst_decode(label_headers, label_modifiers, edge_scores, mask, coreference)
        else:
            return self.greedy_decode(label_headers, label_modifiers, edge_scores, mask)

    def greedy_decode(self, label_headers, label_modifiers, edge_scores, mask):
        # out_arc shape [batch, length, length]
        edge_scores = edge_scores.data
        max_len = edge_scores.size(1)

        # Set diagonal elements to -inf
        edge_scores = edge_scores + torch.diag(edge_scores.new(max_len).fill_(-np.inf))

        # Set invalid positions to -inf
        minus_mask = (1 - mask.float()) * self.minus_inf
        edge_scores = edge_scores + minus_mask.unsqueeze(2) + minus_mask.unsqueeze(1)

        # Compute naive predictions.
        # predition shape = [batch, length]
        _, header_indices = edge_scores.max(dim=1)

        # Based on predicted headers, compute the edge label logits.
        # [batch, length, num_labels]
        header_label_logits = self.compute_header_label_logits(label_headers, label_modifiers, header_indices)
        _, header_labels = header_label_logits.max(dim=2)

        return header_indices, header_labels

    def mst_decode(self, label_headers, label_modifiers, edge_scores, mask, coreference=None):
        batch_size, max_length, label_hidden_size = label_headers.size()
        lengths = mask.data.sum(dim=1).long().cpu().numpy()

        expanded_shape = [batch_size, max_length, max_length, label_hidden_size]
        label_headers = label_headers.unsqueeze(2).expand(*expanded_shape).contiguous()
        label_modifiers = label_modifiers.unsqueeze(1).expand(*expanded_shape).contiguous()
        # [batch, max_header_length, max_modifier_length, num_labels]
        pairwise_label_logits = self.bilinear(label_headers, label_modifiers)

        normalized_edge_label_logits = F.log_softmax(pairwise_label_logits, dim=3).permute(0, 3, 1, 2)

        # Set invalid positions to -inf
        minus_mask = (1 - mask.float()) * self.minus_inf
        edge_scores = edge_scores + minus_mask.unsqueeze(2) + minus_mask.unsqueeze(1)
        # [batch, max_header_length, max_modifier_length]
        normalized_edge_logits = F.log_softmax(edge_scores, dim=1)

        # [batch, num_labels, max_header_length, max_modifier_length]
        batch_energy = torch.exp(normalized_edge_logits.unsqueeze(1) + normalized_edge_label_logits)

        return self._run_mst_decoding(batch_energy, lengths, mask, coreference)

    @staticmethod
    def _run_mst_decoding(batch_energy, lengths, mask, coreference=None):
        heads = []
        head_labels = []
        for i, (energy, length) in enumerate(zip(batch_energy.detach().cpu(), lengths)):
            # energy: [num_labels, max_header_length, max_modifier_length]
            # scores | label_ids : [max_header_length, max_modifier_length]
            scores, label_ids = energy.max(dim=0)
            # Although we need to include the root node so that the MST includes it,
            # we do not want the dummpy root node to be the head of more than one nodes,
            # since there should be only one root in a sentence.
            # Here, we enforce this by setting the scores for all word -> ROOT edges
            # edges to be 0.
            # TODO: set it to -1 seems better?
            scores[0, :] = 0
            # Decode the heads. Because we modify the scores to prevent
            # adding in word -> ROOT edges, we need to find the labels ourselves.
            if coreference is not None:
                coref = coreference[i].detach().cpu().tolist()[:length]
                instance_heads, _ = decode_mst_with_coreference(
                    scores.numpy(), coref, length, has_labels=False)
            else:
                instance_heads, _ = decode_mst(scores.numpy(), length, has_labels=False)

            # Find the labels which correspond to the edges in the max spanning tree.
            instance_head_labels = []
            for child, parent in enumerate(instance_heads):
                instance_head_labels.append(label_ids[parent, child].item())
            # We don't care what the head or tag is for the root token, but by default it's
            # not necessarily the same in the batched vs unbatched case, which is annoying.
            # Here we'll just set them to zero.
            instance_heads[0] = 0
            instance_head_labels[0] = 0
            heads.append(instance_heads)
            head_labels.append(instance_head_labels)
        return torch.from_numpy(np.stack(heads)), torch.from_numpy(np.stack(head_labels))


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
            label: a tuple of (header, modifier) hidden state with size [batch, length, label_hidden_size]
        """

        # Output: [batch, length, edge_hidden_size]
        edge_h = F.elu(self.edge_h(input))
        edge_m = F.elu(self.edge_m(input))

        # Output: [batch, length, label_hidden_size]
        label_h = F.elu(self.label_h(input))
        label_m = F.elu(self.label_m(input))

        # Apply dropout to certain node?
        # [batch, length * 2, hidden_size]
        edge = torch.cat([edge_h, edge_m], dim=1)
        label = torch.cat([label_h, label_m], dim=1)
        edge = self.hidden_state_dropout(edge.transpose(1, 2)).transpose(1, 2)
        label = self.hidden_state_dropout(label.transpose(1, 2)).transpose(1, 2)

        edge_h, edge_m = edge.chunk(2, 1)
        label_h, label_m = label.chunk(2, 1)

        return edge_h, edge_m, label_h, label_m

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

    def load_embedding(self, field, file, vocab, data_type):
        assert field in ["chars", "tokens"]
        if field == "tokens":
            self.token_embedding.load_pretrain_from_file(vocab, file, "token_ids", data_type=="AMR")
        if field == "chars":
            self.char_embedding.load_pretrain_from_file(vocab, file, "token_characters", data_type=="AMR")

    @classmethod
    def from_params(cls, vocab, recover, params, data_params):
        logger.info('Building model...')
        token_emb_size = params['token_emb_size']
        char_emb_size = params['char_emb_size']
        emb_dropout = params['emb_dropout']
        hidden_dropout = params['hidden_dropout']
        use_char_conv = params['use_char_conv']
        num_filters = params['num_filters']
        kernel_size = params['kernel_size']
        encoder_size = params['encoder_size']
        encoder_layers = params['encoder_layers']
        encoder_dropout = params['encoder_dropout']
        edge_hidden_size = params['edge_hidden_size']
        label_hidden_size = params['label_hidden_size']
        decode_type = params['decode_type']

        pretrain_token_emb = data_params['pretrain_token_emb']
        pretrain_char_emb = data_params['pretrain_char_emb']
        data_type = data_params['data_type']

        model = DeepBiaffineParser(
            num_token_embeddings=vocab.get_vocab_size("token_ids"),
            token_embedding_dim=token_emb_size,
            num_char_embeddings=vocab.get_vocab_size("token_characters"),
            char_embedding_dim=char_emb_size,
            embedding_dropout_rate=emb_dropout,
            hidden_state_dropout_rate=hidden_dropout,
            use_char_conv=use_char_conv,
            num_filters=num_filters,
            kernel_size=kernel_size,
            encoder_hidden_size=encoder_size,
            num_encoder_layers=encoder_layers,
            encoder_dropout_rate=encoder_dropout,
            edge_hidden_size=edge_hidden_size,
            label_hidden_size=label_hidden_size,
            num_labels=vocab.get_vocab_size("head_tags"),
            decode_type=decode_type
        )


        if not recover and pretrain_token_emb:
            logger.info("Reading pretrained token embeddings from {} ...".format(pretrain_token_emb))
            model.load_embedding(
                field="tokens",
                file=pretrain_token_emb,
                vocab=vocab,
                data_type=data_type
            )
            logger.info("Done.")

        if not recover and pretrain_char_emb:
            logger.info("Reading pretrained char embeddings from {} ...".format(pretrain_char_emb))
            model.load_embedding(
                field="chars",
                file=pretrain_char_emb,
                vocab=vocab,
                data_type=data_type
            )
            logger.info("Done.")

        logger.info(model)
        return model
