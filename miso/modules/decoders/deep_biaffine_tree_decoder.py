import torch

from miso.utils.nn import masked_log_softmax
from miso.metrics import AttachmentScores
from miso.modules.attention import BiaffineAttention
from miso.modules.linear import BiLinear


class DeepBiaffineTreeDecoder(torch.nn.Module):

    def __init__(self,
                 sentinel: torch.FloatTensor,
                 edge_query_linear: torch.nn.Module,
                 edge_key_linear: torch.nn.Module,
                 label_query_linear: torch.nn.Module,
                 label_key_linear: torch.nn.Module,
                 dropout: torch.nn.Module,
                 attention: torch.nn.Module,
                 edge_label_bilinear: torch.nn.Module
                 ):
        super(DeepBiaffineTreeDecoder, self).__init__()
        self.sentinel = sentinel
        self.edge_query_linear = edge_query_linear
        self.edge_key_linear = edge_key_linear
        self.label_query_linear = label_query_linear
        self.label_key_linear = label_key_linear
        self.dropout = dropout
        self.attention = attention
        self.edge_label_bilinear = edge_label_bilinear

        self.metrics = AttachmentScores()

        self.minus_inf = -1e8

    def forward(self, queries: torch.FloatTensor, keys: torch.FloatTensor, mask: torch.ByteTensor = None):
        """
        :param queries: [batch_size, query_length, query_hidden_size]
        :param keys: [batch_size, key_length, key_hidden_size]
        :param mask: None or [batch_size, query_length, key_length]
                        1 indicates a valid position; otherwise, 0.
        :return:
            edge_heads: [batch_size, query_length]
            edge_labels: [batch_size, query_length]
        """
        keys, mask = self._add_sentinel(keys, mask)
        edge_query_hiddens, edge_key_hiddens, label_query_hiddens, label_key_hiddens = self._mlp(queries, keys)
        edge_scores = self._get_edge_scores(edge_query_hiddens, edge_key_hiddens)
        edge_heads, edge_labels = self._decode(label_query_hiddens, label_key_hiddens, edge_scores, mask)
        # Note: head indices start from 1.
        return edge_heads, edge_labels

    def get_loss(self,
                 queries: torch.FloatTensor,
                 keys: torch.FloatTensor,
                 edge_heads: torch.LongTensor,
                 edge_labels: torch.LongTensor,
                 mask: torch.ByteTensor = None,
                 query_mask: torch.ByteTensor = None
                 ):
        """
        Compute the loss.
        :param queries: [batch_size, query_length, hidden_size]
        :param keys: [batch_size, key_length, hidden_size]
        :param edge_heads: [batch_size, query_length]
                        head indices start from 1.
        :param edge_labels: [batch_size, query_length]
        :param mask: None or [batch_size, query_length, key_length]
                        1 indicates a valid position; otherwise, 0.
        :param query_mask: None or [batch_size, query_length]
                        1 indicates a valid position; otherwise, 0.
        :return:
        """
        batch_size, query_length, _ = queries.size()
        keys, mask = self._add_sentinel(keys, mask)
        edge_query_hiddens, edge_key_hiddens, label_query_hiddens, label_key_hiddens = self._mlp(queries, keys)

        # [batch_size, query_length, key_length]
        edge_scores = self._get_edge_scores(edge_query_hiddens, edge_key_hiddens)
        edge_head_ll = masked_log_softmax(edge_scores, mask, dim=2)

        # [batch_size, query_length, num_labels]
        label_scores = self._get_label_scores(label_query_hiddens, label_key_hiddens, edge_heads)
        edge_label_ll = torch.nn.functional.log_softmax(label_scores, dim=2)

        batch_index = torch.arange(0, batch_size).view(batch_size, 1).type_as(edge_heads)
        modifier_index = torch.arange(0, query_length).view(
            1, query_length).expand(batch_size, query_length).type_as(edge_heads)
        # [batch_size, query_length]
        gold_edge_head_ll = edge_head_ll[batch_index, modifier_index, edge_heads]
        gold_edge_label_ll = edge_label_ll[batch_index, modifier_index, edge_labels]
        if query_mask is not None:
            gold_edge_head_ll.masked_fill_(1 - query_mask, 0)
            gold_edge_label_ll.masked_fill_(1 - query_mask, 0)

        edge_head_nll = - gold_edge_head_ll.sum()
        edge_label_nll = - gold_edge_label_ll.sum()
        num_instances = query_mask.sum().float()

        pred_heads, pred_labels = self._decode(
            label_query_hiddens, label_key_hiddens, edge_scores, mask)

        self.metrics(pred_heads, pred_labels, edge_heads, edge_labels, query_mask,
                     edge_head_nll.item(), edge_label_nll.item())

        return dict(
            edge_heads=pred_heads,
            edge_labels=pred_labels,
            loss=(edge_head_nll + edge_label_nll) / num_instances,
            total_loss=edge_head_nll + edge_label_nll,
            num_instances=num_instances
        )

    def _add_sentinel(self, keys: torch.FloatTensor, mask: torch.ByteTensor):
        """
        Add a sentinel at the beginning of keys.
        :param keys:  [batch_size, key_legnth, key_hidden_size]
        :param mask: None or [batch_size, query_length, key_length]
        :return:
            new_keys: [batch_size, key_length + 1, key_hidden_size]
            mask: None or [batch_size, query_length, key_length + 1]
        """
        batch_size, _, hidden_size = keys.size()
        sentinel = self.sentinel.expand([batch_size, 1, hidden_size])
        new_keys = torch.cat([sentinel, keys], dim=1)
        new_mask = None
        if mask is not None:
            query_length = mask.size(1)
            sentinel_mask = mask.new_ones(batch_size, query_length, 1)
            new_mask = torch.cat([sentinel_mask, mask], dim=2)
        return new_keys, new_mask

    def _mlp(self, queries: torch.FloatTensor, keys: torch.FloatTensor):
        """
        Transform queries and keys into spaces of edge and label.
        :param queries:  [batch_size, query_length, query_hidden_size]
        :param keys: [batch_size, key_length, key_hidden_size]
        :return:
            edge_query_hiddens: [batch_size, query_length, edge_hidden_size]
            edge_key_hiddens: [batch_size, key_length, edge_hidden_size]
            label_query_hiddens: [batch_size, query_length, label_hidden_size]
            label_key_hiddens: [batch_size, key_length, label_hidden_size]
        """
        query_length = queries.size(1)
        edge_query_hiddens = torch.nn.functional.elu(self.edge_query_linear(queries))
        edge_key_hiddens = torch.nn.functional.elu(self.edge_key_linear(keys))

        label_query_hiddens = torch.nn.functional.elu(self.label_query_linear(queries))
        label_key_hiddens = torch.nn.functional.elu(self.label_key_linear(keys))

        edge_hiddens = torch.cat([edge_query_hiddens, edge_key_hiddens], dim=1)
        label_hiddens = torch.cat([label_query_hiddens, label_key_hiddens], dim=1)
        edge_hiddens = self.dropout(edge_hiddens.transpose(1, 2)).transpose(1, 2)
        label_hiddens = self.dropout(label_hiddens.transpose(1, 2)).transpose(1, 2)

        edge_query_hiddens = edge_hiddens[:, :query_length]
        edge_key_hiddens = edge_hiddens[:, query_length:]
        label_query_hiddens = label_hiddens[:, :query_length]
        label_key_hiddens = label_hiddens[:, query_length:]

        return edge_query_hiddens, edge_key_hiddens, label_query_hiddens, label_key_hiddens

    def _get_edge_scores(self, queries: torch.FloatTensor, keys: torch.FloatTensor):
        """
        Compute the edge scores.
        :param queries:  [batch_size, query_length, hidden_size]
        :param keys:  [batch_size, key_length, hidden_size]
        :param mask:  None or [batch_size, query_length, key_length]
        :return: [batch_size, query_length, key_length]
        """
        edge_scores = self.attention(queries, keys).squeeze(1)
        return edge_scores

    def _get_label_scores(self, queries: torch.FloatTensor, keys: torch.FloatTensor, edge_heads: torch.LongTensor):
        """
        Compute the label scores.
        :param queries:  [batch_size, query_length, hidden_size]
        :param keys: [batch_size, key_length, hidden_size]
        :param edge_heads: [batch_size, query_length]
        :return:
            label_scores: [batch_size, query_length, num_labels]
        """
        batch_size = keys.size(0)
        batch_index = torch.arange(0, batch_size).view(batch_size, 1).type_as(edge_heads)
        # [batch_size, query_length, hidden_size]
        _keys = keys[batch_index, edge_heads].contiguous()
        queries = queries.contiguous()

        label_scores = self.edge_label_bilinear(queries, _keys)
        return label_scores

    def _decode(self,
                queries: torch.FloatTensor,
                keys: torch.FloatTensor,
                edge_scores: torch.FloatTensor,
                mask: torch.ByteTensor):
        """
        Predict edge heads and labels.
        :param queries: [batch_size, query_length, hidden_size]
        :param keys:  [batch_size, key_length, hidden_size]
        :param edge_scores:  [batch_size, query_length, key_length]
        :param mask:  None or [batch_size, query_length, key_length]
        :return:
            edge_heads: [batch_size, query_length]
            edge_labels: [batch_size, query_length]
        """

        edge_scores = edge_scores + (1 - mask).float() * self.minus_inf
        _, edge_heads = edge_scores.max(dim=2)

        label_scores = self._get_label_scores(queries, keys, edge_heads)
        _, edge_labels = label_scores.max(dim=2)

        return edge_heads, edge_labels

    @classmethod
    def from_params(cls, vocab, params):
        input_size = params['input_size']
        edge_hidden_size = params['edge_hidden_size']
        label_hidden_size = params['label_hidden_size']
        dropout = params['dropout']

        sentinel = torch.nn.Parameter(torch.randn([1, 1, input_size]))

        # Transform representations into spaces for edge queries and keys.
        edge_query_linear = torch.nn.Linear(input_size, edge_hidden_size)
        edge_key_linear = torch.nn.Linear(input_size, edge_hidden_size)

        # Transform representations into spaces for label queries and keys.
        label_query_linear = torch.nn.Linear(input_size, label_hidden_size)
        label_key_linear = torch.nn.Linear(input_size, label_hidden_size)

        dropout = torch.nn.Dropout2d(p=dropout)

        attention = BiaffineAttention(edge_hidden_size, edge_hidden_size)

        num_labels = vocab.get_vocab_size("head_tags")
        edge_label_bilinear = BiLinear(label_hidden_size, label_hidden_size, num_labels)

        return cls(
            sentinel=sentinel,
            edge_query_linear=edge_query_linear,
            edge_key_linear=edge_key_linear,
            label_query_linear=label_query_linear,
            label_key_linear=label_key_linear,
            dropout=dropout,
            attention=attention,
            edge_label_bilinear=edge_label_bilinear
        )