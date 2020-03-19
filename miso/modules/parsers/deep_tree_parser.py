from typing import Tuple, Dict, Optional
from overrides import overrides
import torch
import torch.nn.functional as F

from allennlp.common.registrable import Registrable
from allennlp.nn.util import masked_log_softmax
from miso.modules.attention import Attention

class DeepTreeParser(torch.nn.Module, Registrable):

    def __init__(self,
                 query_vector_dim: int,
                 key_vector_dim: int,
                 edge_head_vector_dim: int,
                 edge_type_vector_dim: int,
                 attention: Attention,
                 num_labels: int = 0,
                 dropout: float = 0.0) -> None:
        super().__init__()
        self.edge_head_query_linear = torch.nn.Linear(query_vector_dim, edge_head_vector_dim)
        self.edge_head_key_linear = torch.nn.Linear(key_vector_dim, edge_head_vector_dim)
        self.edge_type_query_linear = torch.nn.Linear(query_vector_dim, edge_type_vector_dim)
        self.edge_type_key_linear = torch.nn.Linear(key_vector_dim, edge_type_vector_dim)
        self.attention = attention
        self.sentinel = torch.nn.Parameter(torch.randn([1, 1, key_vector_dim]))
        self.dropout = torch.nn.Dropout2d(p=dropout)
        if num_labels > 0:
            self.edge_type_bilinear = torch.nn.Bilinear(edge_type_vector_dim, edge_type_vector_dim, num_labels)
        else:
            self.edge_type_bilinear = None

        self._minus_inf = -1e8
        self._query_vector_dim = query_vector_dim
        self._key_vector_dim = key_vector_dim
        self._edge_type_vector_dim = edge_type_vector_dim

    def reset_edge_type_bilinear(self, num_labels: int) -> None:
        self.edge_type_bilinear = torch.nn.Bilinear(self._edge_type_vector_dim, self._edge_type_vector_dim, num_labels)

    @overrides
    def forward(self,
                query: torch.FloatTensor,
                key: torch.FloatTensor,
                edge_head_mask: torch.ByteTensor = None,
                gold_edge_heads: torch.Tensor = None
                ) -> Dict:
        """
        :param query: [batch_size, query_length, query_vector_dim]
        :param key: [batch_size, key_length, key_vector_dim]
        :param edge_head_mask: [batch_size, query_length, key_length]
                        1 indicates a valid position; otherwise, 0.
        :param gold_edge_heads: None or [batch_size, query_length].
                        head indices start from 1.
        :return:
            edge_heads: [batch_size, query_length].
            edge_types: [batch_size, query_length].
            edge_head_ll: [batch_size, query_length, key_length + 1(sentinel)].
            edge_type_ll: [batch_size, query_length, num_labels] (based on gold_edge_head) or None.
        """
        key, edge_head_mask = self._add_sentinel(query, key, edge_head_mask)
        edge_head_query, edge_head_key, edge_type_query, edge_type_key = self._mlp(query, key)
        # [batch_size, query_length, key_length + 1]
        edge_head_score = self._get_edge_head_score(edge_head_query, edge_head_key)
        edge_heads, edge_types = self._greedy_search(
            edge_type_query, edge_type_key, edge_head_score, edge_head_mask
        )

        if gold_edge_heads is None:
            gold_edge_heads = edge_heads
        # [batch_size, query_length, num_labels]
        edge_type_score = self._get_edge_type_score(edge_type_query, edge_type_key, edge_heads)

        return dict(
            # Note: head indices start from 1.
            edge_heads=edge_heads,
            edge_types=edge_types,
            # Log-Likelihood.
            edge_head_ll=masked_log_softmax(edge_head_score, edge_head_mask, dim=2),
            edge_type_ll=masked_log_softmax(edge_type_score, None, dim=2)
        )

    def _add_sentinel(self,
                      query: torch.FloatTensor,
                      key: torch.FloatTensor,
                      mask: torch.ByteTensor) -> Tuple:
        """
        Add a sentinel at the beginning of keys.
        :param query:  [batch_size, query_length, input_vector_dim]
        :param key:  [batch_size, key_length, key_vector_size]
        :param mask: None or [batch_size, query_length, key_length]
        :return:
            new_keys: [batch_size, key_length + 1, input_vector_dim]
            mask: None or [batch_size, query_length, key_length + 1]
        """
        batch_size, query_length, _ = query.size()
        if key is None:
            new_keys = self.sentinel.expand([batch_size, 1, self._key_vector_dim])
            new_mask = self.sentinel.new_ones(batch_size, query_length, 1)
            return new_keys, new_mask

        sentinel = self.sentinel.expand([batch_size, 1, self._key_vector_dim])
        new_keys = torch.cat([sentinel, key], dim=1)
        new_mask = None
        if mask is not None:
            sentinel_mask = mask.new_ones(batch_size, query_length, 1)
            new_mask = torch.cat([sentinel_mask, mask], dim=2)
        return new_keys, new_mask

    def _mlp(self,
             query: torch.FloatTensor,
             key: torch.FloatTensor) -> Tuple:
        """
        Transform query and key into spaces of edge and label.
        :param query:  [batch_size, query_length, query_vector_dim]
        :param key: [batch_size, key_length, key_vector_dim]
        :return:
            edge_head_query: [batch_size, query_length, edge_head_vector_ddim]
            edge_head_key: [batch_size, key_length, edge_head_vector_dim]
            edge_type_query: [batch_size, query_length, edge_type_vector_dim]
            edge_type_key: [batch_size, key_length, edge_type_vector_dim]
        """
        query_length = query.size(1)
        edge_head_query = F.elu(self.edge_head_query_linear(query))
        edge_head_key = F.elu(self.edge_head_key_linear(key))

        edge_type_query = F.elu(self.edge_type_query_linear(query))
        edge_type_key = F.elu(self.edge_type_key_linear(key))

        edge_head = torch.cat([edge_head_query, edge_head_key], dim=1)
        edge_type = torch.cat([edge_type_query, edge_type_key], dim=1)
        edge_head = self.dropout(edge_head.transpose(1, 2)).transpose(1, 2)
        edge_type = self.dropout(edge_type.transpose(1, 2)).transpose(1, 2)

        edge_head_query = edge_head[:, :query_length]
        edge_head_key = edge_head[:, query_length:]
        edge_type_query = edge_type[:, :query_length]
        edge_type_key = edge_type[:, query_length:]

        return edge_head_query, edge_head_key, edge_type_query, edge_type_key

    def _get_edge_head_score(self,
                             query: torch.FloatTensor,
                             key: torch.FloatTensor,
                             mask: torch.Tensor = None) -> torch.FloatTensor:
        """
        Compute the edge head scores.
        :param query:  [batch_size, query_length, query_vector_dim]
        :param key:  [batch_size, key_length, key_vector_dim]
        :param mask:  None or [batch_size, query_length, key_length]
        :return: [batch_size, query_length, key_length]
        """
        # TODO: add mask.
        edge_head_score = self.attention(query, key).squeeze(1)
        return edge_head_score

    def _get_edge_type_score(self,
                             query: torch.FloatTensor,
                             key: torch.FloatTensor,
                             edge_head: torch.Tensor) -> torch.Tensor:
        """
        Compute the edge type scores.
        :param query:  [batch_size, query_length, query_vector_dim]
        :param key: [batch_size, key_length, key_vector_dim]
        :param edge_head: [batch_size, query_length]
        :return:
            label_score: None or [batch_size, query_length, num_labels]
        """
        batch_size = key.size(0)
        batch_index = torch.arange(0, batch_size).view(batch_size, 1).type_as(edge_head)
        # [batch_size, query_length, hidden_size]
        selected_key = key[batch_index, edge_head].contiguous()
        query = query.contiguous()

        edge_type_score = self.edge_type_bilinear(query, selected_key)

        return edge_type_score

    def _greedy_search(self,
                       query: torch.FloatTensor,
                       key: torch.FloatTensor,
                       edge_head_score: torch.FloatTensor,
                       edge_head_mask: torch.ByteTensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Predict edge heads and labels.
        :param query: [batch_size, query_length, query_vector_dim]
        :param key:  [batch_size, key_length, key_vector_dim]
        :param edge_head_score:  [batch_size, query_length, key_length]
        :param edge_head_mask:  None or [batch_size, query_length, key_length]
        :return:
            edge_head: [batch_size, query_length]
            edge_type: [batch_size, query_length]
        """
        edge_head_score = edge_head_score.masked_fill_(~edge_head_mask.bool(), self._minus_inf)
        _, edge_head = edge_head_score.max(dim=2)

        edge_type_score = self._get_edge_type_score(query, key, edge_head)
        _, edge_type = edge_type_score.max(dim=2)

        return edge_head, edge_type
