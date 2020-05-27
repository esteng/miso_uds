from typing import Tuple, Dict, Optional
from overrides import overrides
import numpy as np

import torch
import torch.nn.functional as F

from allennlp.common.registrable import Registrable
from allennlp.nn.util import masked_log_softmax
from allennlp.nn.chu_liu_edmonds import decode_mst 

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

    @staticmethod
    def _run_mst_decoding(batch_energy, lengths):
        edge_heads = []
        edge_labels = []

        for i, (energy, length) in enumerate(zip(batch_energy.detach().cpu(), lengths)):
            # energy: [num_labels, max_head_length, max_modifier_length]
            # scores | label_ids : [max_head_length, max_modifier_length]
            # decode heads and labels 
            print(length) 
            instance_heads, instance_head_labels = decode_mst(energy.numpy(), length, has_labels=True)

            edge_heads.append(instance_heads)
            edge_labels.append(instance_head_labels)

        return torch.from_numpy(np.stack(edge_heads)), torch.from_numpy(np.stack(edge_labels))

    def _decode_mst(self, edge_label_h, edge_label_m, edge_node_scores, mask):
        batch_size, max_length, edge_label_hidden_size = edge_label_h.size()
        print(f"maske {mask.shape} ") 
        lengths = mask.data.sum(dim=1).long().cpu().numpy()
        print(f"lengths {lengths}") 

        edge_label_m = edge_label_m[:,1:,:]
        expanded_shape = [batch_size, max_length, max_length, edge_label_hidden_size]
        edge_label_h = edge_label_h.unsqueeze(2).expand(*expanded_shape).contiguous()
        edge_label_m = edge_label_m.unsqueeze(1).expand(*expanded_shape).contiguous()
        # [batch, max_head_length, max_modifier_length, num_labels]
        edge_label_scores = self.edge_type_bilinear(edge_label_h, edge_label_m)
        edge_label_scores = torch.nn.functional.log_softmax(edge_label_scores, dim=3).permute(0, 3, 1, 2)
        print(f"edge_label_scores {edge_label_scores.shape}") 

        # Set invalid positions to -inf
        minus_mask = (1 - mask.float()) * self._minus_inf

        edge_node_scores = edge_node_scores[:,:,1:]
        print(f"edge_node_scores {edge_node_scores.shape}") 
        edge_node_scores = edge_node_scores + minus_mask.unsqueeze(2) + minus_mask.unsqueeze(1) 

        # [batch, max_head_length, max_modifier_length]
        edge_node_scores = torch.nn.functional.log_softmax(edge_node_scores, dim=1)

        # [batch, num_labels, max_head_length, max_modifier_length]
        batch_energy = torch.exp(edge_node_scores.unsqueeze(1) + edge_label_scores)

        edge_heads, edge_labels = self._run_mst_decoding(batch_energy, lengths)
        return edge_heads, edge_labels

    @overrides
    def forward(self,
                query: torch.FloatTensor,
                key: torch.FloatTensor,
                edge_head_mask: torch.ByteTensor = None,
                gold_edge_heads: torch.Tensor = None,
                decode_mst: bool = False,
                valid_node_mask: torch.Tensor = None
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



    #@overrides
    #def forward(self,
    #            query: torch.FloatTensor,
    #            key: torch.FloatTensor,
    #            edge_head_mask: torch.ByteTensor = None,
    #            gold_edge_heads: torch.Tensor = None,
    #            decode_mst: bool = False,
    #            valid_node_mask: torch.ByteTensor = None
    #            ) -> Dict:
    #    """
    #    :param query: [batch_size, query_length, query_vector_dim]
    #    :param key: [batch_size, key_length, key_vector_dim]
    #    :param edge_head_mask: [batch_size, query_length, key_length]
    #                    1 indicates a valid position; otherwise, 0.
    #    :param gold_edge_heads: None or [batch_size, query_length].
    #                    head indices start from 1.
    #    :return:
    #        edge_heads: [batch_size, query_length].
    #        edge_types: [batch_size, query_length].
    #        edge_head_ll: [batch_size, query_length, key_length + 1(sentinel)].
    #        edge_type_ll: [batch_size, query_length, num_labels] (based on gold_edge_head) or None.
    #    """
    #    print(f"coming in {gold_edge_heads}") 
    #    print(f"do mst is {decode_mst}") 
    #    key, edge_head_mask = self._add_sentinel(query, key, edge_head_mask)
    #    edge_head_query, edge_head_key, edge_type_query, edge_type_key = self._mlp(query, key)
    #    # [batch_size, query_length, key_length + 1]
    #    edge_head_score = self._get_edge_head_score(edge_head_query, edge_head_key)
    #    if not decode_mst: 
    #        #edge_heads, edge_types = self._greedy_search(
    #        #    edge_type_query, edge_type_key, edge_head_score, edge_head_mask
    #        #)
    #        #print(f"after greedy {edge_heads}") 
    #        if gold_edge_heads is None:
    #            print(f"setting gold heads to edge heads") 
    #            gold_edge_heads = edge_heads

    #    else:
    #        edge_heads, edge_type_key = self._decode_mst(
    #            edge_type_query, edge_type_key, edge_head_score, valid_node_mask 
    #        )

    #    # [batch_size, query_length, num_labels]
    #    edge_type_score = self._get_edge_type_score(edge_type_query, edge_type_key, gold_edge_heads)

    #    edge_type_llh = masked_log_softmax(edge_type_score, None, dim=2)

    #    edge_head_llh = masked_log_softmax(edge_head_score, edge_head_mask, dim=2)

    #    return dict(
    #        # Note: head indices start from 1.
    #        edge_heads=edge_heads,
    #        edge_types=edge_types,
    #        # Log-Likelihood.
    #        edge_head_ll=edge_head_llh,
    #        edge_type_ll=edge_type_llh
    #    )

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
        print(f"edge head shape {edge_head.shape}") 
        edge_head = edge_head.byte() 
        batch_index = torch.arange(0, batch_size).view(batch_size, 1).type_as(edge_head)
        print(f"batch idx {batch_index.shape}") 
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
