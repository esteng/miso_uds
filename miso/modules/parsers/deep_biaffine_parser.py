from typing import Tuple, Dict, Optional
from overrides import overrides
import torch
import torch.nn.functional as F
import copy

from allennlp.common.registrable import Registrable
from allennlp.nn.util import masked_log_softmax
from allennlp.modules.feedforward import FeedForward

from miso.modules.attention import Attention


class BiaffineAttn(torch.nn.Module):
    def __init__(self, 
                 in_dim1: int, 
                 in_dim2: int, 
                 out_dim: int, 
                 add_head_bias = True, 
                 add_dep_bias = True):
        super().__init__() 

        self.add_head_bias = add_head_bias
        self.add_dep_bias = add_dep_bias
        self.U = torch.nn.Parameter(torch.Tensor(out_dim, 
                                    in_dim1 + int(add_dep_bias),
                                    in_dim2 + int(add_head_bias)))
        self._init_params() 

    def _init_params(self):
        k = self.U.data.shape[1]**2
        torch.nn.init.uniform_(self.U.data, -k, k)

    def forward(self, dep_h, head_h):
        # inp1: b x n x d1
        # inp2: b x n x d2
        # U: o x d1 x d2
        bsz, seq_len, __ = dep_h.shape
        if self.add_dep_bias:
            # b x n x 1
            bias1 = torch.ones((bsz, seq_len, 1)) 
            # b x n x (d1 + 1) 
            dep_h = torch.cat([dep_h, bias1], dim = 2) 
        if self.add_head_bias:
            # b x n x 1
            bias2 = torch.ones((bsz, seq_len, 1))
            # b x n x (d2 + 1) 
            head_h = torch.cat([head_h, bias2], dim = 2)

        # b x 1 x n x d1
        dep_h= dep_h.unsqueeze(1)
        # b x 1 x n x d2
        head_h = head_h.unsqueeze(1)
        # b x o x  n x n
        out_val = dep_h @ self.U @ head_h.permute(0, 1, 3, 2) 
        return out_val 

class DeepBiaffineParser(torch.nn.Module, Registrable):
    def __init__(self,
                 label_mlp: FeedForward,
                 arc_mlp: FeedForward, 
                 n_labels: int):
        super().__init__() 

        self._minus_inf = -1e8

        # label parameters 
        self.label_head_mlp = copy.deepcopy(label_mlp)
        self.label_dep_mlp = copy.deepcopy(label_mlp)
        in_dim1 = self.label_dep_mlp._output_dim
        in_dim2 = self.label_head_mlp._output_dim

        self.label_bilinear = BiaffineAttn(in_dim1, in_dim2, n_labels) 

        # arc parameters
        self.arc_head_mlp = copy.deepcopy(arc_mlp)
        self.arc_dep_mlp = copy.deepcopy(arc_mlp) 
        dep_dim  = self.arc_dep_mlp._output_dim
        head_dim = self.arc_head_mlp._output_dim

        self.arc_bilinear = BiaffineAttn(dep_dim, head_dim, 1, add_head_bias = False) 

        # losses
        self.arc_criterion = torch.nn.CrossEntropyLoss(ignore_index=-1, size_average=True) 
        self.label_criterion = torch.nn.CrossEntropyLoss(ignore_index=-1, size_average=True) 

    def forward(self, encoder_reps: torch.Tensor): 
        # encoder_reps: b x n x d
        # b x n x d1
        arc_dep = self.arc_dep_mlp(encoder_reps) 
        # b x n x d2
        arc_head = self.arc_head_mlp(encoder_reps) 

        # b x 1 x n x n 
        arc_logits = self.arc_bilinear(arc_dep, arc_head)
        arc_logits = arc_logits.squeeze(1)
        
        # b x n x d2
        label_head = self.label_head_mlp(encoder_reps)
        # b x n x d2 
        label_dep = self.label_dep_mlp(encoder_reps) 
        # b x n x n x d
        label_logits = self.label_bilinear(label_dep, label_head) 

        return arc_logits, label_logits 

    def _greedy_search(self,
                      arc_logits, 
                      label_logits,
                      head_mask):
        bsz, n_len, __ = arc_logits.shape 
        __, n_labels, __, __ = label_logits.shape
        arc_logits = arc_logits.reshape(bsz, n_len, n_len) 
        head_mask = head_mask.unsqueeze(-1) 
        edge_head_score = arc_logits.masked_fill_(head_mask, self._minus_inf) 
        _, edge_head = edge_head_score.max(dim=2)

        return edge_head, __ 


    def compute_loss(self, 
                        arc_logits, 
                        label_logits, 
                        gold_heads,
                        gold_labels):
        # arc_logits: b x n x n 
        bsz, n_len, __ = arc_logits.shape 
        # label_logits: b x d x n x n 
        bsz, n_labels, __, __ = label_logits.shape 
        # gold_heads: b x n x 1  -> bxn 
        neg_mask = gold_heads.eq(0).unsqueeze(-1) 
        # gold_labels: b x n x 1
        #arc_logits = arc_logits.masked_fill_(neg_mask, self._minus_inf) 
        arc_logits = arc_logits.reshape(bsz * n_len, n_len) 

        gold_heads_masked = gold_heads.reshape(-1) + -1 * neg_mask.reshape(-1) 

        arc_loss = self.arc_criterion(arc_logits, gold_heads_masked) 

        # b x n x d
        gold_head_inds = gold_heads.reshape(bsz,  1, n_len, 1)
        gold_head_inds = gold_head_inds.repeat(1, n_labels, 1, 1).long()

        chosen_label_logits = torch.gather(label_logits, 
                                          dim = -1, 
                                          index = gold_head_inds)
        ##print("labels: {chosen_label_logits}") 
        neg_mask = neg_mask.unsqueeze(1) 
        #chosen_label_logits = chosen_label_logits.masked_fill_(neg_mask, self._minus_inf)

        chosen_label_logits = chosen_label_logits.reshape(bsz * n_len, n_labels) 
        #__, pred_labels = chosen_label_logits.max(dim=1)
        gold_labels = gold_labels.reshape(-1) 



        # mask out invalid positions 
        gold_labels_masked = gold_labels +  -1 * neg_mask.reshape(-1) 
        __, pred_labels = torch.max(chosen_label_logits, dim=-1) 
        print(f"gold_labels {gold_labels_masked}") 
        print(f"pred labels {pred_labels}") 

        label_loss = self.label_criterion(chosen_label_logits, gold_labels_masked) 

        #print(f"arc loss {arc_loss}") 
        #print(f"label_loss {label_loss}") 
        return arc_loss + label_loss

    def mst_decode(self, encoder_reps: torch.Tensor,
                         mask: torch.Tensor): 
        pass 

