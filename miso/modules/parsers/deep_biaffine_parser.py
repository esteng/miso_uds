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

    def forward(self, inp1, inp2):
        # inp1: b x n x d1
        # inp2: b x n x d2
        # U: o x d1 x d2
        bsz, seq_len, __ = inp1.shape
        if self.add_dep_bias:
            # b x n x 1
            bias1 = torch.ones((bsz, seq_len, 1)) 
            # b x n x (d1 + 1) 
            inp1 = torch.cat([inp1, bias1], dim = 2) 
        if self.add_head_bias:
            # b x n x 1
            bias2 = torch.ones((bsz, seq_len, 1))
            # b x n x (d2 + 1) 
            inp2 = torch.cat([inp2, bias2], dim = 2)

        # b x 1 x n x d1
        inp1 = inp1.unsqueeze(1)
        # b x 1 x n x d2
        inp2 = inp2.unsqueeze(1)
        # b x o x  n x n
        out_val = inp1 @ self.U @ inp2.permute(0, 1, 3, 2) 
        return out_val 

class DeepBiaffineParser(torch.nn.Module, Registrable):
    def __init__(self,
                 label_mlp: FeedForward,
                 arc_mlp: FeedForward, 
                 n_labels: int):
        super().__init__() 

        # label parameters 
        self.label_head_mlp = copy.deepcopy(label_mlp)
        self.label_dep_mlp = copy.deepcopy(label_mlp)
        in_dim1 = self.label_dep_mlp._output_dim
        in_dim2 = self.label_head_mlp._output_dim

        self.label_bilinear = BiaffineAttn(in_dim1, in_dim2, n_labels) 

        # arc parameters
        self.arc_head_mlp = copy.deepcopy(arc_mlp)
        self.arc_dep_mlp = copy.deepcopy(arc_mlp) 
        in_dim1 = self.arc_dep_mlp._output_dim
        in_dim2 = self.arc_head_mlp._output_dim

        self.arc_bilinear = BiaffineAttn(in_dim1, in_dim2, 1, add_dep_bias = False) 

        # losses
        self.arc_criterion = torch.nn.CrossEntropyLoss()
        self.label_criterion = torch.nn.CrossEntropyLoss()

    def forward(self, encoder_reps: torch.Tensor): 
        # encoder_reps: b x n x d
        # b x n x d1
        arc_head = self.arc_head_mlp(encoder_reps) 
        # b x n x d2
        arc_dep = self.arc_dep_mlp(encoder_reps) 
        # b x n x n 
        arc_logits = self.arc_bilinear(arc_dep, arc_head)
        
        # b x n x d2
        label_head = self.label_head_mlp(encoder_reps)
        # b x n x d2 
        label_dep = self.label_dep_mlp(encoder_reps) 
        # b x n x n x d
        label_logits = self.label_bilinear(label_dep, label_head) 

        return arc_logits, label_logits 

    def test_forward(self, encoder_reps: torch.Tensor): 
        arc_logits, label_logits = self(encoder_reps)
        arc_preds = torch.argmax(arc_energy, dim = 2, keepdims=True) 
        

    def compute_loss(self, 
                        arc_logits, 
                        label_logits, 
                        gold_heads,
                        gold_labels):
        # arc_logits: b x 1 x n x n
        # label_logits: b x d x n x n 
        # gold_heads: b x n x 1
        # gold_labels: b x n x 1
        bsz, __, n_len, __ = arc_logits.shape 
        __, n_labels, __, __ = label_logits.shape

        arc_logits = arc_logits.reshape(bsz * n_len, n_len) 
        gold_heads = gold_heads.reshape(-1) 
        arc_loss = self.arc_criterion(arc_logits, gold_heads) 

        # b x n x d
        gold_head_inds = gold_heads.reshape(bsz,  1, n_len, 1)
        gold_head_inds = gold_head_inds.repeat(1, n_labels, 1, 1).long()
        chosen_label_logits = torch.gather(label_logits, 
                                          dim = 3,
                                          index = gold_head_inds)

        chosen_label_logits = chosen_label_logits.reshape(bsz * n_len, n_labels) 
        gold_labels = gold_labels.reshape(-1) 
        
        label_loss = self.label_criterion(chosen_label_logits, gold_labels) 

        return arc_loss + label_loss 

    def mst_decode(self, encoder_reps: torch.Tensor,
                         mask: torch.Tensor): 
        pass 

