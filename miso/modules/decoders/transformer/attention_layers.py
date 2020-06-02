from typing import Tuple, Dict, Optional
from overrides import overrides
import sys
import logging
import copy 
import numpy as np

import torch
import torch.nn.functional as F

from allennlp.common.registrable import Registrable
from allennlp.modules import InputVariationalDropout
from allennlp.nn.util import add_positional_features

from miso.modules.decoders.transformer.norms import Norm 

def _get_activation_fn(activation):
    if activation == "relu":
        return F.relu
    elif activation == "gelu":
        return F.gelu
    else:
        raise RuntimeError("activation should be relu/gelu, not %s." % activation)

class MisoTransformerDecoderLayer(torch.nn.Module, Registrable):
    """
    Modified TransformerDecoderLayer that returns attentions 
    Args:
        d_model: the number of expected features in the input (required).
        nhead: the number of heads in the multiheadattention models (required).
        dim_feedforward: the dimension of the feedforward network model (default=2048).
        dropout: the dropout value (default=0.1).
        activation: the activation function of intermediate layer, relu or gelu (default=relu).
    """

    def __init__(self, 
                d_model, 
                nhead, 
                norm: Norm,
                dim_feedforward=2048,
                dropout=0.1, 
                activation="relu",
                init_scale = 256):

        super(MisoTransformerDecoderLayer, self).__init__()
        self.init_scale = init_scale

        self.self_attn = torch.nn.MultiheadAttention(d_model, nhead, dropout=dropout, add_bias_kv = True)
        self.multihead_attn = torch.nn.MultiheadAttention(d_model, nhead, dropout=dropout, add_bias_kv = True)
        # Implementation of Feedforward model
        self.linear1 = torch.nn.Linear(d_model, dim_feedforward)
        self.dropout = torch.nn.Dropout(dropout)
        self.linear2 = torch.nn.Linear(dim_feedforward, d_model)

        self.norm1 = copy.deepcopy(norm)
        self.norm2 = copy.deepcopy(norm)
        self.norm3 = copy.deepcopy(norm)
        self.norm4 = copy.deepcopy(norm)

        self.dropout1 = torch.nn.Dropout(dropout)
        self.dropout2 = torch.nn.Dropout(dropout)
        self.dropout3 = torch.nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)

        # initialize attention heads 
        for m in self.modules():
            if isinstance(m, torch.nn.MultiheadAttention):

                torch.nn.init.xavier_normal_(m.bias_v, 
                             gain = self._get_gain_from_tensor(self.init_scale, 
                                                              m.bias_v) )
                torch.nn.init.xavier_normal_(m.bias_k,
                             gain = self._get_gain_from_tensor(self.init_scale, 
                                                                m.bias_k))
                torch.nn.init.xavier_normal_(m.in_proj_weight,
                            gain = self._get_gain_from_tensor(self.init_scale, 
                                                            m.in_proj_weight))
                torch.nn.init.uniform_(m.in_proj_bias)

                torch.nn.init.xavier_normal_(m.out_proj.weight, 
                            gain = self._get_gain_from_tensor(self.init_scale, 
                                                            m.out_proj.weight))
                torch.nn.init.uniform_(m.out_proj.bias)

    @staticmethod
    def _get_gain_from_tensor(init_scale, tensor):
        if len(tensor.shape) > 2:
            in_d1, in_d2, out_d = tensor.shape
            in_d = in_d1 * in_d2
        else:
            in_d, out_d = tensor.shape

        # use gain to scale as in SmallInit of https://arxiv.org/pdf/1910.05895.pdf
        return ((in_d + out_d)/(in_d + init_scale * out_d))**(1/2) 
            
    def forward(self, tgt, memory):
        pass 

@MisoTransformerDecoderLayer.register("pre_norm") 
class MisoPreNormTransformerDecoderLayer(MisoTransformerDecoderLayer):
    def __init__(self, 
                d_model, 
                n_head, 
                norm: Norm, 
                dim_feedforward=2048,
                dropout=0.1, 
                activation="relu",
                init_scale = 256):
        super(MisoPreNormTransformerDecoderLayer, self).__init__(d_model, 
                                                                 n_head, 
                                                                 norm, 
                                                                 dim_feedforward,
                                                                 dropout, 
                                                                 activation,
                                                                 init_scale)

    @overrides  
    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None,
                tgt_key_padding_mask=None, memory_key_padding_mask=None):
        r"""Pass the inputs (and mask) through the decoder layer.

        Args:
            tgt: the sequence to the decoder layer (required).
            memory: the sequnce from the last layer of the encoder (required).
            tgt_mask: the mask for the tgt sequence (optional).
            memory_mask: the mask for the memory sequence (optional).
            tgt_key_padding_mask: the mask for the tgt keys per batch (optional).
            memory_key_padding_mask: the mask for the memory keys per batch (optional).

        Shape:
            see the docs in Transformer class.
        """

        # norm before residual as in https://arxiv.org/pdf/1910.05895.pdf
        tgt2 = tgt.clone()
        tgt2 = self.norm1(tgt2)
        tgt2, tgt_attn = self.self_attn(tgt2, tgt2, tgt2, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)

        tgt = tgt + self.dropout1(tgt2)
        #tgt = self.norm1(tgt)

        tgt = self.norm2(tgt)
        tgt2, src_attn = self.multihead_attn(tgt, memory, memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)

        tgt = tgt + self.dropout2(tgt2)
        #tgt = self.norm2(tgt)

        tgt = self.norm3(tgt)
        tgt2 = self.linear2(self.dropout(F.relu(self.linear1(tgt))))

        tgt = tgt + self.dropout3(tgt2)
        #tgt = self.norm3(tgt)

        # additional norm 
        tgt = self.norm4(tgt)

        return tgt, tgt_attn, src_attn

@MisoTransformerDecoderLayer.register("post_norm") 
class MisoPostNormTransformerDecoderLayer(MisoTransformerDecoderLayer): 
    """
    Does PostNorm rather than PreNorm 
    """

    def __init__(self, 
                d_model, 
                n_head, 
                norm: Norm, 
                dim_feedforward=2048,
                dropout=0.1, 
                activation="relu",
                init_scale = 256):
        super(MisoPostNormTransformerDecoderLayer, self).__init__(d_model, 
                                                                 n_head, 
                                                                 norm, 
                                                                 dim_feedforward,
                                                                 dropout, 
                                                                 activation,
                                                                 init_scale)

    @overrides
    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None,
                tgt_key_padding_mask=None, memory_key_padding_mask=None):
        r"""Pass the inputs (and mask) through the decoder layer.

        Args:
            tgt: the sequence to the decoder layer (required).
            memory: the sequnce from the last layer of the encoder (required).
            tgt_mask: the mask for the tgt sequence (optional).
            memory_mask: the mask for the memory sequence (optional).
            tgt_key_padding_mask: the mask for the tgt keys per batch (optional).
            memory_key_padding_mask: the mask for the memory keys per batch (optional).
        """
        tgt2 = tgt.clone()
        tgt2, tgt_attn = self.self_attn(tgt2, tgt2, tgt2, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)

        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)

        tgt2, src_attn = self.multihead_attn(tgt, memory, memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)

        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)

        tgt2 = self.linear2(self.dropout(F.relu(self.linear1(tgt))))

        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)

        return tgt, tgt_attn, src_attn
