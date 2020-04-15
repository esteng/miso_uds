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

from miso.modules.attention_layers import AttentionLayer

logger = logging.getLogger(__name__) 

def _get_clones(module, N):
    return torch.nn.ModuleList([copy.deepcopy(module) for i in range(N)])


def _get_activation_fn(activation):
    if activation == "relu":
        return F.relu
    elif activation == "gelu":
        return F.gelu
    else:
        raise RuntimeError("activation should be relu/gelu, not %s." % activation)

class MisoTransformerDecoderLayer(torch.nn.Module):
    """
    Modified TransformerDecoderLayer that returns attentions 
    Args:
        d_model: the number of expected features in the input (required).
        nhead: the number of heads in the multiheadattention models (required).
        dim_feedforward: the dimension of the feedforward network model (default=2048).
        dropout: the dropout value (default=0.1).
        activation: the activation function of intermediate layer, relu or gelu (default=relu).
    """

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation="relu"):
        super(MisoTransformerDecoderLayer, self).__init__()
        self.self_attn = torch.nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.multihead_attn = torch.nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = torch.nn.Linear(d_model, dim_feedforward)
        self.dropout = torch.nn.Dropout(dropout)
        self.linear2 = torch.nn.Linear(dim_feedforward, d_model)

        self.norm1 = torch.nn.LayerNorm(d_model)
        self.norm2 = torch.nn.LayerNorm(d_model)
        self.norm3 = torch.nn.LayerNorm(d_model)
        self.dropout1 = torch.nn.Dropout(dropout)
        self.dropout2 = torch.nn.Dropout(dropout)
        self.dropout3 = torch.nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)

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
        tgt2, tgt_attn = self.self_attn(tgt, tgt, tgt, attn_mask=tgt_mask,
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


class MisoTransformerDecoder(torch.nn.Module, Registrable):
    def __init__(self, 
                    decoder_layer: MisoTransformerDecoderLayer, 
                    num_layers: int,
                    input_proj_layer: torch.nn.Linear,
                    source_attn_layer: AttentionLayer,
                    target_attn_layer: AttentionLayer,
                    norm=None,
                    dropout=0.1):
        super(MisoTransformerDecoder, self).__init__()

        self.input_proj_layer = input_proj_layer
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm
        self.dropout = InputVariationalDropout(dropout)
        self.source_attn_layer = source_attn_layer
        self.target_attn_layer = target_attn_layer

    @overrides
    def forward(self,
                inputs: torch.Tensor,
                source_memory_bank: torch.Tensor,
                source_mask: torch.Tensor,
                target_mask: torch.Tensor) -> Dict: 

        inputs = inputs.permute(1, 0, 2)
        source_memory_bank = source_memory_bank.permute(1, 0, 2)
        if source_mask is not None:
            source_mask = ~source_mask.bool()
        if target_mask is not None:
            target_mask = ~target_mask.bool() 

        # project to correct dimensionality 
        outputs_projected = self.input_proj_layer(inputs)
        outputs = add_positional_features(outputs_projected) 
        ar_mask = self.make_autoregressive_mask(inputs.size()[0])

        tgt_attns = []
        src_attns = []

        for i in range(len(self.layers)):
            outputs, tgt_attn, src_attn  = self.layers[i](outputs, 
                                    source_memory_bank, 
                                    tgt_mask=ar_mask,
                                    #memory_mask=None,
                                    tgt_key_padding_mask=target_mask,
                                    memory_key_padding_mask=source_mask
                                    )
            tgt_attns.append(tgt_attn)
            src_attns.append(src_attn)

        if self.norm:
            outputs = self.norm(outputs)

        # switch back from pytorch's absolutely moronic batch-second convention
        outputs = outputs.permute(1, 0, 2)
        outputs_projected = outputs_projected.permute(1, 0, 2)
        source_memory_bank = source_memory_bank.permute(1, 0, 2) 

        # coverage implemented as in https://web.stanford.edu/class/archive/cs/cs224n/cs224n.1194/reports/custom/15784595.pdf
        # sum across target tokens 
        coverage = torch.sum(src_attns[-1], dim = 2) 

        logger.info(f"source attn inputs {outputs.shape} {source_memory_bank.shape}") 
        source_attention_output = self.source_attn_layer(outputs, 
                                                         source_memory_bank,
                                                         source_mask,
                                                         True)

        attentional_tensors = source_attention_output['attentional']

        target_attention_output = self.target_attn_layer(attentional_tensors,
                                                         outputs_projected) 

        source_attention_weights = source_attention_output['attention_weights']

        target_attention_weights = target_attention_output['attention_weights']


        return dict(
                outputs=outputs,
                output=outputs[:,-1,:].unsqueeze(1),
                attentional_tensors=attentional_tensors,
                attentional_tensor=attentional_tensors[:,-1,:].unsqueeze(1),
                target_attention_weights = target_attention_weights,
                source_attention_weights = source_attention_weights,
                coverage_history = coverage,
                ) 
                

    def one_step_forward(self,
                         inputs: torch.Tensor,
                         source_memory_bank: torch.Tensor,
                         source_mask: torch.Tensor,
                         decoding_step: int = 0,
                         total_decoding_steps: int = 0,
                         coverage: Optional[torch.Tensor] = None) -> Dict:
        """
        Run a single step decoding.
        :param input_tensor: [batch_size, seq_len, input_vector_dim].
        :param source_memory_bank: [batch_size, source_seq_length, source_vector_dim].
        :param source_mask: [batch_size, source_seq_length].
        :param decoding_step: index of the current decoding step.
        :param total_decoding_steps: the total number of decoding steps.
        :param coverage: [batch_size, 1, source_seq_length].
        :return:
        """
        bsz, og_seq_len, input_dim = inputs.size() 
        new_input = torch.zeros((bsz, 1, input_dim))
        inputs = torch.cat((inputs, new_input), dim = 1)
        # don't look at last pos 
        target_mask = torch.ones((bsz, og_seq_len + 1))
        target_mask[:, -1] = 0
        target_mask = target_mask.bool()
        to_ret = self(inputs, source_memory_bank, source_mask, target_mask)
        to_ret["coverage"] = to_ret["coverage_history"][:,-1].unsqueeze(-1)
        # pad attention weights
        tgt_attn_weights = to_ret['target_attention_weights'][:, -1, :].unsqueeze(1)
        num_done = tgt_attn_weights.shape[2]

        if total_decoding_steps != 1:
            tgt_attn_weights = F.pad(tgt_attn_weights,
                                       [0, total_decoding_steps - num_done ], 
                                      "constant", 0)

        to_ret['target_attention_weights'] = tgt_attn_weights

        to_ret['source_attention_weights'] = to_ret['source_attention_weights'][:,-1,:].unsqueeze(1)

        return to_ret 

    def make_autoregressive_mask(self,
                                 size: int):
        mask = torch.triu(torch.ones((size, size)), diagonal=1).T
        mask = (torch.ones_like(mask) - mask) * float('-inf')
        # replace nans 
        mask[mask != mask] = 0.0
        mask[0,0] = 0.0
        return mask

    @classmethod
    def from_params(cls,
                    params): 

        input_size = params['input_size']
        hidden_size = params['hidden_size']
        ff_size = params['ff_size']
        nhead = params.get('nhead', 8)
        num_layers = params.get('num_layers', 6) 
        dropout = params.get('dropout', 0.1)
        # norm = params.get('norm', 'true')

        # TODO: fix this 
        norm = None
        
        input_dropout = params.get('input_dropout', 0.1)

        input_projection_layer = torch.nn.Linear(input_size, hidden_size)

        transformer_layer = MisoTransformerDecoderLayer(hidden_size, 
                                                        nhead,
                                                        ff_size,
                                                        dropout)
        source_attention_layer = AttentionLayer.from_params(params['source_attention_layer'])
        target_attention_layer = AttentionLayer.from_params(params['target_attention_layer'])
        return cls(transformer_layer, 
                     num_layers, 
                     input_projection_layer, 
                     source_attention_layer,
                     target_attention_layer,
                     norm, dropout)


