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
from miso.modules.decoders.decoder import MisoDecoder
from miso.modules.decoders.transformer.attention_layers import MisoTransformerDecoderLayer

logger = logging.getLogger(__name__) 

def _get_clones(module, N):
    return torch.nn.ModuleList([copy.deepcopy(module) for i in range(N)])

@MisoDecoder.register("transformer_decoder") 
class MisoTransformerDecoder(MisoDecoder):
    def __init__(self, 
                    input_size: int,
                    hidden_size: int,
                    decoder_layer: MisoTransformerDecoderLayer, 
                    num_layers: int,
                    source_attention_layer: AttentionLayer,
                    target_attention_layer: AttentionLayer,
                    norm=None,
                    dropout=0.1,
                    use_coverage=True):
        super(MisoTransformerDecoder, self).__init__()

        if input_size != hidden_size: 
            self.input_proj_layer = torch.nn.Linear(input_size, hidden_size)
        else:
            self.input_proj_layer = None

        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm
        self.dropout = InputVariationalDropout(dropout)
        self.source_attn_layer = source_attention_layer
        self.target_attn_layer = target_attention_layer
        self.use_coverage = use_coverage

        self.source_dedicated_head = torch.nn.MultiheadAttention(hidden_size, 1, dropout=dropout, add_bias_kv = False)
        self.target_dedicated_head = torch.nn.MultiheadAttention(hidden_size, 1, dropout=dropout, add_bias_kv = False)


        for m in [self.source_dedicated_head, self.target_dedicated_head]:
            torch.nn.init.normal_(m.in_proj_bias, mean = 0, std = MisoTransformerDecoderLayer._get_std_from_tensor(decoder_layer.init_scale, m.in_proj_weight))
            torch.nn.init.normal_(m.out_proj.weight, mean = 0, std = MisoTransformerDecoderLayer._get_std_from_tensor(decoder_layer.init_scale, m.out_proj.weight))

            torch.nn.init.constant_(m.in_proj_bias, 0.)
            torch.nn.init.constant_(m.out_proj.bias, 0.)

    @overrides
    def forward(self,
                inputs: torch.Tensor,
                source_memory_bank: torch.Tensor,
                source_mask: torch.Tensor,
                target_mask: torch.Tensor) -> Dict: 

        batch_size, source_seq_length, _ = source_memory_bank.size()
        __, target_seq_length, __ = inputs.size()


        source_padding_mask = None
        target_padding_mask  = None
        if source_mask is not None:
            source_padding_mask = ~source_mask.bool()
        if target_mask is not None:
            target_padding_mask = ~target_mask.bool() 

        # project to correct dimensionality 
        if self.input_proj_layer is not None:
            outputs = self.input_proj_layer(inputs)
        else:
            outputs = inputs 

        # add pos encoding feats 
        outputs = add_positional_features(outputs) 

        # swap to pytorch's batch-second convention 
        outputs = outputs.permute(1, 0, 2)
        source_memory_bank = source_memory_bank.permute(1, 0, 2)


        # get a mask 
        ar_mask = self.make_autoregressive_mask(outputs.shape[0]).to(source_memory_bank.device)

        for i in range(len(self.layers)):

            outputs , __, __ = self.layers[i](outputs, 
                                    source_memory_bank, 
                                    tgt_mask=ar_mask,
                                    memory_mask=None,
                                    tgt_key_padding_mask=target_padding_mask,
                                    memory_key_padding_mask=source_padding_mask
                                    )

        # switch back from pytorch's absolutely moronic batch-second convention
        #outputs = outputs.permute(1, 0, 2)
        #source_memory_bank = source_memory_bank.permute(1, 0, 2) 
        
        # source-side attention
        if not self.use_coverage:
            outputs = outputs.permute(1, 0, 2)
            source_memory_bank = source_memory_bank.permute(1, 0, 2) 

            source_attention_output = self.source_attn_layer(outputs, 
                                                             source_memory_bank,
                                                             source_mask,
                                                             None)
            attentional_tensors = self.dropout(source_attention_output['attentional'])
            source_attention_weights = source_attention_output['attention_weights']
            coverage_history = None

            attentional_tensors = attentional_tensors.permute(1,0,2)
            outputs = outputs.permute(1, 0, 2)
            source_memory_bank = source_memory_bank.permute(1, 0, 2) 

        # try Pytorch implementation DEBUGGING 
        #if not self.use_coverage:
        #    attentional_tensors, source_attention_weights = self.source_dedicated_head(outputs, source_memory_bank, source_memory_bank, key_padding_mask = source_padding_mask) 

        #    coverage_history = None
        #    attentional_tensors = self.dropout(attentional_tensors) 
        #    #attentional_tensors = attentional_tensors.permute(1,0,2) 

        else:
            # need to do step by step because running sum of coverage
            outputs = outputs.permute(1, 0, 2)
            source_memory_bank = source_memory_bank.permute(1, 0, 2) 
            source_attention_weights = []
            attentional_tensors = []

            ## init to zeros 
            coverage = inputs.new_zeros(size=(batch_size, 1, source_seq_length))
            coverage_history = []

            for timestep in range(outputs.shape[1]):
            #for timestep in range(outputs.shape[0]):
                #output = outputs[timestep,:,:].unsqueeze(0) 
                output = outputs[:,timestep,:].unsqueeze(1)
                #attentional_tensor, source_attention_weight = self.source_dedicated_head(output, source_memory_bank, source_memory_bank, key_padding_mask = source_padding_mask)
                source_attention_output = self.source_attn_layer(output,
                                                                 source_memory_bank,
                                                                 source_mask,
                                                                 coverage)
                attentional_tensor = self.dropout(source_attention_output['attentional'])
                #attentional_tensor = self.dropout(attentional_tensor)
                attentional_tensors.append(attentional_tensor)

                source_attention_weights.append(source_attention_output['attention_weights'])
                #source_attention_weights.append(source_attention_weight) 
                coverage = source_attention_output['coverage'] 
                #coverage = torch.sum(torch.cat(source_attention_weights, dim=1),dim=1).unsqueeze(1) 
                coverage_history.append(coverage) 

            ## [batch_size, tgt_seq_len, hidden_dim]
            attentional_tensors = torch.cat(attentional_tensors, dim=1) 
            ## [batch_size, tgt_seq_len, src_seq_len]
            source_attention_weights = torch.cat(source_attention_weights, dim=1) 
            coverage_history = torch.cat(coverage_history, dim=1)

            outputs = outputs.permute(1, 0, 2)
            source_memory_bank = source_memory_bank.permute(1, 0, 2) 
            attentional_tensors = attentional_tensors.permute(1, 0, 2) 

        __, target_attention_weights = self.target_dedicated_head(attentional_tensors, 
                                                                  outputs, 
                                                                  outputs, 
                                                                  attn_mask = ar_mask,
                                                                  key_padding_mask = target_padding_mask)

        attentional_tensors = attentional_tensors.permute(1, 0, 2)

        # switch back from pytorch's absolutely moronic batch-second convention
        outputs = outputs.permute(1, 0, 2)
        source_memory_bank = source_memory_bank.permute(1, 0, 2) 

        #target_attention_output = self.target_attn_layer(attentional_tensors,
        #                                                 outputs) 

        #target_attention_weights = target_attention_output['attention_weights']

        return dict(
                outputs=outputs,
                output=outputs[:,-1,:].unsqueeze(1),
                attentional_tensors=attentional_tensors,
                attentional_tensor=attentional_tensors[:,-1,:].unsqueeze(1),
                target_attention_weights = target_attention_weights,
                source_attention_weights = source_attention_weights,
                coverage_history = coverage_history,
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
        # don't look at last position
        #target_mask = torch.ones((bsz, og_seq_len + 1))
        #target_mask[:, -1] = 0
        #target_mask = ~target_mask.bool()

        target_mask = None  
        to_ret = self(inputs, source_memory_bank, source_mask, target_mask)
        if to_ret['coverage_history'] is not None:
            to_ret["coverage"] = to_ret["coverage_history"][:,-1].unsqueeze(-1)
        else:
            to_ret['coverage'] = None

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
        mask = (torch.triu(torch.ones(size, size)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))

        return mask

