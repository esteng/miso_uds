from typing import List

from overrides import overrides
import torch
from torch.nn import Dropout

from allennlp.modules.feedforward import FeedForward
from allennlp.modules.layer_norm import LayerNorm
from allennlp.modules.seq2seq_encoders.multi_head_self_attention import MultiHeadSelfAttention
from allennlp.modules.seq2seq_encoders.seq2seq_encoder import Seq2SeqEncoder
from allennlp.modules.seq2seq_encoders import StackedSelfAttentionEncoder
from allennlp.nn.activations import Activation
from allennlp.nn.util import add_positional_features
from allennlp.common.registrable import Registrable
from allennlp.modules import InputVariationalDropout

from miso.modules.decoders.transformer.norms import ScaleNorm
from miso.modules.decoders.transformer.transformer_decoder import _get_clones
from miso.modules.seq2seq_encoders.attention_layers import MisoTransformerEncoderLayer, MisoPreNormTransformerEncoderLayer

class MisoTransformerEncoder(torch.nn.Module, Registrable):
    # pylint: disable=line-too-long
    def __init__(self, 
                    input_size: int,
                    hidden_size: int,
                    encoder_layer: MisoTransformerEncoderLayer, 
                    num_layers: int,
                    dropout=0.1):
        super(MisoTransformerEncoder, self).__init__()

        if input_size != hidden_size: 
            self.input_proj_layer = torch.nn.Linear(input_size, hidden_size)
        else:
            self.input_proj_layer = None

        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.dropout = InputVariationalDropout(dropout)

    def forward(self, src, src_mask, src_key_padding_mask):
        pass

@Seq2SeqEncoder.register("prenorm_transformer_encoder")
class PreNormTransformerEncoder(MisoTransformerEncoder):
    # pylint: disable=line-too-long
    def __init__(self, 
                    input_size: int,
                    hidden_size: int,
                    encoder_layer: MisoTransformerEncoderLayer, 
                    num_layers: int,
                    dropout=0.1):
        super(PreNormTransformerEncoder, self).__init__(input_size,
                                                        hidden_size,
                                                        encoder_layer,
                                                        num_layers,
                                                        dropout)

    @overrides
    def forward(self, inputs: torch.Tensor, mask: torch.Tensor): # pylint: disable=arguments-differ
        # always use pos features 
        if self.input_proj_layer is not None:
            output = self.input_proj_layer(inputs)
        else:
            output = inputs 
        # make mask pytorch friendly
        mask = ~mask.bool() 
        # add positional encoding 
        output = add_positional_features(output)

        # switch to pytorch batch-second format 
        output = output.permute(1, 0, 2)

        # iterate over layers of the encoder 
        # mask out pad tokens 
        for i in range(len(self.layers)):
            output, __ = self.layers[i](output, src_key_padding_mask=mask)


        output = output.permute(1, 0, 2)
        output = self.dropout(output) 

        return output
