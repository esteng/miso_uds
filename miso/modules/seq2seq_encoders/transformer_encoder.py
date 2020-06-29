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

from miso.modules.decoders.transformer.norms import ScaleNorm

@Seq2SeqEncoder.register("prenorm_transformer_encoder")
class PreNormTransformerEncoder(StackedSelfAttentionEncoder):
    # pylint: disable=line-too-long
    def __init__(self,
                 input_dim: int,
                 hidden_dim: int,
                 projection_dim: int,
                 feedforward_hidden_dim: int,
                 num_layers: int,
                 num_attention_heads: int,
                 init_scale: int = 128,
                 use_positional_encoding: bool = True,
                 dropout_prob: float = 0.1,
                 residual_dropout_prob: float = 0.2,
                 attention_dropout_prob: float = 0.1) -> None:
        super(PreNormTransformerEncoder, self).__init__(input_dim,
                                                        hidden_dim,
                                                        projection_dim,
                                                        feedforward_hidden_dim,
                                                        num_layers,
                                                        num_attention_heads,
                                                        use_positional_encoding,
                                                        dropout_prob,
                                                        residual_dropout_prob,
                                                        attention_dropout_prob)

        self.init_scale = init_scale

        for i in range(num_layers):
            layer_norm = ScaleNorm(self._attention_layers[i].get_input_dim())
            setattr(self, f"layer_norm_{i}", layer_norm)

        # set output norm 
        setattr(self, f"layer_norm_{num_layers}",    
                ScaleNorm(self._attention_layers[-1].get_output_dim()) )

        # do scaled init 
        # initialize attention heads 
        for m in self.modules():
            if isinstance(m, torch.nn.MultiheadAttention):
                torch.nn.init.normal_(m.bias_v, mean = 0, std = self._get_std_from_tensor(self.init_scale, m.bias_v))
                torch.nn.init.normal_(m.bias_k, mean = 0, std = self._get_std_from_tensor(self.init_scale, m.bias_k))
                torch.nn.init.normal_(m.in_proj_bias, mean = 0, std = self._get_std_from_tensor(self.init_scale, m.in_proj_weight))
                torch.nn.init.normal_(m.out_proj.weight, mean = 0, std = self._get_std_from_tensor(self.init_scale, m.out_proj.weight))

                torch.nn.init.constant_(m.in_proj_bias, 0.)
                torch.nn.init.constant_(m.out_proj.bias, 0.)

            elif isinstance(m, torch.nn.Linear):
                torch.nn.init.normal_(m.weight, mean = 0, std = self._get_std_from_tensor(self.init_scale, m.weight))
                torch.nn.init.constant_(m.bias, 0.)


    @staticmethod
    def _get_std_from_tensor(init_scale, tensor):
        if len(tensor.shape) > 2:
            in_d1, in_d2, out_d = tensor.shape
            in_d = in_d1 * in_d2
        else:
            in_d, out_d = tensor.shape

        # use gain to scale as in SmallInit of https://arxiv.org/pdf/1910.05895.pdf
        return (2 / (in_d + init_scale * out_d)) ** 0.5

    @overrides
    def forward(self, inputs: torch.Tensor, mask: torch.Tensor): # pylint: disable=arguments-differ
        if self._use_positional_encoding:
            output = add_positional_features(inputs)
        else:
            output = inputs
        for i in range(len(self._attention_layers)):
            # It's necessary to use `getattr` here because the elements stored
            # in the lists are not replicated by torch.nn.parallel.replicate
            # when running on multiple GPUs. Please use `ModuleList` in new
            # code. It handles this issue transparently. We've kept `add_module`
            # (in conjunction with `getattr`) solely for backwards compatibility
            # with existing serialized models.
            attention = getattr(self, f"self_attention_{i}")
            feedforward = getattr(self, f"feedforward_{i}")
            feedforward_layer_norm = getattr(self, f"feedforward_layer_norm_{i}")
            layer_norm = getattr(self, f"layer_norm_{i}")

            # do PRENORM 
            cached_input = layer_norm(output)
            # Project output of attention encoder through a feedforward
            # network and back to the input size for the next layer.
            # shape (batch_size, timesteps, input_size)
            feedforward_output = feedforward(output)
            feedforward_output = self.dropout(feedforward_output)
            if feedforward_output.size() == cached_input.size():
                # First layer might have the wrong size for highway
                # layers, so we exclude it here.
                feedforward_output = feedforward_layer_norm(feedforward_output + cached_input)
            # shape (batch_size, sequence_length, hidden_dim)
            attention_output = attention(feedforward_output, mask)
            output = self.dropout(attention_output) + feedforward_output

        # do a final norm 
        final_norm = getattr(self, f"layer_norm_{len(self._attention_layers)}") 
        output = final_norm(output)    

        return output
