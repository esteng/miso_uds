import copy

import torch
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class RNNDecoderBase(torch.nn.Module):

    def __init__(self, rnn_cell, dropout):
        super(RNNDecoderBase, self).__init__()
        self.rnn_cell = rnn_cell
        self.dropout = dropout

    def forward(self, *input):
        raise NotImplementedError


class InputFeedRNNDecoder(RNNDecoderBase):

    def __init__(self, rnn_cell, attention_layer, coref_attention_layer, dropout):
        super(InputFeedRNNDecoder, self).__init__(rnn_cell, dropout)
        self.attention_layer = attention_layer
        self.copy_attention_layer = None  # copy.deepcopy(attention_layer)
        self.coref_attention_layer = coref_attention_layer

    def forward(self, inputs, memory_bank, mask, hidden_state,
                input_feed=None, coref_inputs=None):
        """

        :param inputs: [batch_size, decoder_seq_length, embedding_size]
        :param memory_bank: [batch_size, encoder_seq_length, encoder_hidden_size]
        :param mask:  None or [batch_size, decoder_seq_length]
        :param hidden_state: a tuple of (state, memory) with shape [num_encoder_layers, batch_size, encoder_hidden_size]
        :param input_feed: None or [batch_size, 1, hidden_size]
        :return:
        """
        batch_size, sequence_length, _ = inputs.size()
        one_step_length = [1] * batch_size
        std_attentions = []
        copy_attentions = []
        coref_attentions = []
        output_sequences = []
        rnn_output_sequences = []

        if coref_inputs is None:
            coref_inputs = []
        else:
            coref_inputs = list(coref_inputs.split(1, dim=1))

        if input_feed is None:
            input_feed = inputs.new_zeros(batch_size, 1, self.rnn_cell.hidden_size)

        for step_i, input in enumerate(inputs.split(1, dim=1)):
            # input: [batch_size, 1, embeddings_size]
            # input_feed: [batch_size, 1, hidden_size]
            _input = torch.cat([input, input_feed], 2)
            packed_input = pack_padded_sequence(_input, one_step_length, batch_first=True)
            # hidden_state: a tuple of (state, memory) with shape [num_layers, batch_size, hidden_size]
            packed_output, hidden_state = self.rnn_cell(packed_input, hidden_state)
            # output: [batch_size, 1, hidden_size]
            output, _ = pad_packed_sequence(packed_output, batch_first=True)
            rnn_output_sequences.append(output)
            coref_input = output  # clone()
            output, std_attention = self.attention_layer(
                output, memory_bank, mask)
            output = self.dropout(output)
            input_feed = output

            if self.copy_attention_layer is not None:
                _, copy_attention = self.copy_attention_layer(
                    output, memory_bank, mask)
                copy_attentions.append(copy_attention)
            else:
                copy_attentions.append(std_attention)

            if self.coref_attention_layer is not None:
                if len(coref_inputs) == 0:
                    coref_attention = inputs.new_zeros(batch_size, 1, sequence_length)

                else:
                    coref_mem_bank = torch.cat(coref_inputs, 1)

                    if sequence_length == 1:
                        _, coref_attention = self.coref_attention_layer(
                            coref_input, coref_mem_bank)
                    else:
                        _, coref_attention = self.coref_attention_layer(
                            coref_input, coref_mem_bank)
                        coref_attention = torch.nn.functional.pad(
                            coref_attention, (0, sequence_length - step_i), 'constant', 0
                        )

                coref_attentions.append(coref_attention)

            coref_inputs.append(coref_input)
            output_sequences.append(output)
            std_attentions.append(std_attention)

        output_sequences = torch.cat(output_sequences, 1)
        rnn_output_sequences = torch.cat(rnn_output_sequences, 1)
        coref_inputs = torch.cat(coref_inputs, 1)
        if len(copy_attentions):
            copy_attentions = torch.cat(copy_attentions, 1)
        if len(coref_attentions):
            coref_attentions = torch.cat(coref_attentions, 1)
        return dict(
            output_sequences=output_sequences,
            rnn_output_sequences=rnn_output_sequences,
            coref_inputs=coref_inputs,
            std_attentions=std_attentions,
            copy_attentions=copy_attentions,
            coref_attentions=coref_attentions,
            hidden_state=hidden_state,
            input_feed=input_feed
        )
