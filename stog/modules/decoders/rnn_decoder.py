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

    def __init__(self, rnn_cell, attention_layer, self_attention_layer, dropout):
        super(InputFeedRNNDecoder, self).__init__(rnn_cell, dropout)
        self.attention_layer = attention_layer
        self.self_attention_layer = self_attention_layer

    def forward(self, inputs, memory_bank, mask, hidden_state, input_feed=None, output_sequences=None):
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
        attentions = []
        copy_attentions = []
        if output_sequences is None:
            output_sequences = []
        else:
            output_sequences = list(output_sequences.split(1, dim=1))
        pre_attn_output_seq = []
        switch_input_seq = []

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

            if self.self_attention_layer is not None:
                if step_i == 0:
                    if len(output_sequences) == 0:
                        _, _, copy_attention = self.self_attention_layer(output, None)
                        copy_attention = torch.nn.functional.pad(
                            copy_attention, (0, sequence_length - 1), 'constant', 0
                        )
                    else:
                        _, _, copy_attention = self.self_attention_layer(
                            output, torch.cat(pre_attn_output_seq, 1)
                        )
                else:
                    _, _, copy_attention = self.self_attention_layer(
                        output, torch.cat(pre_attn_output_seq, 1)
                    )
                    copy_attention = torch.nn.functional.pad(
                        copy_attention, (0, sequence_length - step_i), 'constant', 0
                    )
                copy_attentions.append(copy_attention)

            pre_attn_output_seq.append(output)

            output, concat, attention = self.attention_layer(
                output, memory_bank, mask)

            switch_input_seq.append(torch.cat([concat, _input], 2))

            output = self.dropout(output)

            input_feed = output # .clone()

            output_sequences.append(output)
            attentions.append(attention)

        output_sequences = torch.cat(output_sequences, 1)
        switch_input_seq = torch.cat(switch_input_seq, 1)
        if len(copy_attentions):
            copy_attentions = torch.cat(copy_attentions, 1)
        return output_sequences, switch_input_seq, copy_attentions, attentions, hidden_state, input_feed
