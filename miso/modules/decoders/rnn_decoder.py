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

    def __init__(self,
                 rnn_cell,
                 dropout,
                 attention_layer,
                 source_copy_attention_layer=None,
                 coref_attention_layer=None,
                 head_sentinels=None,
                 use_coverage=False):
        super(InputFeedRNNDecoder, self).__init__(rnn_cell, dropout)
        self.attention_layer = attention_layer
        self.source_copy_attention_layer = source_copy_attention_layer
        self.coref_attention_layer = coref_attention_layer
        self.head_sentinels = head_sentinels
        self.use_coverage = use_coverage

    def forward(self, inputs, memory_bank, mask, hidden_state, input_feed=None, heads=None):
        """
        :param inputs: [batch_size, decoder_seq_length, embedding_size]
        :param memory_bank: [batch_size, encoder_seq_length, encoder_hidden_size]
        :param mask:  None or [batch_size, decoder_seq_length]
        :param hidden_state: a tuple of (state, memory) with shape [num_encoder_layers, batch_size, encoder_hidden_size]
        :param input_feed: None or [batch_size, 1, hidden_size]
        :param heads: a list of head indices of shape [batch_size, 1]
        """
        batch_size, sequence_length, _ = inputs.size()
        # Outputs
        decoder_hidden_states = []
        rnn_hidden_states = []
        source_copy_attentions = []
        target_copy_attentions = []
        coverage_records = []

        # Internal use
        target_copy_hidden_states = []
        head_hidden_states = []
        if input_feed is None:
            input_feed = inputs.new_zeros(batch_size, 1, self.rnn_cell.hidden_size)
        coverage = None
        if self.use_coverage:
            coverage = inputs.new_zeros(batch_size, 1, memory_bank.size(1))

        for step_i, input in enumerate(inputs.split(1, dim=1)):
            coverage_records.append(coverage)

            output_dict = self.one_step_forward(
                input, memory_bank, mask, hidden_state, input_feed,
                heads, head_hidden_states, target_copy_hidden_states, coverage, step_i, sequence_length)

            hidden_state = output_dict['rnn_hidden_state']
            head_hidden_states.append(output_dict['rnn_output'])
            target_copy_hidden_states.append(output_dict['decoder_output'])

            decoder_hidden_states.append(output_dict['decoder_output'])
            rnn_hidden_states.append(output_dict['rnn_output'])
            source_copy_attentions.append(output_dict['source_copy_attention'])
            target_copy_attentions.append(output_dict['target_copy_attention'])
            input_feed = output_dict['input_feed']
            coverage = output_dict['coverage']

        decoder_hidden_states = torch.cat(decoder_hidden_states, 1)
        rnn_hidden_states = torch.cat(rnn_hidden_states, 1)
        source_copy_attentions = torch.cat(source_copy_attentions, 1)
        target_copy_attentions = torch.cat(target_copy_attentions, 1)
        if self.use_coverage and len(coverage_records):
            coverage_records = torch.cat(coverage_records, 1)
        else:
            coverage_records = None

        return dict(
            decoder_hidden_states=decoder_hidden_states,
            rnn_hidden_states=rnn_hidden_states,
            source_copy_attentions=source_copy_attentions,
            target_copy_attentions=target_copy_attentions,
            coverage_records=coverage_records
        )

    def one_step_forward(self,
                         input,
                         memory_bank,
                         mask,
                         hidden_state,
                         input_feed,
                         heads,
                         head_hidden_states,
                         target_copy_hidden_states,
                         coverage=None,
                         step_i=0,
                         target_seq_length=1):
        """
        :param inputs: [batch_size, decoder_seq_length, embedding_size]
        :param memory_bank: [batch_size, encoder_seq_length, encoder_hidden_size]
        :param mask:  None or [batch_size, decoder_seq_length]
        :param hidden_state: a tuple of (state, memory) with shape [num_encoder_layers, batch_size, encoder_hidden_size]
        :param input_feed: None or [batch_size, 1, hidden_size]
        :param heads: a list of head indices of shape [batch_size, 1]
        :param head_hidden_states: [batch_size, prev_seq_length, hidden_size]
        :param target_copy_hidden_states: [batch_size, seq_length, hidden_size]
        :param coverage: None or [batch_size, 1, encode_seq_length]
        :param step_i: int
        :param target_seq_length: int
        :return:
        """
        head_hidden = self.get_head_hidden(heads, step_i, head_hidden_states, input.size(0))
        input_feed = torch.cat([input_feed, head_hidden], dim=2)

        rnn_output, hidden_state = self.one_step_rnn_forward(input, hidden_state, input_feed)

        output, std_attention, coverage = self.attention_layer(
            rnn_output, memory_bank, mask, coverage)
        output = self.dropout(output)

        if self.source_copy_attention_layer is not None:
            _, source_copy_attention, _ = self.source_copy_attention_layer(
                output, memory_bank, mask)
        else:
            source_copy_attention = std_attention

        target_copy_attention = self.get_target_copy_attention(
            output, target_copy_hidden_states, step_i, target_seq_length)

        return dict(
            decoder_output=output,
            rnn_output=rnn_output,
            rnn_hidden_state=hidden_state,
            source_copy_attention=source_copy_attention,
            target_copy_attention=target_copy_attention,
            input_feed=output,
            coverage=coverage
        )

    def one_step_rnn_forward(self, input, hidden_state, input_feed):
        """
        :param input: [batch_size, 1, embedding_size]
        :param hidden_state: a tuple of (state, memory) with shape [num_encoder_layers, batch_size, encoder_hidden_size]
        :param input_feed: [batch_size, 1, hidden_size]
        """
        one_step = [1] * input.size(0)
        _input = torch.cat([input, input_feed], 2)
        packed_input = pack_padded_sequence(_input, one_step, batch_first=True)
        # hidden_state: a tuple of (state, memory) of shape [num_layers, batch_size, hidden_size]
        packed_output, hidden_state = self.rnn_cell(packed_input, hidden_state)
        # output: [batch_size, 1, hidden_size]
        output, _ = pad_packed_sequence(packed_output, batch_first=True)

        return output, hidden_state

    def get_head_hidden(self, heads, step_i, hidden_states, batch_size):
        if step_i < 3:
            head_hidden = self.head_sentinels.new_zeros(
                batch_size, 1, self.head_sentinels.size(2))
        else:
            head_index = heads[step_i - 3]
            batch_index = torch.arange(batch_size).view(-1, 1).type_as(head_index)
            previous_hidden_states = torch.cat(hidden_states, dim=1)
            head_hidden = previous_hidden_states[batch_index, head_index]
        return head_hidden

    def get_target_copy_attention(self, input, list_of_memories, step_i, target_seq_length):
        """
        :param input: [batch_size, 1, hidden_size]
        :param list_of_memories: a list of memory of shape [batch_size, 1, hidden_size]
        :param step_i: int
        :param target_seq_length: int
        """
        batch_size = input.size(0)

        if len(list_of_memories) == 0:
            target_copy_attention = input.new_zeros(batch_size, 1, target_seq_length)
        else:
            target_copy_memory = torch.cat(list_of_memories, 1)

            if target_seq_length == 1:
                _, target_copy_attention, _ = self.coref_attention_layer(
                    input, target_copy_memory)
            else:
                _, target_copy_attention, _ = self.coref_attention_layer(
                    input, target_copy_memory)
                target_copy_attention = torch.nn.functional.pad(
                    target_copy_attention, (0, target_seq_length - step_i), 'constant', 0
                )

        return target_copy_attention

