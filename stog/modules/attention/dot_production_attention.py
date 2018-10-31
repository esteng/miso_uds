import torch

from stog.utils.checks import check_dimensions_match


class DotProductAttention(torch.nn.Module):

    def __init__(self, decoder_hidden_size, encoder_hidden_size, attention_hidden_size=-1):
        super(DotProductAttention, self).__init__()
        self.decoder_hidden_size = decoder_hidden_size
        self.encoder_hidden_size = encoder_hidden_size
        self.attention_hidden_size = attention_hidden_size
        if attention_hidden_size == -1:
            check_dimensions_match(decoder_hidden_size, encoder_hidden_size, 'decoder hidden size', 'encoder hidden size')
            self.linear_layer = None
        else:
            check_dimensions_match(attention_hidden_size, encoder_hidden_size, 'attention hidden size', 'encoder hidden size')
            self.linear_layer = torch.nn.Linear(decoder_hidden_size, attention_hidden_size, bias=False)

    def forward(self, decoder_input, encoder_input):
        """
        :param decoder_input:  [batch, decoder_seq_length, decoder_hidden_size]
        :param encoder_input:  [batch, encoder_seq_length, encoder_hidden_size]
        :return:  [batch, decoder_seq_length, encoder_seq_length]
        """
        batch_size, decoder_seq_length, decoder_hidden_size = decoder_input.size()

        if self.linear_layer is not None:
            decoder_input = decoder_input.view(batch_size * decoder_seq_length, decoder_hidden_size)
            decoder_input = self.linear_layer(decoder_input)
            decoder_input = decoder_input.view(batch_size, decoder_seq_length, self.attention_hidden_size)

        encoder_input = encoder_input.transpose(1, 2)
        return torch.bmm(decoder_input, encoder_input)
