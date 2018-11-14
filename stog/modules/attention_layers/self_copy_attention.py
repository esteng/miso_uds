import torch
import torch.nn.functional as F



class SelfCopyAttention(torch.nn.Module):

    def __init__(self, attention):
        super(SelfCopyAttention, self).__init__()
        self.attention = attention

    def forward(self, source, memory_bank):
        """
        Args:
          source (`FloatTensor`): query vectors `[batch x tgt_len x dim]`
          memory_bank (`FloatTensor`): source vectors `[batch x src_len x dim]`
        Returns:
          (`FloatTensor`, `FloatTensor`):
          * Computed vector `[tgt_len x batch x dim]`
          * Attention distribtutions for each query
             `[tgt_len x batch x src_len]`
        """

        # one step input
        if source.dim() == 2:
            one_step = True
            source = source.unsqueeze(1)
        else:
            one_step = False

        batch, source_l, dim = memory_bank.size()
        batch_, target_l, dim_ = source.size()

        align = self.attention(source, memory_bank)

        mask = None
        if target_l != 1:
            # Not a single step
            assert source_l == target_l
            mask = torch.ones(target_l, source_l)
            mask = torch.tril(mask, diagonal=-1)
            mask[0, 0] = 1
            mask = mask.byte().unsqueeze(0)

        if mask is not None:
            align.masked_fill_(1 - mask, -float('inf'))

        align_vectors = F.softmax(align, 2)

        if mask is not None:
            # If this is not a single step, we force the first node
            # to have no attention, so that copying won't be considered in the generator.
            align_vectors[:, 0, 0] = 0

        if one_step:
            align_vectors = align_vectors.squeeze(1)

        return align_vectors
