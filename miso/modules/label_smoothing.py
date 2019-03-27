import torch


class LabelSmoothing(torch.nn.Module):
    "Implement label smoothing."

    def __init__(self, padding_idx=0, smoothing=0.0):
        super(LabelSmoothing, self).__init__()
        self.criterion = torch.nn.KLDivLoss(reduction='sum')
        self.padding_idx = padding_idx
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing

    def forward(self, x, target):
        """
        :param x: log-probs [num_instances, vocab_size]
        :param target: [num_instances]
        """
        vocab_size = x.size(1)
        true_dist = torch.zeros_like(x)
        true_dist.fill_(self.smoothing / (vocab_size - 2))
        true_dist.scatter_(1, target.unsqueeze(1), self.confidence)
        true_dist[:, self.padding_idx] = 0
        mask = target.eq(self.padding_idx)
        true_dist.masked_fill_(mask.unsqueeze(1), 0.0)
        return self.criterion(x, true_dist)