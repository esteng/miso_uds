
class UnlabeledAttachScore:

    def __init__(self):
        self.accumulated_uas = 0.0
        self.num_tokens = 0

    def __call__(self, pred_headers, gold_headers, mask):
        num_tokens = mask.sum().item() - mask.size(0)
        self.num_tokens += num_tokens
        self.accumulated_uas += uas(pred_headers, gold_headers, mask) * num_tokens

    def reset(self):
        self.accumulated_uas = 0.0
        self.num_tokens = 0

    @property
    def score(self):
        return self.accumulated_uas * 100 / self.num_tokens


def uas(pred_headers, gold_headers, mask):
    """
    Compute unlabelled attachment score.
    :param pred_headers: [batch, max_length]
    :param gold_headers: [batch, max_length]
    :param mask: [batch, max_length]
    :return: float
    """
    equality = (pred_headers[:, 1:] == gold_headers[:, 1:]) # exclude ROOT
    equality = equality.long() * mask[:, 1:].long().data.cpu()
    equality = equality.sum().item()
    num_tokens = mask.sum().item() - mask.size(0)
    return equality / num_tokens

