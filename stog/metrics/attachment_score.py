
def uas(pred_headers, gold_headers, mask):
    """
    Compute unlabelled attachment score.
    :param pred_headers: [batch, max_length]
    :param gold_headers: [batch, max_length]
    :param mask: [batch, max_length]
    :return: float
    """
    equality = (pred_headers[:, 1:] == gold_headers[:, 1:]) # exclude ROOT
    equality = equality.long() * mask
    num_tokens = mask.sum().item() - mask.size(0)
    return 100.0 * equality.sum().item() / num_tokens

