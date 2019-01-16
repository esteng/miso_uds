import torch

from stog.metrics.coref_metrics import CorefMetrics


class CorefScorer(torch.nn.Module):

    def __init__(self, sentinel, scorer):
        super(CorefScorer, self).__init__()
        # [1, 1, hidden_size]
        self.sentinel = sentinel
        self.scorer = scorer
        self.metrics = CorefMetrics()
        self.eps = 1e-20

    def forward(self, hiddens):
        """

        :param hiddens: [batch_size, num_tokens, hidden_size]
        """
        batch_size, num_tokens, hidden_size = hiddens.size()
        sentinel = self.sentinel.expand(batch_size, 1, hidden_size)
        hiddens = torch.cat([sentinel, hiddens], 1)
        # [batch_size, num_tokens + 1, num_tokens + 1]
        scores = self.scorer(hiddens, hiddens)
        mask = scores.new_ones(num_tokens + 1, num_tokens + 1).byte()
        mask = torch.tril(mask, diagonal=-1).unsqueeze(0)
        scores.masked_fill_(1 - mask, -float('inf'))
        # [batch_size, num_tokens, num_tokens + 1]
        probs = torch.nn.functional.softmax(scores[:, 1:], 2)
        _, predictions = probs.max(2)

        return dict(
            probs=probs,
            predictions=predictions
        )

    def compute_loss(self, probs, predictions, targets, mask):
        """

        :param probs: [batch_size, num_tokens, num_tokens + 1]
        :param predictions: [batch_size, num_tokens]
        :param targets:  [batch_size, num_tokens]
        :param mask: [batch, num_tokens]
        """
        coref_mask = targets.ne(0)
        mask = mask & coref_mask

        likelihood = probs.gather(dim=2, index=targets.unsqueeze(2)).squeeze(2)
        likelihood = likelihood + self.eps
        loss = -likelihood.log().mul(mask.float())
        num_tokens = mask.sum().item()

        pred_eq = predictions.eq(targets).mul(mask)
        num_correct_pred = pred_eq.sum().item()

        self.metrics(loss.sum().item(), num_tokens, num_correct_pred)

        if num_tokens == 0:
            loss = 0
        else:
            loss = loss.sum().div(float(num_tokens))
        return dict(loss=loss)
