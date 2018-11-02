"""Sequence-to-sequence metrics"""
import math

from overrides import overrides

from stog.metrics.metric import Metric


class Seq2SeqMetrics(Metric):
    """
    Accumulator for loss statistics.
    Currently calculates:
    * accuracy
    * perplexity
    * elapsed time
    """

    def __init__(self, loss=0, n_words=0, n_correct=0):
        self.loss = loss
        self.n_words = n_words
        self.n_correct = n_correct

    def __call__(self, loss, n_words, n_correct):
        """
        Update statistics by suming values with another `Statistics` object
        """
        self.loss += loss
        self.n_words += n_words
        self.n_correct += n_correct

    def accuracy(self):
        """ compute accuracy """
        return 100 * (self.n_correct / self.n_words)

    def xent(self):
        """ compute cross entropy """
        return self.loss / self.n_words

    def ppl(self):
        """ compute perplexity """
        return math.exp(min(self.loss / self.n_words, 100))

    def get_metric(self, reset: bool = False):
        return dict(
            accuracy = self.accuracy(),
            xent = self.xent(),
            ppl = self.ppl()
        )

    @overrides
    def reset(self):
        self.loss = 0
        self.n_words = 0
        self.n_correct = 0
