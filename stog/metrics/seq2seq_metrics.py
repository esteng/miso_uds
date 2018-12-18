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

    def __init__(self, loss=0, n_words=0, n_correct=0, n_copies=0, n_correct_copies=0, n_correct_binaries=0):
        self.loss = loss
        self.n_words = n_words
        self.n_correct = n_correct
        self.n_copies = n_copies
        self.n_correct_copies = n_correct_copies
        self.n_correct_binaries = n_correct_binaries

    def __call__(self, loss, n_words, n_correct, n_copies=0, n_correct_copies=0, n_correct_binaries=0):
        """
        Update statistics by suming values with another `Statistics` object
        """
        self.loss += loss
        self.n_words += n_words
        self.n_correct += n_correct
        self.n_copies += n_copies
        self.n_correct_copies += n_correct_copies
        self.n_correct_binaries += n_correct_binaries

    def accuracy(self):
        """ compute accuracy """
        return 100 * (self.n_correct / self.n_words)

    def xent(self):
        """ compute cross entropy """
        return self.loss / self.n_words

    def ppl(self):
        """ compute perplexity """
        return math.exp(min(self.loss / self.n_words, 100))

    def copy_accuracy(self):
        if self.n_copies == 0:
            return -1
        else:
            return 100 * (self.n_correct_copies / self.n_copies)

    def binary_accuracy(self):
        if self.n_copies == 0:
            return -1
        else:
            return 100 * (self.n_correct_binaries / self.n_copies)

    def get_metric(self, reset: bool = False):
        metrics = dict(
            accuracy=self.accuracy(),
            copy_acc=self.copy_accuracy(),
            bina_acc=self.binary_accuracy(),
            xent=self.xent(),
            ppl=self.ppl()
        )
        if reset:
            self.reset()
        return metrics

    @overrides
    def reset(self):
        self.loss = 0
        self.n_words = 0
        self.n_correct = 0
        self.n_copies = 0
        self.n_correct_copies = 0
        self.n_correct_binaries = 0
