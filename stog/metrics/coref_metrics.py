"""Coreference metrics"""
import math

from overrides import overrides

from stog.metrics.metric import Metric


class CorefMetrics(Metric):
    """
    Accumulator for loss statistics.
    """

    def __init__(self, loss=0, n_words=0, n_correct=0, n_corefs=0, n_correct_corefs=0):
        self.loss = loss
        self.n_words = n_words
        self.n_correct = n_correct
        self.n_corefs = n_corefs
        self.n_correct_corefs = n_correct_corefs

    def __call__(self, loss=0, n_words=0, n_correct=0, n_corefs=0, n_correct_corefs=0):
        """
        Update statistics by suming values with another `Statistics` object
        """
        self.loss += loss
        self.n_words += n_words
        self.n_correct += n_correct
        self.n_corefs += n_corefs
        self.n_correct_corefs += n_correct_corefs

    def accuracy(self):
        """ compute accuracy """
        if self.n_words == 0:
            return -1
        else:
            return 100 * (self.n_correct / self.n_words)

    def coref_accuracy(self):
        if self.n_corefs == 0:
            return -1
        else:
            return 100 * (self.n_correct_corefs / self.n_corefs)

    def get_metric(self, reset: bool = False):
        metrics = dict(
            cacc1=self.accuracy(),
            cacc2=self.coref_accuracy()
        )
        if reset:
            self.reset()
        return metrics

    @overrides
    def reset(self):
        self.loss = 0
        self.n_words = 0
        self.n_correct = 0
        self.n_corefs = 0
        self.n_correct_corefs = 0
