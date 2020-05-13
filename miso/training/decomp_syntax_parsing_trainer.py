import logging
import math
import numpy
import subprocess
from typing import Dict, Optional, List, Tuple, Iterable
import sys
from overrides import overrides

from allennlp.training.trainer_base import TrainerBase

from miso.data.dataset_readers.decomp_parsing.decomp_with_syntax import DecompGraphWithSyntax
from miso.training.decomp_parsing_trainer import DecompTrainer 
#from miso.data.iterators.data_iterator import DecompDataIterator, DecompBasicDataIterator 
from miso.metrics.s_metric.s_metric import S, compute_s_metric

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name

@TrainerBase.register("decomp_syntax_parsing")
class DecompSyntaxTrainer(DecompTrainer):

    def __init__(self,
                 validation_data_path: str,
                 validation_prediction_path: str,
                 semantics_only: bool,
                 drop_syntax: bool,
                 include_attribute_scores: bool = False,
                 warmup_epochs: int = 0,
                 *args, **kwargs):
        super(DecompSyntaxTrainer, self).__init__(validation_data_path, 
                                                  validation_prediction_path,
                                                  semantics_only,
                                                  drop_syntax,
                                                  include_attribute_scores,
                                                  warmup_epochs,
                                                  *args, **kwargs)

    @overrides
    def _update_validation_s_score(self, pred_instances: List[Dict[str, numpy.ndarray]],
                                         true_instances):
        """Write the validation output in pkl format, and compute the S score."""
        logger.info("Computing S")

        for batch in true_instances:
            assert(len(batch) == 1)

        true_graphs = [true_inst for batch in true_instances for true_inst in batch[0]['graph'] ]
        true_sents = [true_inst for batch in true_instances for true_inst in batch[0]['src_tokens_str']]
        pred_graphs = [DecompGraphWithSyntax.from_prediction(pred_inst) for pred_inst in pred_instances]
        pred_sem_graphs, pred_syn_graphs = zip(*pred_graphs)

        ret = compute_s_metric(true_graphs, pred_sem_graphs, true_sents, 
                               self.semantics_only, 
                               self.drop_syntax, 
                               self.include_attribute_scores)

        self.model.val_s_precision = float(ret[0]) * 100
        self.model.val_s_recall = float(ret[1]) * 100
        self.model.val_s_f1 = float(ret[2]) * 100

        # TODO: add syntactic metrics 

