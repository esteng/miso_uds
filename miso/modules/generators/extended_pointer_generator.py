from typing import Dict, Optional

from overrides import overrides
import torch
from allennlp.common.registrable import Registrable
import logging
logger = logging.getLogger(__name__)


class ExtendedPointerGenerator(torch.nn.Module, Registrable):

    def __init__(self,
                 input_vector_dim: int,
                 vocab_size: int = 100,
                 vocab_pad_index: int = 0,
                 source_copy: bool = True,
                 target_copy: bool = True) -> None:
        super().__init__()
        self.vocab_linear = torch.nn.Linear(input_vector_dim, vocab_size)
        self.switch_linear = torch.nn.Linear(input_vector_dim, 1 + source_copy + target_copy)
        self._input_vector_dim = input_vector_dim
        self._vocab_size = vocab_size
        self._vocab_pad_index = vocab_pad_index
        self._source_copy = source_copy
        self._target_copy = target_copy
        self._eps = 1e-20

    def reset_vocab_linear(self,
                           vocab_size: int,
                           vocab_pad_index: int) -> None:
        self.vocab_linear = torch.nn.Linear(self._input_vector_dim, vocab_size)
        self._vocab_size = vocab_size
        self._vocab_pad_index = vocab_pad_index

    @overrides
    def forward(self,
                inputs: torch.Tensor,
                source_attention_weights: Optional[torch.Tensor] = None,
                source_attention_map: Optional[torch.Tensor] = None,
                target_attention_weights: Optional[torch.Tensor] = None,
                target_attention_map: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Compute a distribution over the target dictionary
        extended by the dynamic dictionary implied by copying target nodes.

        :param inputs: [batch_size, target_length, input_vector_dim]
        :param source_attention_weights: attention of each source token,
            [batch_size, target_length, source_length].
        :param source_attention_map: a sparse indicator matrix
            mapping each source token to its index in the dynamic vocabulary.
            [batch_size, source_length, source_dynamic_vocab_size]
        :param target_attention_weights: attention of each target token,
            [batch_size, target_length, target_length]
        :param target_attention_map: a sparse indicator matrix
            mapping each target token to its index in the dynamic vocabulary.
            [batch_size, target_length, target_dynamic_vocab_size]
        :return hybrid_prob_dist: [batch_size, target_length, final_vocab_size].
        """
        hybrid_prob_dist = []
        batch_size, target_length, _ = inputs.size()

        # Soft switch: [batch_size, target_length, num_switches].
        logger.info(f"switch inputs {inputs[0,0:4,0:4]}") 
        p = torch.nn.functional.softmax(self.switch_linear(inputs), dim=2)
        logger.info(f"switch linear {self.switch_linear(inputs)}" ) 
        logger.info(f"p gen, source, copy {p[0,0,:]}")
        generation_switch = p[:, :, 0].unsqueeze(2)
        source_copy_switch = p[:, :, 1].unsqueeze(2)
        target_copy_switch = p[:, :, 2].unsqueeze(2)

        # Vocab generation.
        # [batch_size, target_length, vocab_size]
        scores = self.vocab_linear(inputs)
        scores[:, :, self._vocab_pad_index] = -float('inf')
        vocab_prob_dist = torch.nn.functional.softmax(scores, dim=2)
        vocab_prob_part = torch.mul(vocab_prob_dist, generation_switch.expand_as(vocab_prob_dist))
        hybrid_prob_dist.append(vocab_prob_part)

        logger.info(f"generation hybrid_prob {torch.sum(hybrid_prob_dist[0])}") 

        # Source-side copy.
        if self._source_copy:
            # [batch_size, target_length, source_dynamic_vocab_size]
            source_copy_prob_dist = torch.bmm(source_attention_weights, source_attention_map.float())
            source_copy_prob_part = torch.mul(
                source_copy_prob_dist, source_copy_switch.expand_as(source_copy_prob_dist)
            )
            hybrid_prob_dist.append(source_copy_prob_part)
        logger.info(f"source copy hybrid_prob {torch.sum(hybrid_prob_dist[1])}") 

        # Target-side copy.
        if self._target_copy:
            # [batch_size, target_length, target_dymanic_vocab_size]
            target_copy_prob_dist = torch.bmm(target_attention_weights, target_attention_map.float())
            target_copy_prob_part = torch.mul(
                target_copy_prob_dist, target_copy_switch.expand_as(target_copy_prob_dist)
            )
            hybrid_prob_dist.append(target_copy_prob_part)
        logger.info(f"target copy hybrid_prob {torch.sum(hybrid_prob_dist[2])}") 

        return {"hybrid_prob_dist": torch.cat(hybrid_prob_dist, dim=2)}
