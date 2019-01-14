"""
Modified from AllenNLP:
    https://raw.githubusercontent.com/allenai/allennlp/master/allennlp/models/language_model.py

Purpose is for testing environment and data setups.
"""

from typing import Dict, List, Tuple, Union

import torch
import numpy as np

from stog.utils.checks import ConfigurationError
from stog.utils.logging import init_logger
from stog.data.vocabulary import Vocabulary
from stog.models.model import Model
from stog.modules import StackedLstm
from stog.modules.text_field_embedders import BasicTextFieldEmbedder
from stog.modules.seq2seq_encoders import PytorchSeq2SeqWrapper
from stog.utils.nn import get_text_field_mask

logger = init_logger()

class _SoftmaxLoss(torch.nn.Module):
    """
    Given some embeddings and some targets, applies a linear layer
    to create logits over possible words and then returns the
    negative log likelihood.
    """
    def __init__(self,
                 num_words: int,
                 embedding_dim: int) -> None:
        super().__init__()

        self.softmax_w = torch.nn.Parameter(
                torch.randn(embedding_dim, num_words) / np.sqrt(embedding_dim)
        )
        self.softmax_b = torch.nn.Parameter(torch.zeros(num_words))

    def forward(self, embeddings: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # pylint: disable=arguments-differ
        # embeddings is size (n, embedding_dim)
        # targets is (batch_size, ) with the correct class id
        # Does not do any count normalization / divide by batch size
        probs = torch.nn.functional.log_softmax(
                torch.matmul(embeddings, self.softmax_w) + self.softmax_b,
                dim=-1
        )

        return torch.nn.functional.nll_loss(probs, targets.long(), reduction="sum")


class LanguageModel(Model):
    """
    The ``LanguageModel`` applies a "contextualizing"
    ``Seq2SeqEncoder`` to uncontextualized embeddings, using a ``SoftmaxLoss``
    module (defined above) to compute the language modeling loss.

    If bidirectional is True,  the language model is trained to predict the next and
    previous tokens for each token in the input. In this case, the contextualizer must
    be bidirectional. If bidirectional is False, the language model is trained to only
    predict the next token for each token in the input; the contextualizer should also
    be unidirectional.

    If your language model is bidirectional, it is IMPORTANT that your bidirectional
    ``Seq2SeqEncoder`` contextualizer does not do any "peeking ahead". That is, for its
    forward direction it should only consider embeddings at previous timesteps, and for
    its backward direction only embeddings at subsequent timesteps. Similarly, if your
    language model is unidirectional, the unidirectional contextualizer should only
    consider embeddings at previous timesteps. If this condition is not met, your
    language model is cheating.

    Parameters
    ----------
    vocab: ``Vocabulary``
    text_field_embedder: ``TextFieldEmbedder``
        Used to embed the indexed tokens we get in ``forward``.
    contextualizer: ``Seq2SeqEncoder``
        Used to "contextualize" the embeddings. As described above,
        this encoder must not cheat by peeking ahead.
    dropout: ``float``, optional (default: None)
        If specified, dropout is applied to the contextualized embeddings before computation of
        the softmax. The contextualized embeddings themselves are returned without dropout.
    """
    def __init__(self,
                 vocab: Vocabulary,
                 text_field_embedder: BasicTextFieldEmbedder,
                 contextualizer: PytorchSeq2SeqWrapper,
                 dropout: float = None) -> None:
        super().__init__()
        self._text_field_embedder = text_field_embedder

        self._contextualizer = contextualizer

        # The dimension for making predictions just in the forward direction
        self._forward_dim = contextualizer.get_output_dim()
        self._softmax_loss = _SoftmaxLoss(num_words=vocab.get_vocab_size(),
                                          embedding_dim=self._forward_dim)
        self._bidirectional = False
        self.register_buffer('_last_average_loss', torch.zeros(1))
         
        if dropout:
            self._dropout = torch.nn.Dropout(dropout)
        else:
            self._dropout = lambda x: x
        self.return_dict = {}
            
    def _get_target_token_embeddings(self,
                                     token_embeddings: torch.Tensor,
                                     mask: torch.Tensor,
                                     direction: int) -> torch.Tensor:
        # Need to shift the mask in the correct direction
        zero_col = token_embeddings.new_zeros(mask.size(0), 1).byte()
        if direction == 0:
            # forward direction, get token to right
            shifted_mask = torch.cat([zero_col, mask[:, 0:-1]], dim=1)
        else:
            shifted_mask = torch.cat([mask[:, 1:], zero_col], dim=1)
        return token_embeddings.masked_select(shifted_mask.unsqueeze(-1)).view(-1, self._forward_dim)

    def _compute_loss(self,
                      lm_embeddings: torch.Tensor,
                      token_embeddings: torch.Tensor,
                      forward_targets: torch.Tensor,
                      backward_targets: torch.Tensor = None) -> Tuple[torch.Tensor, torch.Tensor]:
        # If bidirectional, lm_embeddings is shape (batch_size, timesteps, dim * 2)
        # If unidirectional, lm_embeddings is shape (batch_size, timesteps, dim)
        # forward_targets, backward_targets (None in the unidirectional case) are
        # shape (batch_size, timesteps) masked with 0
        if self._bidirectional:
            forward_embeddings, backward_embeddings = lm_embeddings.chunk(2, -1)
            backward_loss = self._loss_helper(1, backward_embeddings, backward_targets, token_embeddings)
        else:
            forward_embeddings = lm_embeddings
            backward_loss = None

        forward_loss = self._loss_helper(0, forward_embeddings, forward_targets, token_embeddings)
        return forward_loss, backward_loss

    def _loss_helper(self,  # pylint: disable=inconsistent-return-statements
                     direction: int,
                     direction_embeddings: torch.Tensor,
                     direction_targets: torch.Tensor,
                     token_embeddings: torch.Tensor) -> Tuple[int, int]:
        mask = direction_targets > 0
        # we need to subtract 1 to undo the padding id since the softmax
        # does not include a padding dimension

        # shape (batch_size * timesteps, )
        non_masked_targets = direction_targets.masked_select(mask) - 1

        # shape (batch_size * timesteps, embedding_dim)
        non_masked_embeddings = direction_embeddings.masked_select(
                mask.unsqueeze(-1)
        ).view(-1, self._forward_dim)
        return self._softmax_loss(non_masked_embeddings, non_masked_targets)

    def num_layers(self) -> int:
        """
        Returns the depth of this LM. That is, how many layers the contextualizer has plus one for
        the non-contextual layer.
        """
        if hasattr(self._contextualizer, 'num_layers'):
            return self._contextualizer.num_layers + 1
        else:
            raise NotImplementedError(f"Contextualizer of type {type(self._contextualizer)} " +
                                      "does not report how many layers it has.")

    def forward(self,  # type: ignore
                source: Dict[str, torch.LongTensor],
                for_training=False) -> Dict[str, torch.Tensor]:
        """
        Computes the averaged forward (and backward, if language model is bidirectional)
        LM loss from the batch.

        By convention, the input dict is required to have at least a ``"tokens"``
        entry that's the output of a ``SingleIdTokenIndexer``, which is used
        to compute the language model targets.

        Parameters
        ----------
        tokens: ``torch.Tensor``, required.
            The output of ``Batch.as_tensor_dict()`` for a batch of sentences.
        for_training: unused

        Returns
        -------
        Dict with keys:

        ``'loss'``: ``torch.Tensor``
            forward negative log likelihood, or the average of forward/backward
            if language model is bidirectional
        ``'forward_loss'``: ``torch.Tensor``
            forward direction negative log likelihood
        ``'backward_loss'``: ``torch.Tensor`` or ``None``
            backward direction negative log likelihood. If language model is not
            bidirectional, this is ``None``.
        ``'lm_embeddings'``: ``Union[torch.Tensor, List[torch.Tensor]]``
            (batch_size, timesteps, embed_dim) tensor of top layer contextual representations or
            list of all layers. No dropout applied.
        ``'noncontextual_token_embeddings'``: ``torch.Tensor``
            (batch_size, timesteps, token_embed_dim) tensor of bottom layer noncontextual
            representations
        ``'mask'``: ``torch.Tensor``
            (batch_size, timesteps) mask for the embeddings
        """
        # pylint: disable=arguments-differ
        mask = get_text_field_mask(source["input_tokens"])

        # shape (batch_size, timesteps, embedding_size)
        embeddings = self._text_field_embedder(source["input_tokens"])

        # Either the top layer or all layers.
        contextual_embeddings: Union[torch.Tensor, List[torch.Tensor]] = self._contextualizer(
                embeddings, mask
        )

        return_dict = {}

        # If we have target tokens, calculate the loss.
        token_ids = source.get("output_tokens")["tokens"]
        if token_ids is not None:
            assert isinstance(contextual_embeddings, torch.Tensor)

            # Use token_ids to compute targets
            forward_targets = torch.zeros_like(token_ids)
            forward_targets[:, 0:-1] = token_ids[:, 1:]

            if self._bidirectional:
                backward_targets = torch.zeros_like(token_ids)
                backward_targets[:, 1:] = token_ids[:, 0:-1]
            else:
                backward_targets = None

            # add dropout
            contextual_embeddings_with_dropout = self._dropout(contextual_embeddings)

            # compute softmax loss
            forward_loss, backward_loss = self._compute_loss(contextual_embeddings_with_dropout,
                                                             embeddings,
                                                             forward_targets,
                                                             backward_targets)

            num_targets = torch.sum((forward_targets > 0).long())
            if num_targets > 0:
                if self._bidirectional:
                    average_loss = 0.5 * (forward_loss + backward_loss) / num_targets.float()
                else:
                    average_loss = forward_loss / num_targets.float()
            else:
                average_loss = torch.tensor(0.0).to(forward_targets.device)  # pylint: disable=not-callable
            # this is stored to compute perplexity if needed
            self._last_average_loss[0] = average_loss.detach().item()

            if num_targets > 0:
                return_dict.update({
                        'loss': average_loss,
                        'forward_loss': forward_loss / num_targets.float(),
                        'backward_loss': (backward_loss / num_targets.float()
                                          if backward_loss is not None else None),
                        'batch_weight': num_targets.float()
                })
            else:
                # average_loss zero tensor, return it for all
                return_dict.update({
                        'loss': average_loss,
                        'forward_loss': average_loss,
                        'backward_loss': average_loss if backward_loss is not None else None
                })

        return_dict.update({
                # Note: These embeddings do not have dropout applied.
                'lm_embeddings': contextual_embeddings,
                'noncontextual_token_embeddings': embeddings,
                'mask': mask
        })
        self.return_dict = return_dict
        return return_dict

    def get_metrics(self, reset=False, mimick_test=False):
        # Not actually on dev
        return {'loss': self.return_dict['loss'],
                'fwd': self.return_dict['forward_loss']}
    
    @classmethod
    def from_params(cls, vocab, params):
        logger.info('Building model...')
        text_field_embedder = BasicTextFieldEmbedder.from_params(vocab, params['text_field'])
        dropout = params['dropout']
        encoder = PytorchSeq2SeqWrapper(
            module=StackedLstm.from_params(params['encoder']),
            stateful=True
        )
        
        model = LanguageModel(
            vocab=vocab,
            text_field_embedder=text_field_embedder,
            contextualizer=encoder,
            dropout=dropout,
        )
        logger.info(model)
        return model
