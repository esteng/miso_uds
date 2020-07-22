from typing import Type
import logging

import torch
from allennlp.common import Params
from allennlp.common.checks import ConfigurationError
from allennlp.modules.seq2seq_encoders import PytorchSeq2SeqWrapper, Seq2SeqEncoder
#from allennlp.modules.seq2seq_encoders import _Seq2SeqWrapper
from allennlp.modules.encoder_base import RnnStateStorage

from miso.modules.stacked_bilstm import MisoStackedBidirectionalLstm

from .seq2seq_bert_encoder import Seq2SeqBertEncoder, BaseBertWrapper

#class _Seq2SeqWrapper:
#    """
#    For :class:`Registrable` we need to have a ``Type[Seq2SeqEncoder]`` as the value registered for each
#    key.  What that means is that we need to be able to ``__call__`` these values (as is done with
#    ``__init__`` on the class), and be able to call ``from_params()`` on the value.
#
#    In order to accomplish this, we have two options: (1) we create a ``Seq2SeqEncoder`` class for
#    all of pytorch's RNN modules individually, with our own parallel classes that we register in
#    the registry; or (2) we wrap pytorch's RNNs with something that `mimics` the required
#    API.  We've gone with the second option here.
#
#    This is a two-step approach: first, we have the :class:`PytorchSeq2SeqWrapper` class that handles
#    the interface between a pytorch RNN and our ``Seq2SeqEncoder`` API.  Our ``PytorchSeq2SeqWrapper``
#    takes an instantiated pytorch RNN and just does some interface changes.  Second, we need a way
#    to create one of these ``PytorchSeq2SeqWrappers``, with an instantiated pytorch RNN, from the
#    registry.  That's what this ``_Wrapper`` does.  The only thing this class does is instantiate
#    the pytorch RNN in a way that's compatible with ``Registrable``, then pass it off to the
#    ``PytorchSeq2SeqWrapper`` class.
#
#    When you instantiate a ``_Wrapper`` object, you give it an ``RNNBase`` subclass, which we save
#    to ``self``.  Then when called (as if we were instantiating an actual encoder with
#    ``Encoder(**params)``, or with ``Encoder.from_params(params)``), we pass those parameters
#    through to the ``RNNBase`` constructor, then pass the instantiated pytorch RNN to the
#    ``PytorchSeq2SeqWrapper``.  This lets us use this class in the registry and have everything just
#    work.
#    """
#    PYTORCH_MODELS = [torch.nn.GRU, torch.nn.LSTM, torch.nn.RNN]
#
#    def __init__(self, module_class: Type[torch.nn.modules.RNNBase]) -> None:
#        self._module_class = module_class
#
#    def __call__(self, **kwargs) -> PytorchSeq2SeqWrapper:
#        return self.from_params(Params(kwargs))
#
#    # Logic requires custom from_params
#    def from_params(self, 
#                    params: Params,
#                    constructor_to_call,
#                    constructor_to_inspect
#                    ) -> PytorchSeq2SeqWrapper:
#        if not params.pop_bool('batch_first', True):
#            raise ConfigurationError("Our encoder semantics assumes batch is always first!")
#        if self._module_class in self.PYTORCH_MODELS:
#            params['batch_first'] = True
#        stateful = params.pop_bool('stateful', False)
#        module = self._module_class(**params.as_dict(infer_type_and_cast=True))
#        return PytorchSeq2SeqWrapper(module, stateful=stateful)
#
#
#
#class _PytorchSeq2SeqWrapper(PytorchSeq2SeqWrapper):
#    """
#    Inherit ``PytorchSeq2SeqWrapper'' to expose the final states.
#    """
#
#    def __init__(self, *args, **kwargs) -> None:
#        super().__init__(*args, **kwargs)
#
#
#
#class _MisoSeq2SeqWrapper(PytorchSeq2SeqWrapper):
#    """
#    It inherits ``_Seq2SeqWrapper'' from AllenNLP such that we have more flexibility
#    of defining our own seq2seq encoders.
#    """
#    def __init__(self, *args, **kwargs) -> None:
#        super().__init__(*args, **kwargs)
#
#    #def __call__(self, **kwargs) -> PytorchSeq2SeqWrapper:
#    #    return self.from_params(Params(kwargs))
#
#    def get_final_states(self) -> RnnStateStorage:
#        return self._states
#
#    # Logic requires custom from_params
#    #def from_params(self, 
#    #                params: Params,
#    #                constructor_to_call,
#    #                constructor_to_inspect
#    #                ) -> _PytorchSeq2SeqWrapper:
#    #    if not params.pop_bool('batch_first', True):
#    #        raise ConfigurationError("Our encoder semantics assumes batch is always first!")
#    #    if self._module_class in self.PYTORCH_MODELS:
#    #        params['batch_first'] = True
#    #    stateful = params.pop_bool('stateful', False)
#    #    module = self._module_class(**params.as_dict(infer_type_and_cast=True))
#    #    return _PytorchSeq2SeqWrapper(module, stateful=stateful)
#
#
## pylint: disable=protected-access
##Seq2SeqEncoder.register("miso_stacked_bilstm")(_MisoSeq2SeqWrapper(MisoStackedBidirectionalLstm))
