#import overrides 
#import logging
#from collections import defaultdict
#from typing import Dict, List, Union, Iterator, Iterable
#
#import numpy
#import torch
#
#from allennlp.common.checks import ConfigurationError
#from allennlp.common.util import ensure_list
#from allennlp.data.instance import Instance
#from allennlp.data.vocabulary import Vocabulary
#from allennlp.data.dataset import Batch 
#
#
#logger = logging.getLogger(__name__)  # pylint: disable=invalid-name
#class DecompBatch(Batch):
#
#    @overrides
#    def as_tensor_dict(self,
#                       padding_lengths: Dict[str, Dict[str, int]] = None,
#                       verbose: bool = False) -> Dict[str, Union[torch.Tensor, Dict[str, torch.Tensor]]]:
#        # This complex return type is actually predefined elsewhere as a DataArray,
#        # but we can't use it because mypy doesn't like it.
#        if padding_lengths is None:
#            padding_lengths = defaultdict(dict)
#        # First we need to decide _how much_ to pad.  To do that, we find the max length for all
#        # relevant padding decisions from the instances themselves.  Then we check whether we were
#        # given a max length for a particular field and padding key.  If we were, we use that
#        # instead of the instance-based one.
#        if verbose:
#            logger.info("Padding batch of size %d to lengths %s", len(self.instances), str(padding_lengths))
#            logger.info("Getting max lengths from instances")
#        instance_padding_lengths = self.get_padding_lengths()
#        if verbose:
#            logger.info("Instance max lengths: %s", str(instance_padding_lengths))
#        lengths_to_use: Dict[str, Dict[str, int]] = defaultdict(dict)
#        for field_name, instance_field_lengths in instance_padding_lengths.items():
#            for padding_key in instance_field_lengths.keys():
#                if padding_key in padding_lengths[field_name]:
#                    lengths_to_use[field_name][padding_key] = padding_lengths[field_name][padding_key]
#                else:
#                    lengths_to_use[field_name][padding_key] = instance_field_lengths[padding_key]
#
#        # Now we actually pad the instances to tensors.
#        field_tensors: Dict[str, list] = defaultdict(list)
#        if verbose:
#            logger.info("Now actually padding instances to length: %s", str(lengths_to_use))
#        for instance in self.instances:
#            for field, tensors in instance.as_tensor_dict(lengths_to_use).items():
#                field_tensors[field].append(tensors)
#
#        # Finally, we combine the tensors that we got for each instance into one big tensor (or set
#        # of tensors) per field.  The `Field` classes themselves have the logic for batching the
#        # tensors together, so we grab a dictionary of field_name -> field class from the first
#        # instance in the batch.
#        field_classes = self.instances[0].fields
#        final_fields = {}
#        for field_name, field_tensor_list in field_tensors.items():
#            logger.info(field_name)
#            logger.info(len(field_tensor_list))
#            sys.exit()
#
#            final_fields[field_name] = field_classes[field_name].batch_tensors(field_tensor_list)
#
#        return final_fields
