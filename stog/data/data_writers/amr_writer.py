import torch
from stog.data.data_writers.data_writer import DataWriter
from overrides import overrides
from stog.data.amr import AMRTree
class AbstractMeaningRepresentationDataWriter(DataWriter):

    @overrides
    def predict_instance_batch(self, torch_dict, batch):
        batch_size, max_len = torch_dict['mask'].size()
        #import pdb;pdb.set_trace()
        for i in range(batch_size):
            current_len = int(torch.sum(torch_dict['mask'][i]).tolist())
            head_tags = [self.vocab.get_token_from_index(x, "head_tags") for x in torch_dict['relations'][i][:current_len].tolist()]
            head_indices = torch_dict['headers'][i][:current_len].tolist()
            head_indices = [x for x in head_indices]
            tokens = [ self.vocab.get_token_from_index(x,"token_ids") for x in batch['words']['tokens'][i][:current_len].tolist() ]
            coref = batch['coref'][i][:current_len].tolist()

            yield self.predict_instance(
                    {
                        'head_indices' : head_indices,
                        'head_tags' : head_tags,
                        'tokens' : tokens,
                        'coref' : coref
                    }
            )

    def predict_instance(self, list_dict):
        t = AMRTree()
        t.recover_from_list(list_dict)
        return t








