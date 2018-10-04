import argparse
import sys
import torch
from torchtext import data
from torchtext.data.field import RawField

class StupidDict(dict):
    def __init__(self, list):
        self.list = list

    def items(self):
        return self.list

    def values(self):
        return [ (name, value) for _, (name , value) in self.list]

    def __getitem__(self, i):
        return self.list[i]

class RelationField(RawField):
    """
    A class for relations between tokens
    """
    def __init__(
            self,
            batch_first=False,
            is_target=False
    ):
        self.batch_first = batch_first
        self.is_target = is_target

    def preprocess(self, x):
        return x

    def process(self, batch, device=None, train=False):
        max_len = max(len(item) for item in batch)
        batch_size = len(batch)

        # Batch tensors
        batch_relation_tensor = torch.zeros(
            [ batch_size, max_len, max_len + 1]
        )
        batch_relation_tensor_mask = torch.zeros(
            [ batch_size, max_len, max_len + 1]
        )
        batch_relation_tensor_mask[:, :max_len,:max_len + 1] = 1

        for idx_in_batch, example in enumerate(batch):
            # map token index in UD to integer
            index_mapper = {item[0] : i for i, item in enumerate(example)}
            index_mapper["0"] = len(index_mapper)
            relations_child = [index_mapper[item[0]] for item in example]
            relations_father = [index_mapper[item[1]] for item in example]

            batch_relation_tensor[idx_in_batch, relations_child, relations_father] = 1

        # Move to GPU if needed
        batch_relation_tensor.to(device)
        batch_relation_tensor_mask.to(device)

        return batch_relation_tensor, batch_relation_tensor_mask




def get_fields(opt):
    """
    Build fields, include token and relations
    :param opt:
    :return:
    """
    fields = {}

    fields['tokens'] = data.Field(
        sequential=True,
        lower=opt.lower,
        batch_first=opt.batch_first
    )

    fields['chars'] = data.NestedField(
        data.Field(
            tokenize=list,
            init_token="<bos>",
            eos_token="<eos>"
        ),
    )

    #fields['relations'] = RelationField(
    #    batch_first=opt.batch_first,
    #    is_target=True
    #)

    fields['headers'] = data.Field(
        sequential=True,
    )

    return fields

def get_dataset_splits(opt):
    """
    Build train dev test data set
    :param opt: some options
    :return: three data set splits
    """

    # get fields first
    fields = get_fields(opt)

    # It's quite hacky here, but that's the best I can do without modifying torch text
    stupid_dict = StupidDict(
        [
            ("tokens", ( "tokens", fields['tokens'] ) ),
            ("tokens", ( "chars", fields["chars"] ) ),
            ("relations" , ( "relations", fields["relations"]))
        ]
    )
    train, dev, test = data.TabularDataset.splits(
        path=opt.data, format="JSON",
        train='train.json', validation='dev.json', test='test.json',
        fields=stupid_dict
    )

    for dataset in [train, dev, test]:
        dataset.fields = fields

    # Build vocab
    fields['tokens'].build_vocab(train)
    fields['chars'].build_vocab(train)

    return train, dev, test

def get_dataset(path):
    """
    Build only one data set
    :param path: path to json data
    :return: data set
    """
    dataset = data.TabularDataset(
        path=path, format="JSON"
    )
    return dataset

def get_iterator(dataset, batch_size):
    """
    Build an iterator for training or inference given dataset
    :param dataset: torchtext.dataset
    :param batch_size:
    :return: iterator
    """
    train_iter = data.BucketIterator(
        dataset = dataset, batch_size = batch_size,
        sort_key = lambda x: len(x)
    )
    return train_iter

if __name__== "__main__":
    parser = argparse.ArgumentParser('dataset')
    parser.add_argument("--data", required=True,
                        help="The path of data directory. The the files in data path should be {train,dev,test}.json")
    parser.add_argument("--save", required=True,
                        help="The place to save data")
    parser.add_argument("--lower", action="store_true", default=False,
                        help="Whether lower the tokens")
    parser.add_argument("--batch_first", action="store_true", default=False,
                        help="Whether let the batch dim first")
    parser.add_argument("--debug", action="store_true", default=False,
                        help="Debug mode, will stop at an exception")
    opt = parser.parse_args()

    sys.stderr.write("We can process data in training scripts for now since the data set is quite small\n")






