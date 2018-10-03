import sys
import argparse
import torch
import dill
from torchtext import data
from torchtext.data.field import RawField
from torchtext.vocab import Vectors
from stog.utils.opts import preprocess_opts
from stog.utils.opts import Options
from stog.utils.logging import init_logger, logger
from stog.utils import ExceptionHook
ROOT_TOKEN="<root>"
ROOT_CHARS=list(ROOT_TOKEN)
#sys.excepthook = ExceptionHook()

class StupidDict(dict):
    def __init__(self, list):
        self.list = list

    def items(self):
        return self.list

    def values(self):
        return [ (name, value) for _, (name , value) in self.list]

    def __getitem__(self, i):
        return self.list[i]

class HeaderField(RawField):
    def __init__(
            self,
            batch_first=True,
            is_target=True
    ):
        self.batch_first = batch_first
        self.is_target = is_target

    def preprocess(self, x):
        # add 0 for root node
        return [0] + [int(item) for item in x]

    def process(self, batch, device):
        max_len = max(len(item) for item in batch)
        batch_size = len(batch)
        if self.batch_first:
            batch_headers = torch.zeros(batch_size, max_len, dtype=torch.long)
            batch_mask = torch.zeros(batch_size, max_len, dtype=torch.float)
            for idx_in_batch, example in enumerate(batch):
                batch_headers[idx_in_batch, :len(example)] = torch.LongTensor(example)
                batch_mask[idx_in_batch, : len(example)] = 1
        else:
            raise NotImplementedError

        batch_headers.to(device)
        batch_mask.to(device)

        return batch_headers, batch_mask


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

    def process(self, batch, device=None):
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
        batch_first=opt.batch_first,
        preprocessing=lambda x: [ROOT_TOKEN] + x
    )

    fields['chars'] = data.NestedField(
        data.Field(
            tokenize=list,
            eos_token=None,
        ),
        preprocessing=lambda x: [ROOT_TOKEN] + x
    )

    fields['headers'] = HeaderField(
        batch_first=opt.batch_first
    )

    return fields


def get_dataset(data_path, fields):
    """
    Build train dev test data set
    :param opt: some options
    :return: three data set splits
    """

    # It's quite hacky here, but that's the best I can do without modifying torch text
    stupid_dict = StupidDict(
        [
            ("tokens", ( "tokens", fields['tokens'] ) ),
            ("tokens", ( "chars", fields["chars"] ) ),
            ("headers" , ( "headers", fields["headers"]))
        ]
    )
    dataset = data.TabularDataset(
        path=data_path, format="JSON",
        fields=stupid_dict
    )

    dataset.fields = fields


    return dataset

def build_vocab(fields, data):
    # Build vocab
    fields['tokens'].build_vocab(data)
    fields['chars'].build_vocab(data)


def get_iterator(dataset, opt):
    """
    Build an iterator for training or inference given dataset
    :param dataset: torchtext.dataset
    :param batch_size:
    :return: iterator
    """
    iter = data.BucketIterator(
        dataset = dataset, batch_size = opt.batch_size,
        sort_key = lambda x: len(x)
    )
    return iter

def save_fields_to_vocab(fields):
    """
    Save Vocab objects in Field objects to `vocab.pt` file.
    """
    vocab = []
    for k, f in fields.items():
        if f is not None and 'vocab' in f.__dict__:
            f.vocab.stoi = f.vocab.stoi
            vocab.append((k, f.vocab))
    return vocab

def preprocess(opt):
    init_logger()

    logger.info("Getting fields ...")
    fields = get_fields(opt)

    logger.info("Building train datasets ...")
    train_data = get_dataset(opt.train_data, fields)

    logger.info("Building dev datasets ...")
    dev_data = get_dataset(opt.dev_data, fields)

    logger.info("Building vocabulary ...")
    build_vocab(fields, train_data)

    return train_data, dev_data
    #TODO save the preprocesse data. This is tricky since torchtext use lambda expression, which can't be serialized.
    #logger.info("Saving train data at {} ... ".format(opt.save_data + ".train.pt"))
    #with open(opt.save_data + ".train.pt", 'wb') as f:
    #    dill.dump(train_data, f)
    #
    #logger.info("Saving dev data at {} ... ".format(opt.save_data  + ".dev.pt"))
    #with open(opt.save_data + ".dev.pt", 'wb') as f:
    #    dill.dump(dev_data, f)
    #
    #logger.info("Saving vocab data at {} ... ".format(opt.save_data  + ".vocab.pt"))
    #with open(opt.save_data + ".vocab.pt", 'wb') as f:
    #   vocab = save_fields_to_vocab(fields)
    #    dill.dump(vocab, f)

    #logger.info("Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser("preprocess.py")
    preprocess_opts(parser)
    opt = Options(parser)
    train_data, dev_data = preprocess(opt)
    train_iter = data.BucketIterator(
        dataset=train_data,
        batch_size=64,
        sort_key=lambda x: len(x),
        device=None,
        train=True,
        repeat=None,
        shuffle=None
    )

    from stog.utils.tqdm import Tqdm
    for batch in Tqdm.tqdm(train_iter):
        import pdb;pdb.set_trace()
