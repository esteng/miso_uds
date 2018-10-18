import sys
import argparse
import torch
from stog.utils.params import data_opts
from stog.utils.params import Params
from stog.utils import logging
from stog.utils import ExceptionHook
from stog.data.dataset_readers import UniversalDependenciesDatasetReader, AbstractMeaningRepresentationDatasetReader
from stog.data.iterators import BucketIterator, BasicIterator
from stog.data.token_indexers import SingleIdTokenIndexer,TokenCharactersIndexer
ROOT_TOKEN="<root>"
ROOT_CHAR="<r>"
#sys.excepthook = ExceptionHook()
logger = logging.init_logger()

def load_dataset(path, dataset_type):
    if dataset_type == "UD":
        dataset_reader = UniversalDependenciesDatasetReader(
            token_indexers= {
                "tokens" : SingleIdTokenIndexer(namespace="token_ids"),
                "characters" : TokenCharactersIndexer(namespace="token_characters")
            }
        )
    else:
        dataset_reader = AbstractMeaningRepresentationDatasetReader(
            token_indexers= {
                "tokens" : SingleIdTokenIndexer(namespace="token_ids"),
                "characters" : TokenCharactersIndexer(namespace="token_characters")
            }
        )

    return dataset_reader.read(path)


def dataset_from_params(opt):


    logger.info("Building train datasets ...")
    train_data = load_dataset(opt.train_data, opt.data_type)

    logger.info("Building dev datasets ...")
    dev_data = load_dataset(opt.dev_data, opt.data_type)

    test_data = None
    if opt.test_data:
        logger.info("Building test datasets ...")
        test_data = load_dataset(opt.test_data, opt.data_type)

    logger.info("Building vocabulary ...")
    #build_vocab(fields, train_data)

    return dict(
        train=train_data,
        dev=dev_data,
        test=test_data
    )

def iterator_from_params(vocab, opt):
    # TODO: There are some other options for iterator, I think we consider about it later.
    if opt.iter_type == "BucketIterator":
        train_iterator = BucketIterator(
            sorting_keys=[("words", "num_tokens")],
            batch_size=opt.batch_size,
        )
    elif opt.iter_type == "BasicIterator":
        train_iterator = BasicIterator(
            batch_size=opt.batch_size
        )
    else:
        raise NotImplementedError

    dev_iterator = BasicIterator(
        batch_size=opt.batch_size
    )

    train_iterator.index_with(vocab)
    dev_iterator.index_with(vocab)

    return train_iterator, dev_iterator


if __name__ == "__main__":
    parser = argparse.ArgumentParser("build_dataset.py")
    data_opts(parser)
    opt = Params.from_parser(parser)
    dataset = dataset_from_params(opt)
    import pdb;pdb.set_trace()
