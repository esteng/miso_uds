import os
import argparse

from stog.utils.params import Params
from stog.utils import logging
from stog.data.dataset_readers import (UniversalDependenciesDatasetReader,
                                       AbstractMeaningRepresentationDatasetReader,
                                       Seq2SeqDatasetReader,
                                       LanguageModelingDatasetReader,
                                       AidaEreDatasetReader)
from stog.data.iterators import BucketIterator, BasicIterator
from stog.data.token_indexers import SingleIdTokenIndexer,TokenCharactersIndexer

ROOT_TOKEN="<root>"
ROOT_CHAR="<r>"
logger = logging.init_logger()

def load_dataset_reader(dataset_type):
    if dataset_type == "UD":
        dataset_reader = UniversalDependenciesDatasetReader(
            token_indexers= {
                "tokens" : SingleIdTokenIndexer(namespace="token_ids"),
                "characters" : TokenCharactersIndexer(namespace="token_characters")
            }
        )
    elif dataset_type == "AMR":
        # TODO: Xutai
        dataset_reader = AbstractMeaningRepresentationDatasetReader(
            token_indexers= dict(
                encoder_tokens=SingleIdTokenIndexer(namespace="encoder_token_ids"),
                encoder_characters=TokenCharactersIndexer(namespace="encoder_token_characters"),
                decoder_tokens=SingleIdTokenIndexer(namespace="decoder_token_ids"),
                decoder_characters=TokenCharactersIndexer(namespace="decoder_token_characters")
            )
        )

    elif dataset_type == "MT":
        dataset_reader = Seq2SeqDatasetReader(
            token_indexers= dict(
                encoder_tokens=SingleIdTokenIndexer(namespace="encoder_token_ids"),
                encoder_characters=TokenCharactersIndexer(namespace="encoder_token_characters"),
                decoder_tokens=SingleIdTokenIndexer(namespace="decoder_token_ids"),
                decoder_characters=TokenCharactersIndexer(namespace="decoder_token_characters")
            )
        )
    elif dataset_type == "LM":
        dataset_reader = LanguageModelingDatasetReader() # all defaults

    elif dataset_type == "AIDA":
        dataset_reader = AidaEreDatasetReader()

    return dataset_reader

def load_dataset(path, dataset_type):
    return load_dataset_reader(dataset_type).read(path)

def dataset_from_params(params):

    train_data = os.path.join(params['data_dir'], params['train_data'])
    dev_data = os.path.join(params['data_dir'], params['dev_data'])
    test_data = params['test_data']
    data_type = params['data_type']

    logger.info("Building train datasets ...")
    train_data = load_dataset(train_data, data_type)

    logger.info("Building dev datasets ...")
    dev_data = load_dataset(dev_data, data_type)

    if test_data:
        test_data = os.path.join(params['data_dir'], params['test_data'])
        logger.info("Building test datasets ...")
        test_data = load_dataset(test_data, data_type)

    #logger.info("Building vocabulary ...")
    #build_vocab(fields, train_data)

    return dict(
        train=train_data,
        dev=dev_data,
        test=test_data
    )

def iterator_from_params(vocab, params):
    # TODO: There are some other options for iterator, I think we consider about it later.
    iter_type = params['iter_type']
    train_batch_size = params['train_batch_size']
    test_batch_size = params['test_batch_size']

    if iter_type == "BucketIterator":
        train_iterator = BucketIterator(
            sorting_keys=list(map(tuple, params.get('sorting_keys', []))),
            batch_size=train_batch_size,
        )
    elif iter_type == "BasicIterator":
        train_iterator = BasicIterator(
            batch_size=train_batch_size
        )
    else:
        raise NotImplementedError

    dev_iterator = BasicIterator(
        batch_size=train_batch_size
    )

    test_iterator = BasicIterator(
        batch_size=test_batch_size
    )

    train_iterator.index_with(vocab)
    dev_iterator.index_with(vocab)
    test_iterator.index_with(vocab)

    return train_iterator, dev_iterator, test_iterator
