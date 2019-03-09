import os
import argparse

from miso.utils.params import Params
from miso.utils import logging
from miso.data.dataset_readers import UniversalDependenciesDatasetReader, AbstractMeaningRepresentationDatasetReader, Seq2SeqDatasetReader
from miso.data.iterators import BucketIterator, BasicIterator
from miso.data.token_indexers import SingleIdTokenIndexer,TokenCharactersIndexer

ROOT_TOKEN="<root>"
ROOT_CHAR="<r>"
logger = logging.init_logger()


def load_dataset_reader(dataset_type, *args, **kwargs):
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
            token_indexers=dict(
                encoder_tokens=SingleIdTokenIndexer(namespace="encoder_token_ids"),
                encoder_characters=TokenCharactersIndexer(namespace="encoder_token_characters"),
                decoder_tokens=SingleIdTokenIndexer(namespace="decoder_token_ids"),
                decoder_characters=TokenCharactersIndexer(namespace="decoder_token_characters")
            ),
            word_splitter=kwargs.get('word_splitter', None)
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

    return dataset_reader


def load_dataset(path, dataset_type, *args, **kwargs):
    return load_dataset_reader(dataset_type, *args, **kwargs).read(path)


def dataset_from_params(params):

    train_data = os.path.join(params['data_dir'], params['train_data'])
    dev_data = os.path.join(params['data_dir'], params['dev_data'])
    test_data = params['test_data']
    data_type = params['data_type']

    logger.info("Building train datasets ...")
    train_data = load_dataset(train_data, data_type, **params)

    logger.info("Building dev datasets ...")
    dev_data = load_dataset(dev_data, data_type, **params)

    if test_data:
        test_data = os.path.join(params['data_dir'], params['test_data'])
        logger.info("Building test datasets ...")
        test_data = load_dataset(test_data, data_type, **params)

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
            sorting_keys=[("tgt_tokens", "num_tokens")],
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
