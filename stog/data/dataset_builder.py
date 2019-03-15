import os
import argparse

from collections import defaultdict

from stog.utils.params import Params
from stog.utils import logging
from stog.data.dataset_readers import DatasetReader
from stog.data.iterators.data_iterator import DataIterator
from stog.data.token_indexers import SingleIdTokenIndexer,TokenCharactersIndexer
from stog.data.vocabulary import Vocabulary

ROOT_TOKEN="<root>"
ROOT_CHAR="<r>"
logger = logging.init_logger()

def seq2seq_token_char_indexers(*args, **kwargs):
    return dict(
        encoder_tokens=SingleIdTokenIndexer(namespace="encoder_token_ids"),
        encoder_characters=TokenCharactersIndexer(namespace="encoder_token_characters"),
        decoder_tokens=SingleIdTokenIndexer(namespace="decoder_token_ids"),
        decoder_characters=TokenCharactersIndexer(namespace="decoder_token_characters")
    )

def shared_seq2seq_token_char_indexers(*args, **kwargs):
    return dict(
        tokens=SingleIdTokenIndexer(namespace="token_ids"),
        characters=TokenCharactersIndexer(namespace="token_characters")
    )

def load_dataset_reader(params):
    if params["share_vocab"]:
        indexer = shared_seq2seq_token_char_indexers() 
    else:
        indexer = seq2seq_token_char_indexers()

    return DatasetReader.by_name(params["type"])(
        token_indexers=indexer,
        word_splitter=params.get('word_splitter', None)
    )

def load_dataset(path, params):
    return load_dataset_reader(params).read(path)

def dataset_from_params(params):
    data_dict = defaultdict()

    for section in ["train", "dev", "test"]:
        if params.get(section, None) is not None:
            logger.info("Building {} datasets ...".format(section))
            data_dict[section] = load_dataset(
                os.path.join(params['directory'], params[section]), 
                params
            )

    return data_dict


def iterator_from_params(vocab, params):
    train_batch_size = params['train_batch_size']
    test_batch_size = params['test_batch_size']

    train_iter_param = dict(batch_size=train_batch_size)

    if params["type"] == "bucket":
        train_iter_param["sorting_keys"]=[("tgt_tokens", "num_tokens")]

    train_iterator = DataIterator.by_name(params["type"])(**train_iter_param)
    train_iterator.index_with(vocab)

    dev_iterator = DataIterator.by_name("basic")(batch_size=train_batch_size)
    dev_iterator.index_with(vocab)

    test_iterator = DataIterator.by_name("basic")(batch_size=test_batch_size)
    test_iterator.index_with(vocab)

    return {
        "train" : train_iterator, 
        "dev" : dev_iterator, 
        "test" : test_iterator
    }

def vocab_from_params(params, dataset, path_to_save=None):
    # Vocabulary and iterator are created here.
    vocab = Vocabulary.from_instances(instances=dataset, **params)
    # Initializing the model can have side effect of expanding the vocabulary
    if path_to_save is not None:
        vocab.save_to_files(path_to_save)
    return vocab
