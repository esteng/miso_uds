import argparse
from stog.dataset import get_iterator, get_dataset_splits
from stog.utils import init_logger, logger

def validate():
    pass

def test():
    pass

def train(opt):
    init_logger()

    logger.info("Building datasets")
    train_data, dev_data, test_data = get_dataset_splits(opt)
    logger.info("Building training iterator")
    train_iter = get_iterator(train_data, batch_size=opt.batch_size)

    for epoch_idx in range(opt.epochs):
        logger.info("Start epoch {}".format(epoch_idx + 1))
        for batch_idx, batch_data in enumerate(train_iter):
            # tokens_tensor size (batch_size, len)
            tokens_tensor = batch_data.tokens
            # relation*_tenor size (batch_size, len, len + 1)
            relation_tensor, relation_mask_tensor = batch_data.relations

            #TODO do somethin here use tokens and relations


if __name__ == "__main__":
    parser = argparse.ArgumentParser('train')
    parser.add_argument("--data", required=True,
                        help="The path of data directory. The the files in data path should be {train,dev,test}.json")
    parser.add_argument("--lower", action="store_true", default=False,
                        help="Whether lower the tokens")
    parser.add_argument("--batch_first", action="store_true", default=False,
                        help="Whether let the batch dim first")
    parser.add_argument("--batch_size", default=64,
                        help="Batch size")
    parser.add_argument("--epochs", default=10,
                        help="Epochs to run")
    opt = parser.parse_args()

    train(opt)

