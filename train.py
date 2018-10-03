import argparse
import sys
from torchtext.data import BucketIterator
from stog.data import get_iterator, get_dataset_splits
from stog.utils import init_logger, logger
from stog.utils import ExceptionHook
from stog.utils.opts import preprocess_opts, model_opts, train_opts, Options
from stog.model_builder import build_model
from preprocess import preprocess, get_iterator
from stog.models.deep_biaffine_parser import DeepBiaffineParser
from stog.trainer import Trainer
from stog.modules.optimizer import build_optim

def main(opt):

    init_logger()

    # preprocess data
    logger.info("Loading data ...")
    train_data, dev_data = preprocess(opt)

    #logger.info("Building training iterator ...")
    #train_iter = get_iterator(opt, train_data)

    # build model
    logger.info("Building model ...")
    model = build_model(opt, train_data)

    # build optimizer
    #logger.info("Building optimizer ...")
    optim = build_optim(opt, model)

    # build trainer
    logger.info("Building Trainer...")

    trainer = Trainer(
        model=model,
        optimizer=optim,
        iterator=BucketIterator,
        training_dataset=train_data,
        dev_dataset=dev_data,
        dev_iterator=BucketIterator,
        dev_metric='loss',
        use_gpu=opt.gpu,
        patience=None,
        grad_clipping=None,
        shuffle=opt.shuffle,
        num_epochs=20,
        serialization_dir=opt.save_model,
        num_serialized_models_to_keep=20,
        model_save_interval=opt.model_save_interval,
        summary_interval=100,
        batch_size=opt.batch_size
    )

    trainer.train()


if __name__ == "__main__":
    parser = argparse.ArgumentParser('train.py')
    preprocess_opts(parser)
    model_opts(parser)
    train_opts(parser)
    opt =  Options(parser)
    main(opt)

