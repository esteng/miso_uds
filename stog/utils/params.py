import json
import argparse

from stog.utils import logging

logger = logging.init_logger()


def model_opts(parser : argparse.ArgumentParser):
    # Embedding Options
    group = parser.add_argument_group('Model-Embeddings')

    group.add_argument('--token_emb_size', type=int, default=300,
                       help='Size of token embedding')

    group.add_argument('--char_emb_size', type=int, default=100,
                       help='Size of char embedding')

    group.add_argument('--pretrain_token_emb', type=str,
                       help="Pretrained embeddings, GloVe or fasttext")

    group.add_argument('--pretrain_char_emb', type=str,
                       help="")

    group.add_argument('--emb_dropout', type=float, default=0.33,
                       help="Dropout rate for embeddings")

    # Model Options
    group = parser.add_argument_group('Model')

    group.add_argument('--model_type', type=str, default="DeepBiaffineParser",
                       help="Model needs to be trained")

    group.add_argument('--hidden_dropout', type=float, default=0.33,
                       help="Dropout rate for hidden state")

    group.add_argument('--use_char_conv', action='store_true', default=False,
                       help="Whether use char conv")

    group.add_argument('--num_filters', type=int, default=100,
                       help="")

    group.add_argument('--kernel_size', type=int, default=3,
                       help="")

    group.add_argument('--encoder_layers', type=int, default=3,
                       help='Number of layers in the encoder')

    group.add_argument('--encoder_size', type=int, default=512,
                       help='Size of rnn hidden states')

    group.add_argument('--encoder_dropout', type=float, default=0.33,
                       help="Dropout rate for encoder hidden state")

    group.add_argument('--edge_hidden_size', type=int, default=512,
                       help='')

    group.add_argument('--type_hidden_size', type=int, default=128,
                       help='')

    group.add_argument('--num_labels', type=int, default=50,
                       help='')
    group.add_argument('--decode_type', choices=['greedy', 'mst'], default='greedy',
                       help='Algorithm for graph decoding.')


def preprocess_opts(parser):
    """ Pre-procesing options """
    # Data options
    group = parser.add_argument_group('Data')
    group.add_argument('--train_data', required=True,
                       help="Path to the training data")
    group.add_argument('--dev_data', required=True,
                       help="Path to the dev data")
    group.add_argument('--save_data', required=False,
                       help="place to same data")
    group.add_argument('--lower',
                       help="truecase the tokens")
    # Data processing options
    group = parser.add_argument_group('Random')
    group.add_argument('--shuffle', action="store_true", default=False,
                       help="Shuffle data")
    group.add_argument('--batch_first', action="store_true", default=False,
                       help="Batch first")

def train_opts(parser):
    """ Training and saving options """

    group = parser.add_argument_group('General')
    group.add_argument('--save_model', default=None,
                       help="""Model filename (the model will be saved as
                           <save_model>_N.pt where N is the number
                           of steps""")

    group.add_argument('--model_save_interval', default=None,
                       help="save model evert this seconds")

    group.add_argument('--file_friendly_logging', action='store_true',
                       help="Enable file friendly logging.")

    group.add_argument('--save_checkpoint_steps', type=int, default=5000,
                       help="""Save a checkpoint every X steps""")
    # GPU
    group.add_argument('--gpu', action="store_true", default=False,
                       help="deprecated see world_size and gpu_ranks.")
    group.add_argument('--cuda_device', default=0, type=int,
                       help="Cuda device ID.")

    group.add_argument('--seed', type=int, default=1,
                       help="""Python random seed used for the experiments
                           reproducibility.""")
    group.add_argument('--numpy_seed', type=int, default=1,
                       help="""NumPy random seed used for the experiments
                           reproducibility.""")
    group.add_argument('--torch_seed', type=int, default=1,
                       help="""PyTorch random seed used for the experiments
                           reproducibility.""")

    # Init options
    group = parser.add_argument_group('Initialization')
    group.add_argument('--param_init', type=float, default=0.1,
                       help="""Parameters are initialized over uniform distribution
                           with support (-param_init, param_init).
                           Use 0 to not use initialization""")

    group.add_argument('--train_from', default='', type=str,
                       help="""If training from a checkpoint then this is the
                           path to the pretrained model's state_dict.""")

    # Pretrained word vectors
    group.add_argument('--pre_word_vecs_enc',
                       help="""If a valid path is specified, then this will load
                           pretrained word embeddings on the encoder side.
                           See README for specific formatting instructions.""")
    group.add_argument('--pre_word_vecs_dec',
                       help="""If a valid path is specified, then this will load
                           pretrained word embeddings on the decoder side.
                           See README for specific formatting instructions.""")
    # Fixed word vectors
    group.add_argument('--fix_word_vecs_enc',
                       action='store_true',
                       help="Fix word embeddings on the encoder side.")
    group.add_argument('--fix_word_vecs_dec',
                       action='store_true',
                       help="Fix word embeddings on the decoder side.")

    # Optimization options
    group = parser.add_argument_group('Optimization- Type')
    group.add_argument('--batch_size', type=int, default=64,
                       help='Maximum batch size for training')
    group.add_argument('--valid_steps', type=int, default=10000,
                       help='Perfom validation every X steps')
    group.add_argument('--valid_batch_size', type=int, default=32,
                       help='Maximum batch size for validation')
    group.add_argument('-train_steps', type=int, default=100000,
                       help='Number of training steps')
    group.add_argument('--epochs', type=int, default=20,
                       help='Deprecated epochs see train_steps')
    group.add_argument('--optim', default='sgd',
                       choices=['sgd', 'adagrad', 'adadelta', 'adam',
                                'sparseadam'],
                       help="""Optimization method.""")
    group.add_argument('--adagrad_accumulator_init', type=float, default=0,
                       help="""Initializes the accumulator values in adagrad.
                           Mirrors the initial_accumulator_value option
                           in the tensorflow adagrad (use 0.1 for their default).
                           """)
    group.add_argument('--max_grad_norm', type=float, default=5,
                       help="""If the norm of the gradient vector exceeds this,
                           renormalize it to have the norm equal to
                           max_grad_norm""")
    group.add_argument('--dropout', type=float, default=0.3,
                       help="Dropout probability; applied in LSTM stacks.")
    group.add_argument('--truncated_decoder', type=int, default=0,
                       help="""Truncated bptt.""")
    group.add_argument('--adam_beta1', type=float, default=0.9,
                       help="""The beta1 parameter used by Adam.
                           Almost without exception a value of 0.9 is used in
                           the literature, seemingly giving good results,
                           so we would discourage changing this value from
                           the default without due consideration.""")
    group.add_argument('--adam_beta2', type=float, default=0.999,
                       help="""The beta2 parameter used by Adam.
                           Typically a value of 0.999 is recommended, as this is
                           the value suggested by the original paper describing
                           Adam, and is also the value adopted in other frameworks
                           such as Tensorflow and Kerras, i.e. see:
                           https://www.tensorflow.org/api_docs/python/tf/train/AdamOptimizer
                           https://keras.io/optimizers/ .
                           Whereas recently the paper "Attention is All You Need"
                           suggested a value of 0.98 for beta2, this parameter may
                           not work well for normal models / default
                           baselines.""")
    # learning rate
    group = parser.add_argument_group('Optimization- Rate')
    group.add_argument('--learning_rate', type=float, default=1.0,
                       help="""Starting learning rate.
                           Recommended settings: sgd = 1, adagrad = 0.1,
                           adadelta = 1, adam = 0.001""")
    group.add_argument('--learning_rate_decay', type=float, default=0.5,
                       help="""If update_learning_rate, decay learning rate by
                           this much if (i) perplexity does not decrease on the
                           validation set or (ii) steps have gone past
                           start_decay_steps""")
    group.add_argument('--start_decay_steps', type=int, default=50000,
                       help="""Start decaying every decay_steps after
                           start_decay_steps""")
    group.add_argument('--decay_steps', type=int, default=10000,
                       help="""Decay every decay_steps""")
    group.add_argument('--decay_method', type=str, default="",
                       choices=['noam'], help="Use a custom decay rate.")
    group.add_argument('-warmup_steps', type=int, default=4000,
                       help="""Number of warmup steps for custom decay.""")

    group = parser.add_argument_group('Logging')
    group.add_argument('--report_every', type=int, default=50,
                       help="Print stats at this interval.")
    group.add_argument('--log_file', type=str, default="",
                       help="Output logs to a file under this path.")
    group.add_argument('--exp_host', type=str, default="",
                       help="Send logs to this crayon server.")
    group.add_argument('--exp', type=str, default="",
                       help="Name of the experiment for logging.")
    # Use TensorboardX for visualization during training
    group.add_argument('--tensorboard', action="store_true",
                       help="""Use tensorboardX for visualization during training.
                           Must have the library tensorboardX.""")
    group.add_argument("--tensorboard_log_dir", type=str,
                       default="runs/onmt",
                       help="""Log directory for Tensorboard.
                           This is also the name of the run.
                           """)

class Params(object):
    """
    Parameters
    """
    def __init__(self):
        self._param_dict = {}

    def __eq__(self, other):
        if not isinstance(other, Params):
            logger.info('The params you compare is not an instance of Params.')
            return False
        if len(self._param_dict) != len(other._param_dict):
            logger.info('The numbers of parameters are different: {} != {}'.format(
                len(self._param_dict),
                len(other._param_dict)
            ))
            return False
        same = True
        for k, v in self._param_dict.items():
            if k not in other._param_dict:
                logger.info('The parameter "{}" is not specified.'.format(k))
                same = False
            elif other._param_dict[k] != v:
                logger.info('The values of "{}" not not the same: {} != {}'.format(
                    k, v, other._param_dict[k]
                ))
                same = False
        return same

    def __getattr__(self, item):
        return getattr(self, item, None)

    def set_param(self, k, v):
        self._param_dict[k] = v
        setattr(self, k, v)

    def to_file(self, output_json_file):
        with open(output_json_file, 'w', encoding='utf-8') as f:
            json.dump(self._param_dict, f)

    @classmethod
    def from_file(cls, params_json_file):
        with open(params_json_file, encoding='utf-8') as f:
            params_dict = json.load(f)

        params = cls()
        for k, v in params_dict.items():
            params.set_param(k, v)

    @classmethod
    def from_parser(cls, parser):
        params_dict = vars(parser.parse_args())

        params = cls()
        for k, v in params_dict.items():
            params.set_param(k, v)

    def __repr__(self):
        return '\n'.join(["*** {} : {}".format(key,value) for key, value in sorted(self.opt_dict.items())])

