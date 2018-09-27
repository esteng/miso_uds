import argparse

def model_opts(parser : argparse.ArgumentParser):
    # Embedding Options
    group = parser.add_argument_group('Model-Embeddings')
    group.add_argument('--word_vec_size', type=int, default=-1,
                       help='Word embedding size for src and tgt.')

    # Encoder-Decoder Options
    group = parser.add_argument_group('Model')
    group.add_argument('--model_type', default=None,
                       help="Which model to run")
    group.add_argument('--enc_layers', type=int, default=2,
                       help='Number of layers in the encoder')
    group.add_argument('--rnn_size', type=int, default=500,
                       help='Size of rnn hidden states')
    group.add_argument('--rnn_type', type=str, default='LSTM',
                       choices=['LSTM', 'GRU'],
                       help="""The gate type to use in the RNNs""")

def preprocess_opts(parser):
    """ Pre-procesing options """
    # Data options
    group = parser.add_argument_group('Data')
    group.add_argument('--train_data', required=True,
                       help="Path to the training data")
    group.add_argument('--dev_data', required=True,
                       help="Path to the dev data")
    group.add_argument('--save_data',
                       help="place to same data"
                       )

    # Data processing options
    group = parser.add_argument_group('Random')
    group.add_argument('--shuffle', type=int, default=1,
                       help="Shuffle data")
    group.add_argument('--seed', type=int, default=3435,
                       help="Random seed")

    group = parser.add_argument_group('Logging')
    group.add_argument('--report_every', type=int, default=100000,
                       help="Report status every this many sentences")
    group.add_argument('--log_file', type=str, default="",
                       help="Output logs to a file under this path.")

def train_opts(parser):
    """ Training and saving options """

    group = parser.add_argument_group('General')
    group.add_argument('--save_model', default='model',
                       help="""Model filename (the model will be saved as
                           <save_model>_N.pt where N is the number
                           of steps""")

    group.add_argument('--save_checkpoint_steps', type=int, default=5000,
                       help="""Save a checkpoint every X steps""")
    # GPU
    group.add_argument('--gpuid', default=[], nargs='+', type=int,
                       help="Deprecated see world_size and gpu_ranks.")

    group.add_argument('--seed', type=int, default=-1,
                       help="""Random seed used for the experiments
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
    group.add_argument('--epochs', type=int, default=0,
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

class Options(object):
    """
    Options
    """
    def __init__(self, parser):
        parser.add_argument("--json",
                           help="a json file that contains all the options")
        opt = parser.parse_argument()

        if opt.json:
            import json
            with open(opt.json, 'r') as f:
                options = json.load(f)
        else:
            options = vars(args)

        for key, value in options.items():
            setattr(self, key, value)






