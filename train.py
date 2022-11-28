import argparse
import ast
import os
import random
import re
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)

import numpy as np
import pandas as pd
import tensorflow as tf

from synthesizer.hparams import hparams, get_image_list, get_image_list_2
from synthesizer.train import tacotron_train
from utils.argutils import print_args


def prepare_run(args):
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = str(args.tf_log_level)
    run_name = args.name
    log_dir = os.path.join(args.models_dir, "logs-{}".format(run_name))
    os.makedirs(log_dir, exist_ok=True)

    # seed the randomizer
    seed = hparams.tacotron_random_seed
    np.random.seed(seed)
    random.seed(seed)
    tf.random.set_random_seed(seed)
    tf.set_random_seed(seed)

    hparams.set_hparam('img_height', args.image_height)
    hparams.set_hparam('img_width', args.image_width)

    hparams.set_hparam('tacotron_batch_size', args.batch_size)
    args.num_test_batches = args.num_test_samples // args.batch_size

    if args.greyscale: 
        hparams.set_hparam('num_channels', 1)

    if args.num_speakers:
        hparams.set_hparam('speaker_disentanglement', True)
        hparams.set_hparam('num_speakers', args.num_speakers)

    if args.use_deltas_features:
        hparams.set_hparam('use_deltas_features', True)
        hparams.set_hparam('encoder_lstm_units', 128)

    # scheduled teacher forcing
    if args.tf_decay_start and args.tf_decay_steps:
        hparams.set_hparam('tacotron_teacher_forcing_mode', 'scheduled')
        hparams.set_hparam('tacotron_teacher_forcing_start_decay', args.tf_decay_start)
        hparams.set_hparam('tacotron_teacher_forcing_decay_steps', args.tf_decay_steps)

    # scheduled learning rate
    if args.lr_decay_start and args.lr_decay_steps:
        hparams.set_hparam('tacotron_start_decay', args.lr_decay_start)
        hparams.set_hparam('tacotron_decay_steps', args.lr_decay_steps)

    return log_dir, hparams


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("name", help="Name of the run and of the logging directory.")
    parser.add_argument("-m", "--models_dir", type=str, default="synthesizer/saved_models/",
                        help="Path to the output directory that will contain the saved model weights and the logs.")
    parser.add_argument("--mode", default="synthesis", help="mode for synthesis of tacotron after training")
    parser.add_argument("--GTA", default="True", help="Ground truth aligned synthesis, defaults to True, only "
                                                      "considered in Tacotron synthesis mode")
    parser.add_argument("--restore", type=bool, default=True, help="Set this to False to do a fresh training")
    parser.add_argument("--summary_interval", type=int, default=100,
                        help="Steps between running summary ops")
    parser.add_argument("--embedding_interval", type=int, default=1000000000,
                        help="Steps between updating embeddings projection visualization")
    parser.add_argument("--checkpoint_interval", type=int, default=1000, # Was 5000
                        help="Steps between writing checkpoints")
    parser.add_argument("--eval_interval", type=int, default=100, # Was 10000
                        help="Steps between eval on test data")
    parser.add_argument("--tacotron_train_steps", type=int, default=2000000, # Was 100000
                        help="total number of tacotron training steps")
    parser.add_argument("--tf_log_level", type=int, default=1, help="Tensorflow C++ log level.")
    parser.add_argument('log_number', type=int)
    parser.add_argument('--num_test_samples', type=int, default=512)
    parser.add_argument('--checkpoint_path')
    parser.add_argument('--videos_root')
    parser.add_argument('--apply_augmentation', action='store_true')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--image_height', type=int, default=128)  # face ROI
    parser.add_argument('--image_width', type=int, default=128)
    parser.add_argument('--training_sample_pool_location', default='/tmp/training_sample_pool')
    parser.add_argument('--val_sample_pool_location', default='/tmp/val_sample_pool')
    parser.add_argument('--use_selection_weights', action='store_true')
    parser.add_argument('--greyscale', action='store_true')
    parser.add_argument('--num_speakers', type=int)  # 50 for SRAVI
    parser.add_argument('--use_deltas_features', action='store_true')
    parser.add_argument('--tf_decay_start', type=int)
    parser.add_argument('--tf_decay_steps', type=int)
    parser.add_argument('--lr_decay_start', type=int)
    parser.add_argument('--lr_decay_steps', type=int)
    parser.add_argument('--use_cer_metric', action='store_true')
    parser.add_argument('--cer_interval', type=int)

    args = parser.parse_args()

    log_dir, hparams = prepare_run(args)
    print_args(args, parser)

    tacotron_train(args, log_dir, hparams)
