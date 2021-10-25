import random
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

from synthesizer.hparams import hparams, get_image_list, get_image_list_2
from synthesizer.train import tacotron_train
from utils.argutils import print_args
#from synthesizer import infolog
import argparse
from pathlib import Path
import os

#Prepares the data.
def prepare_run(args):
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = str(args.tf_log_level)
    run_name = args.name
    log_dir = os.path.join(args.models_dir, "logs-{}".format(run_name))
    os.makedirs(log_dir, exist_ok=True)

    # all_images = get_image_list('test', args.data_root)
    # all_test_images = get_image_list('test', args.data_root)

    all_speakers = [p.name for p in Path(args.data_root).glob('*') if p.is_dir()]
    test_speakers = ['s27', 's34']  # hold out speakers (man, woman) for testing (~2000 examples)
    training_speakers = list(set(all_speakers) - set(test_speakers))
    print('Num Total Speakers:', len(all_speakers))
    print('Num Training Speakers:', len(training_speakers))

    # grab phrases from users in training to hold out for testing
    all_training_phrases = list([p.name for p in Path(args.data_root).glob('*/*')
                                 if p.is_dir() and p.parts[-2] in training_speakers])
    print('Total Num Training Phrases:', len(all_training_phrases))
    random.seed(hparams.tacotron_data_random_state)
    random.shuffle(all_training_phrases)
    holdout_phrases = all_training_phrases[:100]  # 100 examples
    holdout_phrases = ['swag7p', 'bwiq7s', 'sgia2n', 'prwv4p', 'lrab2s', 'bbwq8s', 'bwwl4n', 'sria3a', 'pbal7a', 'lbii9s', 'swwn8s', 'srwn6a', 'sgif4n', 'sbat3s', 'sgbg8p', 'swbv4a', 'srwu3n', 'bbak5a', 'lbib6p', 'lgay8p', 'bgaz8n', 'bgam8n', 'swbr8n', 'lgwr4s', 'bwbs5p', 'brbr9n', 'lway3n', 'prbn5p', 'pbaf5s', 'lradzs', 'sbie2n', 'lgwq7p', 'sbbh5a', 'pwbh3p', 'brajzp', 'lwwf5p', 'bwbz5a', 'bwbg3n', 'lwbl3n', 'bgbn7p', 'pgio6s', 'bgaf4s', 'lbit9p', 'lgac2n', 'lrbp6s', 'brwl7a', 'sgab3s', 'lgwz1p', 'prbn2s', 'pgij8a', 'lwbq6p', 'swbczp', 'bbii5a', 'pwwv4a', 'lwax7s', 'prao9p', 'pbwc1n', 'sbas6s', 'sbim6a', 'swiz7a', 'sbaf2s', 'brir7a', 'lgidzp', 'lgbv9a', 'bwiz2n', 'pwwn8p', 'pbia5p', 'lbbh6a', 'lwav9s', 'swwu1s', 'srbu6s', 'sgag3n', 'bwie3s', 'sgbi5p', 'brix8a', 'pwan5a', 'pgbj5p', 'bbwk5n', 'pgbb9n', 'sgib9s', 'sban1p', 'pbis6p', 'lbim7s', 'lbbb3s', 'brwkzs', 'sriq4n', 'pgaq4p', 'sbiy6a', 'pgao9p', 'lwal2n', 'pbanzs', 'pbam7n', 'lrwfzs', 'lgbj5s', 'sbwb2s', 'lwwz7a', 'lrae2n', 'pwbp8p', 'sgwi8n', 'srbf9n']
    print('Holdout Phrases:', holdout_phrases)

    all_training_images = get_image_list_2(args.data_root, training_speakers, holdout_phrases)
    all_test_images = get_image_list_2(args.data_root, test_speakers)

    hparams.add_hparam('all_images', all_training_images)
    hparams.add_hparam('all_test_images', all_test_images)

    print('Training on {} hours'.format(len(all_training_images) / (3600. * hparams.fps)))
    print('Validating on {} hours'.format(len(all_test_images) / (3600. * hparams.fps)))

    return log_dir, hparams

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("name", help="Name of the run and of the logging directory.")
    parser.add_argument("--data_root", help="Speaker folder path", required=True)

    parser.add_argument("-m", "--models_dir", type=str, default="synthesizer/saved_models/", help=\
        "Path to the output directory that will contain the saved model weights and the logs.")

    parser.add_argument("--mode", default="synthesis",
                        help="mode for synthesis of tacotron after training")
    
    parser.add_argument("--GTA", default="True",
                        help="Ground truth aligned synthesis, defaults to True, only considered "
							 "in Tacotron synthesis mode")
    parser.add_argument("--restore", type=bool, default=True,
                        help="Set this to False to do a fresh training")
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
    parser.add_argument('--num_test_batches', type=int, default=16)
    parser.add_argument('--checkpoint_path')

    args = parser.parse_args()
    print_args(args, parser)
    
    log_dir, hparams = prepare_run(args)

    tacotron_train(args, log_dir, hparams)
