import ast
import random
import re
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import pandas as pd
from synthesizer.hparams import hparams, get_image_list, get_image_list_2
from synthesizer.train import tacotron_train
from utils.argutils import print_args
import argparse
from pathlib import Path
import os
import numpy as np
import tensorflow as tf


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

    # if args.dataset == 'LRW':
    #     all_training_images = get_image_list('train', args.data_root)
    #     all_test_images = get_image_list('val', args.data_root)
    # elif args.dataset == 'GRID':
    #     all_speakers = [p.name for p in Path(args.data_root).glob('*') if p.is_dir()]
    #     test_speakers = ['s27', 's34']  # hold out speakers (man, woman) for testing (~2000 examples)
    #     training_speakers = list(set(all_speakers) - set(test_speakers))
    #     print('Num Total Speakers:', len(all_speakers))
    #     print('Num Training Speakers:', len(training_speakers))
    #
    #     # grab phrases from users in training to hold out for testing
    #     all_training_phrases = list([p.name for p in Path(args.data_root).glob('*/*')
    #                                  if p.is_dir() and p.parts[-2] in training_speakers])
    #     print('Total Num Training Phrases:', len(all_training_phrases))
    #     random.seed(hparams.tacotron_data_random_state)
    #     random.shuffle(all_training_phrases)
    #     # holdout_phrases = all_training_phrases[:100]  # 100 examples
    #     holdout_phrases = ['swag7p', 'bwiq7s', 'sgia2n', 'prwv4p', 'lrab2s', 'bbwq8s', 'bwwl4n', 'sria3a', 'pbal7a', 'lbii9s', 'swwn8s', 'srwn6a', 'sgif4n', 'sbat3s', 'sgbg8p', 'swbv4a', 'srwu3n', 'bbak5a', 'lbib6p', 'lgay8p', 'bgaz8n', 'bgam8n', 'swbr8n', 'lgwr4s', 'bwbs5p', 'brbr9n', 'lway3n', 'prbn5p', 'pbaf5s', 'lradzs', 'sbie2n', 'lgwq7p', 'sbbh5a', 'pwbh3p', 'brajzp', 'lwwf5p', 'bwbz5a', 'bwbg3n', 'lwbl3n', 'bgbn7p', 'pgio6s', 'bgaf4s', 'lbit9p', 'lgac2n', 'lrbp6s', 'brwl7a', 'sgab3s', 'lgwz1p', 'prbn2s', 'pgij8a', 'lwbq6p', 'swbczp', 'bbii5a', 'pwwv4a', 'lwax7s', 'prao9p', 'pbwc1n', 'sbas6s', 'sbim6a', 'swiz7a', 'sbaf2s', 'brir7a', 'lgidzp', 'lgbv9a', 'bwiz2n', 'pwwn8p', 'pbia5p', 'lbbh6a', 'lwav9s', 'swwu1s', 'srbu6s', 'sgag3n', 'bwie3s', 'sgbi5p', 'brix8a', 'pwan5a', 'pgbj5p', 'bbwk5n', 'pgbb9n', 'sgib9s', 'sban1p', 'pbis6p', 'lbim7s', 'lbbb3s', 'brwkzs', 'sriq4n', 'pgaq4p', 'sbiy6a', 'pgao9p', 'lwal2n', 'pbanzs', 'pbam7n', 'lrwfzs', 'lgbj5s', 'sbwb2s', 'lwwz7a', 'lrae2n', 'pwbp8p', 'sgwi8n', 'srbf9n']
    #     print('Holdout Phrases:', holdout_phrases)
    #
    #     all_training_images = get_image_list_2(args.data_root, training_speakers, holdout_phrases)
    #     all_test_images = get_image_list_2(args.data_root, test_speakers)
    # elif args.dataset == 'SRAVI_extended':
    #     training_speakers = ['AP', 'DB', 'JMC', 'ML']
    #     test_speakers = ['RMC', 'LMQ']
    #     all_training_images = get_image_list_2(args.data_root, training_speakers)
    #     all_test_images = get_image_list_2(args.data_root, test_speakers)
    # elif args.dataset == 'SRAVI_local':
    #     # TODO: Filter by ASR results
    #
    #     data_captures = [
    #         'PAVA_V1',
    #         'SRAVI_2021_03_12',
    #         'SRAVI_Demo',
    #         'SRAVI_Extended',
    #         'SRAVI_Recognition_Demo_Front',
    #         'SRAVI_V2'
    #     ]
    #     excluded_speakers = [
    #         'dev1',  # 1 sample, no lm
    #         'dev2',  # most samples no lm
    #         'd.f.mcauley@qub.ac.uk'  # pose angle, quiet audio
    #         'michael1@wandlebury.co.uk'  # starts with SRAVI
    #         'x100',  # no sound, wrong labels
    #         'dev4',  # dev videos, no lm or sound
    #         'sravi',  # mixture of people, patients and SRAVI at start
    #         'SRAVI_V2/michael@tormagna.com',  # dev videos
    #         'SRAVI_2021_03_12/jack.mcdonough@liopa.co.uk',  # poor sound
    #         'SRAVI_2021_03_12/domhnall.boyle@liopa.co.uk'  # poor/no sound
    #     ]
    #     test_speakers = [
    #         'SRAVI_V2/liam@liopa.ai',  # dependent speaker (has some samples in training)
    #         'dwhstewart@gmail.com',  # independent speaker (not in training) - male
    #         'helen.bear@liopa.co.uk',  # independent speaker - female
    #         'chimetesting53@gmail.com'  # independent speaker - over enunciating
    #     ]
    #
    #     """
    #     Training samples - samples within the top 1-3 ranks that aren't in the test speaker set
    #         - Excluded training samples - no predictions or not in the top 3 predictions or in the excluded speakers set
    #     Test samples - samples within top 1 prediction that are in the test speakers set
    #     """
    #
    #     def load_good_recognition_results(csv_path):
    #         good_samples = []
    #         with csv_path.open('r') as f:
    #             for line in f.read().splitlines():
    #                 video_path, groundtruth, predictions = re.match(r'(.+),(.+),(\[.*\])', line).groups()
    #                 speaker_id = video_path.split('/')[-2]
    #                 predictions = ast.literal_eval(predictions)
    #                 if not predictions:
    #                     continue
    #                 is_r1 = groundtruth == predictions[0]  # is rank 1 accuracy?
    #                 if groundtruth in predictions:  # if in top 3 predictions, it's a good sample
    #                     good_samples.append([speaker_id, video_path, is_r1])
    #         good_samples_df = pd.DataFrame(data=good_samples, columns=['Speaker ID', 'Video Path', 'Is R1'])
    #
    #         return good_samples_df
    #
    #     training_videos, test_videos = [], []
    #     for data_capture in data_captures:
    #         dc_path = Path(args.videos_root).joinpath(data_capture)
    #
    #         # grab the results of the DNN and ASR phrase recognition on the SRAVI data
    #         dnn_recognition_results_path = dc_path.joinpath('dnn_phrase_recognition_results.csv')
    #         if not dnn_recognition_results_path.exists():
    #             print(f'{data_capture} - no DNN results')
    #             continue
    #         dnn_good_samples_df = load_good_recognition_results(dnn_recognition_results_path)
    #
    #         asr_recognition_results_path = dc_path.joinpath('asr_phrase_recognition_results.csv')
    #         if not asr_recognition_results_path.exists():
    #             print(f'{data_capture} - no ASR results')
    #             continue
    #         asr_good_samples_df = load_good_recognition_results(asr_recognition_results_path)
    #
    #         # good samples are those that have good DNN and ASR results i.e. R1-R3 in DNN and ASR results
    #         good_samples = []
    #         for index, row in dnn_good_samples_df.iterrows():
    #             if row['Video Path'] in asr_good_samples_df['Video Path']:
    #                 good_samples.append([row['Speaker ID'], row['Video Path'], row['Is R1']])
    #         good_samples_df = pd.DataFrame(data=good_samples, columns=['Speaker ID', 'Video Path', 'Is R1'])
    #
    #         # loop through the speakers in the data capture
    #         for speaker_path in dc_path.glob('*'):
    #             if not speaker_path.is_dir():
    #                 continue
    #
    #             speaker_id = speaker_path.name
    #             speaker_id_dc = f'{speaker_path.parts[-2]}/{speaker_id}'
    #
    #             if speaker_id in excluded_speakers or speaker_id_dc in excluded_speakers:
    #                 continue
    #
    #             # get any ignored video paths
    #             ignored_path = speaker_path.joinpath('ignore.txt')
    #             ignored_paths = []
    #             if ignored_path.exists():
    #                 with open(ignored_path, 'r') as f:
    #                     ignored_paths = f.read().splitlines()
    #
    #             # get the sub sample set
    #             sub_samples_df = good_samples_df[
    #                 (good_samples_df['Speaker ID'] == speaker_id) &
    #                 (~good_samples_df['Video Path'].isin(ignored_paths))  # NOT
    #             ]
    #
    #             if speaker_id in test_speakers or speaker_id_dc in test_speakers:
    #                 # test samples are in the test speaker set, have R1 accuracy and are not an ignored path
    #                 test_speaker_samples = sub_samples_df[sub_samples_df['Is R1'] == True]
    #                 test_videos.extend([row['Video Path'] for index, row in test_speaker_samples.iterrows()])
    #             else:
    #                 # training samples are R1-R3 and are not an ignored path
    #                 training_videos.extend([row['Video Path'] for index, row in sub_samples_df.iterrows()])
    #
    #     def get_image_paths(video_paths):
    #         image_paths = []
    #         for video_path in video_paths:
    #             preprocessed_dirname = '/'.join(video_path.split('/')[-3:]).replace('.mp4', '')
    #             preprocessed_location = Path(args.data_root).joinpath(preprocessed_dirname)
    #             if preprocessed_location.exists():
    #                 if not all([preprocessed_location.joinpath(p).exists() for p in ['mels.npz', 'ref.npz']]):
    #                     continue  # make sure mels and ref exist
    #                 image_paths.extend([str(p) for p in preprocessed_location.glob('*.jpg')])
    #         return image_paths
    #
    #     print('Num Training Videos:', len(training_videos))
    #     print('Num Test Videos:', len(test_videos))
    #
    #     # show distribution of speakers in the training set
    #     training_speaker_counts = {}
    #     for video_path in training_videos:
    #         speaker_id = video_path.split('/')[-2]
    #         training_speaker_counts[speaker_id] = training_speaker_counts.get(speaker_id, 0) + 1
    #     print('Training sample counts:', sorted(training_speaker_counts.items(), key=lambda x: x[1]))
    #
    #     all_training_images = get_image_paths(training_videos)
    #     all_test_images = get_image_paths(test_videos)
    #
    #     with open('train_video_paths.txt', 'w') as f:
    #         for video_path in training_videos:
    #             video_path = '/'.join(video_path.split('/')[-3:]).replace('.mp4', '')
    #             f.write(f'{video_path}\n')
    #
    #     with open('test_video_paths.txt', 'w') as f:
    #         for video_path in test_videos:
    #             video_path = '/'.join(video_path.split('/')[-3:]).replace('.mp4', '')
    #             f.write(f'{video_path}\n')
    # else:
    #     print(f'Incorrect dataset:', args.dataset)
    #     exit()

    hparams.set_hparam('img_height', args.image_height)
    hparams.set_hparam('img_width', args.image_width)
    # hparams.add_hparam('all_images', all_training_images)
    # hparams.add_hparam('all_test_images', all_test_images)
    hparams.set_hparam('tacotron_batch_size', args.batch_size)
    args.num_test_batches = args.num_test_samples / args.batch_size

    # print('Training on {} hours'.format(len(all_training_images) / (3600. * hparams.fps)))
    # print('Validating on {} hours'.format(len(all_test_images) / (3600. * hparams.fps)))

    return log_dir, hparams


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("name", help="Name of the run and of the logging directory.")
    # parser.add_argument("--data_root", help="Speaker folder path", required=True)

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
    parser.add_argument('--num_test_samples', type=int, default=512)
    parser.add_argument('--checkpoint_path')
    parser.add_argument('dataset', choices=['LRW', 'GRID', 'SRAVI'])
    parser.add_argument('--videos_root')
    parser.add_argument('--apply_augmentation', action='store_true')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--image_height', type=int, default=128)  # face ROI
    parser.add_argument('--image_width', type=int, default=128)
    parser.add_argument('--training_sample_pool_location', default='/tmp/training_sample_pool')
    parser.add_argument('--val_sample_pool_location', default='/tmp/val_sample_pool')

    args = parser.parse_args()
    log_dir, hparams = prepare_run(args)
    print_args(args, parser)

    tacotron_train(args, log_dir, hparams)
