"""
Adds video paths onto a Redis Queue for synthesiser training
"""
import argparse
import ast
import re
from pathlib import Path

import numpy as np
import pandas as pd
import redis


def load_good_recognition_results(csv_path):
    # returns all video paths that were in the top 3 predictions
    # also returns info on whether R1 or not
    good_samples = []
    total = 0
    if not csv_path.exists():
        return None, 0

    with csv_path.open('r') as f:
        for line in f.read().splitlines():
            video_path, groundtruth, predictions = re.match(r'(.+),(.+),(\[.*\])', line).groups()
            speaker_id = video_path.split('/')[-2]
            predictions = ast.literal_eval(predictions)
            if not predictions:
                continue
            is_r1 = groundtruth == predictions[0]  # is rank 1 accuracy?
            if groundtruth in predictions:  # if in top 3 predictions, it's a good sample
                good_samples.append([speaker_id, video_path, is_r1])
            total += 1
    good_samples_df = pd.DataFrame(data=good_samples, columns=['Speaker ID', 'Video Path', 'Is R1'])

    return good_samples_df, total


def main(args):
    videos_root = Path(args.videos_root)
    dataset = args.dataset
    redis_server = redis.Redis(host=args.redis_host, port=args.redis_port)

    if dataset == 'GENERIC':
        method = 'glob'
        if args.recursive:
            method = 'r' + method
        video_paths = list(getattr(videos_root, method)(args.videos_glob_path))
        if args.seed:
            np.random.seed(args.seed)
        if args.num_samples:
            video_paths = np.random.choice(video_paths, args.num_samples, replace=False)
    elif dataset == 'LRS2':
        if not args.samples_text_file:
            video_paths = list(videos_root.glob(args.videos_glob_path))
        else:
            video_paths = []
            with open(args.samples_text_file, 'r') as f:
                samples = f.read().splitlines()
            for sample in samples:
                video_path = videos_root.joinpath(f'{sample}.mp4')
                if not video_path.exists():
                    continue
                video_paths.append(video_path)
    elif dataset == 'SRAVI':
        # requires videos root to contain data captures with speaker directories
        # or just the speaker directories
        video_paths = []

        if args.data_captures:
            data_capture_paths = [videos_root.joinpath(data_capture) for data_capture in args.data_captures]
        else:
            data_capture_paths = [videos_root]

        included_speakers = args.included_speakers
        excluded_speakers = args.excluded_speakers

        for data_capture_path in data_capture_paths:
            # grab VSR results
            dnn_recognition_results_path = data_capture_path.joinpath('dnn_phrase_recognition_results.csv')
            if not dnn_recognition_results_path.exists() and not args.filter_by_asr_only:
                print(f'{data_capture_path} - no VSR results')
                continue
            vsr_good_samples_df, vsr_total = load_good_recognition_results(dnn_recognition_results_path)

            # grab ASR results
            asr_recognition_results_path = data_capture_path.joinpath('asr_phrase_recognition_results.csv')
            if not asr_recognition_results_path.exists():
                print(f'{data_capture_path} - no ASR results')
                continue
            asr_good_samples_df, asr_total = load_good_recognition_results(asr_recognition_results_path)

            # good samples are those that have good VSR and ASR results i.e. R1-R3 in both
            good_samples = []
            if args.filter_by_asr_only:
                for index, row in asr_good_samples_df.iterrows():
                    good_samples.append([row['Speaker ID'], row['Video Path'], row['Is R1']])
            else:
                for index, row in vsr_good_samples_df.iterrows():
                    if row['Video Path'] in asr_good_samples_df['Video Path'].values:
                        good_samples.append([row['Speaker ID'], row['Video Path'], row['Is R1']])
            good_samples_df = pd.DataFrame(data=good_samples, columns=['Speaker ID', 'Video Path', 'Is R1'])

            usage_percentage = round((len(good_samples_df) / max(vsr_total, asr_total) * 100), 1)
            print(f'{data_capture_path.name}: using {usage_percentage}%')

            for speaker_path in data_capture_path.glob('*'):
                if not speaker_path.is_dir():
                    continue
                speaker_id = speaker_path.name
                speaker_id_dc = f'{speaker_path.parts[-2]}/{speaker_id}'

                if included_speakers and not \
                        (speaker_id in included_speakers or speaker_id_dc in included_speakers):
                    continue

                if excluded_speakers and \
                        (speaker_id in excluded_speakers or speaker_id_dc in excluded_speakers):
                    continue

                # get any ignored paths
                ignored_path = speaker_path.joinpath('ignore.txt')
                ignored_paths = []
                if ignored_path.exists():
                    with ignored_path.open('r') as f:
                        ignored_paths = f.read().splitlines()

                # training samples are R1-R3 and are not an ignored path
                sub_samples_df = good_samples_df[
                    (good_samples_df['Speaker ID'] == speaker_id) &
                    (~good_samples_df['Video Path'].isin(ignored_paths))  # NOT
                ]

                if args.use_r1_data:
                    # test samples are R1 accuracy and not an ignored path
                    sub_samples_df = sub_samples_df[sub_samples_df['Is R1'] == True]

                video_paths.extend([row['Video Path'] for index, row in sub_samples_df.iterrows()])
    elif dataset == 'GRID':
        video_paths = []

        included_speakers = args.included_speakers
        excluded_speakers = args.excluded_speakers

        for speaker_path in videos_root.glob('*'):
            if not speaker_path.is_dir():
                continue

            speaker_id = speaker_path.name
            if included_speakers and not (speaker_id in included_speakers):
                continue

            if excluded_speakers and (speaker_id in excluded_speakers):
                continue

            video_paths.extend([str(p) for p in speaker_path.glob('*.mpg')])
    else:
        print('Dataset doesn\'t exist')
        exit()

    if args.clear_push_list:
        redis_server.delete(args.push_list_name)

    print('\nNum video paths in list:', redis_server.llen(args.push_list_name))

    for video_path in video_paths:
        redis_server.rpush(args.push_list_name, str(video_path))

    print(f'Pushed {len(video_paths)} video paths to list {args.push_list_name}')
    print('Num video paths in list:', redis_server.llen(args.push_list_name))


def list_type(s):
    return s.split(',')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('videos_root')
    parser.add_argument('--redis_host', default='redis')
    parser.add_argument('--redis_port', type=int, default=6379)
    parser.add_argument('--push_list_name', default='feed_list')
    parser.add_argument('--clear_push_list', action='store_true')

    sub_parsers = parser.add_subparsers(dest='dataset')

    parser_1 = sub_parsers.add_parser('GENERIC')
    parser_1.add_argument('videos_glob_path', help='*/train/*.mp4')
    parser_1.add_argument('--num_samples', type=int)
    parser_1.add_argument('--seed', type=int, default=1234)
    parser_1.add_argument('--recursive', action='store_true')

    parser_2 = sub_parsers.add_parser('SRAVI')
    parser_2.add_argument('--data_captures', type=list_type)
    parser_2.add_argument('--included_speakers', type=list_type)
    parser_2.add_argument('--excluded_speakers', type=list_type)
    parser_2.add_argument('--use_r1_data', action='store_true')
    parser_2.add_argument('--filter_by_asr_only', action='store_true')

    parser_3 = sub_parsers.add_parser('GRID')
    parser_3.add_argument('--included_speakers', type=list_type)
    parser_3.add_argument('--excluded_speakers', type=list_type)

    parser_4 = sub_parsers.add_parser('LRS2')
    parser_4.add_argument('--videos_glob_path', help='*/*.mp4')
    parser_4.add_argument('--samples_text_file')

    main(parser.parse_args())
