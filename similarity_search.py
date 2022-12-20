"""Create mappings between audio and lip movement embeddings

Then find the closest lip movement embedding, giving you the audio embedding
"""
import argparse
import multiprocessing
import os
import pickle
from pathlib import Path
from tqdm import tqdm

import lshashpy3 as h
import numpy as np
import redis

from audio_utils import get_audio_embeddings, get_rate_of_speech, preprocess_audio
from video_utils import get_lip_embeddings, get_video_frame, get_video_rotation, extract_audio

HASH_SIZE = 6
NUM_DIMS = 256
HASH_PATH = 'hash.npz'
MATRICES_PATH = 'weights.npz'

np.random.seed(1234)
lsh = h.LSHash(hash_size=HASH_SIZE,
               input_dim=NUM_DIMS,
               storage_config={'dict': None},
               matrices_filename=MATRICES_PATH,
               hashtable_filename=HASH_PATH,
               overwrite=True)


def get_audio(video_path, audio_preprocessing=False, use_old_ffmpeg=False):
    audio_file = extract_audio(video_path, use_old_ffmpeg=use_old_ffmpeg)
    if audio_preprocessing:
        preprocessed_audio_file = preprocess_audio(audio_file)
        audio_file.close()
        audio_file = preprocessed_audio_file

    return audio_file


def process(args, process_id, video_paths):
    mappings = []
    for video_path in tqdm(video_paths):
        audio_file = get_audio(video_path, args.audio_preprocessing, args.use_old_ffmpeg)

        audio_embeddings = get_audio_embeddings(audio_file)
        lip_embeddings = get_lip_embeddings(video_path)
        if not audio_embeddings or not lip_embeddings:
            continue

        extra_data = (video_path,)
        rate_of_speech = None
        if args.rate_of_speech:
            # approximate words per second using DeepSpeech and video length
            try:
                rate_of_speech = get_rate_of_speech(audio_file.name, tmp_audio_path=f'/tmp/ros_vad_{process_id}.wav')
                audio_file.close()
            except Exception as e:
                print(e)
                audio_file.close()
                continue
        extra_data += (rate_of_speech,)

        lip_embeddings = np.asarray(lip_embeddings)
        mappings.append([lip_embeddings[0], extra_data])

    return mappings


def generate_mapping(args):
    for p in [MATRICES_PATH, HASH_PATH]:
        if Path(p).exists():
            os.remove(p)

    redis_server = redis.Redis(host=args.redis_host, port=args.redis_port)

    video_paths = []
    print('Collecting video paths...')
    for i in tqdm(list(range(redis_server.llen(args.pull_list_name)))):
        video_path = redis_server.lindex(args.pull_list_name, i).decode('utf-8')
        if not Path(video_path).exists():
            continue

        video_paths.append(video_path)

    if not video_paths:
        print('No video paths to process')
        exit()

    tasks = []
    num_per_process = len(video_paths) // args.num_processes
    for i in range(args.num_processes):
        start = i * num_per_process
        end = start + num_per_process
        tasks.append([args, i, video_paths[start: end]])

    mappings = []
    print('Generating mappings...')
    with multiprocessing.Pool(processes=args.num_processes) as p:
        results = p.starmap(process, tasks)
        for result in results:
            mappings.extend(result)

    print('Num Mappings:', len(mappings))
    for lip_embedding, extra_data in mappings:
        # create mapping between lip embedding and the video path
        lsh.index(lip_embedding, extra_data=extra_data)

    # save hash table and mapping to disk
    lsh.save()


def query(video_path, by_content=False, by_rate_of_speech=False, audio_preprocessing=False, exclude_speaker=False,
          use_old_ffmpeg=False):
    if by_content:
        # experiment to query by content
        video_path = Path(video_path)
        content = video_path.stem.split('_')[0]

        if by_rate_of_speech:

            audio_file = get_audio(video_path, audio_preprocessing, use_old_ffmpeg)
            try:
                rate_of_speech = get_rate_of_speech(audio_path=audio_file.name)
                audio_file.close()
            except Exception as e:
                print(e)
                audio_file.close()
                return

            best_result = None
            best_ros_diff = np.inf
            for ((vec, extra_data), distance) in lsh.query([0] * 256):
                video_path, rate_of_speech_q = extra_data
                content_q = Path(video_path).stem.split('_')[0]
                if content == content_q:
                    ros_diff = round(abs(rate_of_speech - rate_of_speech_q), 2)
                    if ros_diff < best_ros_diff:
                        best_ros_diff = ros_diff
                        best_result = extra_data, distance

            return best_result
    else:
        lip_embeddings = get_lip_embeddings(video_path)
        if not lip_embeddings:
            return

        exclude_speaker = Path(video_path).parents[0].name if exclude_speaker else None

        if by_rate_of_speech:
            audio_file = get_audio(video_path, audio_preprocessing, use_old_ffmpeg)
            try:
                rate_of_speech = get_rate_of_speech(audio_path=audio_file.name)
                audio_file.close()
            except Exception as e:
                print(e)
                audio_file.close()
                return

            query_results = lsh.query(lip_embeddings[0])  # defaults to return all results ranked by distance
            for ((vec, extra_data), distance) in query_results:
                video_path_q, rate_of_speech_q = extra_data
                if exclude_speaker and exclude_speaker == Path(video_path_q).parents[0].name:
                    # don't select the excluded speaker video path
                    continue
                if rate_of_speech == rate_of_speech_q or round(abs(rate_of_speech - rate_of_speech_q), 2) == 0.01:
                    # within at least 0.01 of a difference

                    return extra_data, distance

        # fallback to just returning the top result
        query_results = lsh.query(lip_embeddings[0])
        for ((vec, extra_data), distance) in query_results:
            video_path_q, rate_of_speech_q = extra_data
            if exclude_speaker and exclude_speaker == Path(video_path_q).parents[0].name:
                continue

            return extra_data, distance


def _query(args):
    extra_data, distance = query(args.video_path, args.by_rate_of_speech, args.audio_preprocessing, args.use_old_ffmpeg)
    print(extra_data, distance)


def main(args):
    f = {
        'generate_mapping': generate_mapping,
        'query': _query
    }
    f[args.run_type](args)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    sub_parsers = parser.add_subparsers(dest='run_type')

    parser_1 = sub_parsers.add_parser('generate_mapping')
    parser_1.add_argument('pull_list_name')
    parser_1.add_argument('--redis_host', default='redis')
    parser_1.add_argument('--redis_port', default=6379)
    parser_1.add_argument('--audio_preprocessing', action='store_true')
    parser_1.add_argument('--use_old_ffmpeg', action='store_true')
    parser_1.add_argument('--num_processes', type=int, default=5)
    parser_1.add_argument('--rate_of_speech', action='store_true')

    parser_2 = sub_parsers.add_parser('query')
    parser_2.add_argument('video_path')
    parser_2.add_argument('--by_rate_of_speech', action='store_true')
    parser_2.add_argument('--audio_preprocessing', action='store_true')
    parser_2.add_argument('--use_old_ffmpeg', action='store_true')

    main(parser.parse_args())
