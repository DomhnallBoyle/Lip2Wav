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

from audio_utils import get_audio_embeddings, preprocess_audio
from video_utils import get_lip_embeddings, extract_audio

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


def process(args, video_paths):
    mappings = []
    for video_path in tqdm(video_paths):
        audio_file = extract_audio(video_path, use_old_ffmpeg=args.use_old_ffmpeg)
        if args.audio_preprocessing:
            preprocessed_audio_file = preprocess_audio(audio_file)
            audio_file.close()
            audio_file = preprocessed_audio_file

        audio_embeddings = get_audio_embeddings(audio_file)
        lip_embeddings = get_lip_embeddings(video_path)
        if not audio_embeddings or not lip_embeddings:
            continue

        lip_embeddings = np.asarray(lip_embeddings)
        mappings.append([lip_embeddings[0], video_path])
        audio_file.close()

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

    tasks = []
    num_per_process = len(video_paths) // args.num_processes
    for i in range(args.num_processes):
        start = i * num_per_process
        end = start + num_per_process
        tasks.append([args, video_paths[start: end]])

    mappings = []
    print('Generating mappings...')
    with multiprocessing.Pool(processes=args.num_processes) as p:
        results = p.starmap(process, tasks)
        for result in results:
            mappings.extend(result)

    print('Num Mappings:', len(mappings))
    for lip_embedding, video_path in mappings:
        # create mapping between lip embedding and the video path
        lsh.index(lip_embedding, extra_data=video_path)

    # save hash table and mapping to disk
    lsh.save()


def query(video_path):
    lip_embeddings = get_lip_embeddings(video_path)
    if not lip_embeddings:
        return

    query_results = lsh.query(lip_embeddings[0], num_results=1)
    for ((vec, extra_data), distance) in query_results:
        return extra_data, distance


def _query(args):
    extra_data, distance = query(args.video_path)
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

    parser_2 = sub_parsers.add_parser('query')
    parser_2.add_argument('video_path')

    main(parser.parse_args())
