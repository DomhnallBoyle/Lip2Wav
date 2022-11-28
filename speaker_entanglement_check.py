"""
Get videos of same content, different speakers
Synthesise with a neutral embedding
Pass synthesised audio through speaker encoder
Plot embeddings

If embeddings are clustered together = focusing on content
If embeddings are spread out = focusing on speaker and content (i.e. entangled)
"""
import argparse
import random
import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import redis
import umap

from audio_utils import get_audio_embeddings


def create_dataset(args):
    phrase = 'Whattimeisit'

    redis_server = redis.Redis(host=args.redis_host)

    # get video paths of the same content
    video_paths = []
    for i in range(redis_server.llen(args.pull_list_name)):
        video_path = redis_server.lindex(args.pull_list_name, i).decode('utf-8')
        if phrase in video_path:
            video_paths.append(video_path)
    random.shuffle(video_paths)

    output_directory = Path('SRAVI_test_videos_entanglement')
    output_directory.mkdir(exist_ok=True)
    for i, video_path in enumerate(video_paths[:50]):
        shutil.copyfile(video_path, output_directory.joinpath(f'{i}.mp4'))


def check_entanglement(args):
    all_embeddings = []
    for generated_audio_path in Path('SRAVI_test_videos_entanglement').glob(f'{args.test_directory_name}/*/generated_audio.wav'):
        with generated_audio_path.open('rb') as f:
            audio_embedding = get_audio_embeddings(f)[0]
            all_embeddings.append(audio_embedding)

    reducer = umap.UMAP()
    projected = reducer.fit_transform(all_embeddings)
    plt.scatter(projected[:, 0], projected[:, 1])
    plt.gca().set_aspect('equal', 'datalim')
    plt.title('Speaker Entanglement')
    plt.savefig('speaker_entanglement.png')


def main(args):
    f = {
        'create_dataset': create_dataset,
        'check_entanglement': check_entanglement
    }
    f[args.run_type](args)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    sub_parsers = parser.add_subparsers(dest='run_type')

    parser_1 = sub_parsers.add_parser('create_dataset')
    parser_1.add_argument('pull_list_name')
    parser_1.add_argument('redis_host')

    parser_2 = sub_parsers.add_parser('check_entanglement')
    parser_2.add_argument('test_directory_name')

    main(parser.parse_args())
