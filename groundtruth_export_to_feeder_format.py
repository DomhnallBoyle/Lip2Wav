import argparse
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from asr import DeepSpeechASR
from video_utils import extract_audio


def main(args):
    videos_directory = Path(args.videos_directory)

    with Path(args.phrases_path).open('r') as f:
        phrases = [p.lower().strip() for p in f.read().splitlines()]

    deepspeech_asr = DeepSpeechASR(host=args.deepspeech_host)

    data = []
    for video_path in tqdm(list(videos_directory.glob('*.mp4'))):
        audio_file = extract_audio(video_path=str(video_path))
        predictions = deepspeech_asr.run(audio_path=audio_file.name)
        audio_file.close()
        if predictions is None:
            continue

        good_sample = False
        for pred in predictions:
            if pred in phrases:
                good_sample = True
                break
        if not good_sample:
            continue
        gt = pred

        data.append([str(video_path), gt, predictions])

    with videos_directory.joinpath('asr_phrase_recognition_results.csv').open('w') as f:
        for video_path, gt, predictions in data:
            f.write(f'{video_path},{gt},{predictions}\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('videos_directory')
    parser.add_argument('deepspeech_host')
    parser.add_argument('phrases_path')

    main(parser.parse_args())
