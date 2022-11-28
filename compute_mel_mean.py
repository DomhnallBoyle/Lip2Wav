"""
Compute mel-mean over training videos
Used for training, see 'Speaker Disentanglement' Paper
"""
import argparse
import multiprocessing
import tempfile

import redis
import numpy as np
from tqdm import tqdm

from audio_encoder.audio import preprocess_wav
from audio_utils import preprocess_audio
from synthesizer import audio as sa, hparams
from video_utils import extract_audio

NUM_MEL_CHANNELS = 80


def process(video_paths):
    num_video_paths = len(video_paths)
    mel_means = np.zeros((num_video_paths, NUM_MEL_CHANNELS))
    num_frames = np.zeros((num_video_paths, 1))

    for i, video_path in enumerate(tqdm(video_paths)):
        audio = extract_audio(video_path=video_path)  # extract audio from video
        preprocessed_audio = preprocess_audio(audio_file=audio)  # denoise and normalise
        wav = preprocess_wav(preprocessed_audio.name)  # trim silences
        mel_spec = sa.melspectrogram(wav, hparams.hparams).T  # convert to mel-spec

        audio.close()
        preprocessed_audio.close()

        mel_means[i] = mel_spec.mean(axis=0)
        num_frames[i] = len(mel_spec)

    return mel_means, num_frames


def main(args):
    redis_server = redis.Redis(args.redis_host)

    num_video_paths = redis_server.llen(args.pull_list_name)
    video_paths = [redis_server.lindex(args.pull_list_name, i).decode('utf-8')
                   for i in range(num_video_paths)]

    num_processes = 5
    num_videos_per_process = num_video_paths // num_processes
    tasks = []
    for i in range(num_processes):
        start = i * num_videos_per_process
        end = start + num_videos_per_process
        tasks.append([video_paths[start:end]])

    mel_means, num_frames = None, None
    with multiprocessing.Pool(processes=num_processes) as p:
        results = p.starmap(process, tasks)
        for _mel_means, _num_frames in results:
            if mel_means is None and num_frames is None:
                mel_means = _mel_means
                num_frames = _num_frames
            else:
                mel_means = np.vstack(_mel_means)
                num_frames = np.vstack(_num_frames)

    mel_mean = (num_frames * mel_means).sum(axis=0) / num_frames.sum()

    np.save('mel_mean.npy', mel_mean)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('redis_host')
    parser.add_argument('pull_list_name')

    main(parser.parse_args())
