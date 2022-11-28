import argparse
import math
import random
import pickle
import time

import matplotlib.pyplot as plt
import numpy as np
import redis

from sample_pool import SamplePool
from synthesizer import hparams as hp
from video_utils import run_frame_augmentation, show_frames

NUM_TIMESTEPS = hp.hparams.T
HALF_NUM_TIMESTEPS = NUM_TIMESTEPS // 2
FPS = hp.hparams.fps
SAMPLE_RATE = hp.hparams.sample_rate
MEL_STEP_SIZE = hp.hparams.mel_step_size
MEL_HOP_SIZE = hp.hparams.hop_size


def get_window(frames, center_id=None, lrw=False):
    # function selects a window of 25 frames
    # makes sure that selected window is within the bounds of the video
    # has functionality to select windows at the start and the end of the video
    # which is important for non-speech silence generation i.e. not lrw
    num_video_frames = len(frames)
    if num_video_frames < NUM_TIMESTEPS:
        return

    if not center_id:
        center_id = random.randint(0, num_video_frames - 1)  # inclusive

    if lrw:
        if NUM_TIMESTEPS % 2:
            window_ids = range(center_id - HALF_NUM_TIMESTEPS, center_id + HALF_NUM_TIMESTEPS + 1)
        else:
            window_ids = range(center_id - HALF_NUM_TIMESTEPS, center_id + HALF_NUM_TIMESTEPS)
        start = center_id - HALF_NUM_TIMESTEPS
    else:
        if center_id < HALF_NUM_TIMESTEPS:
            start = 0
            end = NUM_TIMESTEPS
        elif center_id > (num_video_frames - HALF_NUM_TIMESTEPS):
            start = (num_video_frames - NUM_TIMESTEPS) + 1
            end = num_video_frames + 1
        else:
            start = center_id - HALF_NUM_TIMESTEPS
            end = center_id + HALF_NUM_TIMESTEPS + 1

        attempts = 2
        while attempts != 0:
            window_ids = range(start, end)
            assert len(window_ids) == NUM_TIMESTEPS
            if list(window_ids)[-1] >= num_video_frames:
                start -= 1
                end -= 1
                attempts -= 1
            else:
                break

    try:
        window = [frames[i] for i in window_ids]
    except IndexError:
        return

    assert len(window) == NUM_TIMESTEPS

    return window, start


def crop_audio_window(spec, start_frame_id):
    # estimate total number of frames from spec (num_features, T)
    # num_frames = (T x hop_size * fps) / sample_rate
    total_num_frames = int((spec.shape[0] * MEL_HOP_SIZE * FPS) / SAMPLE_RATE)
    start_idx = int(spec.shape[0] * start_frame_id / float(total_num_frames))
    end_idx = start_idx + MEL_STEP_SIZE

    return spec[start_idx: end_idx, :]


def main(args):
    redis_server = redis.Redis(host=args.redis_host, port=args.redis_port)
    sample_pool = SamplePool(location=args.sample_pool_location, redo=args.redo_sample_pool)

    while True:
        print(f'No. processed objects in list: {redis_server.llen(args.pull_list_name)}\r', end='', flush=True)

        # this is selecting from the head of the list which contains objects preprocessed at random
        preprocessed_obj = redis_server.lpop(args.pull_list_name)
        if not preprocessed_obj:
            time.sleep(1)
            continue

        video_path, video_frames, mel_spec, speaker_embeddings = pickle.loads(preprocessed_obj)
        _class = None
        if args.group_by_user:
            _class = video_path.split('/')[-2]
        elif args.group_by_word:
            _class = video_path.split('/')[-1].split('_')[1]

        # only apply time masking to training data
        mean_frame = np.mean(video_frames, axis=0) if args.use_time_masking and args.is_training_data else None

        speaker_embedding = speaker_embeddings[0].astype(np.float32)

        for _ in range(5):
            # grab video window
            window = get_window(video_frames, lrw=args.lrw)
            if not window:
                continue
            video_frames_window, start_frame_id = window

            # crop mel spec
            mel_spec_window = crop_audio_window(mel_spec.T, start_frame_id).astype(np.float32)
            if mel_spec_window.shape[0] != MEL_STEP_SIZE:
                continue

            # time masking augmentation - https://arxiv.org/pdf/2202.13084v1.pdf
            if mean_frame is not None:
                mask_duration_secs = np.random.uniform(0, 0.4)
                mask_duration_frames = math.ceil(NUM_TIMESTEPS * mask_duration_secs)  # choose num frames to mask
                frame_index = random.randint(0, NUM_TIMESTEPS - mask_duration_frames)  # choose start index of mask
                for i in range(frame_index, frame_index + mask_duration_frames):
                    video_frames_window[i] = mean_frame

            # only apply augmentation if training data and randomly at 50%
            apply_video_augmentation = args.is_training_data and random.random() < args.augmentation_prob

            # apply augmentation to window
            if apply_video_augmentation and args.frame_augmentation:
                video_frames_window = run_frame_augmentation(video_frames_window, method='mouth')

            if args.debug:
                show_frames(video_frames_window, 25, 'Sample Window Frames')

            video_frames_window = (np.asarray(video_frames_window) / 255.).astype(np.float32)

            sample = [video_path, video_frames_window, mel_spec_window, speaker_embedding, len(mel_spec_window)]
            sample_pool.write(sample, _class=_class, max_size=args.max_sample_pool_size)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('sample_pool_location')
    parser.add_argument('max_sample_pool_size', type=int)
    parser.add_argument('--redo_sample_pool', action='store_true')
    parser.add_argument('--is_training_data', action='store_true')
    parser.add_argument('--augmentation_prob', type=float, default=0.5)
    parser.add_argument('--frame_augmentation', action='store_true')
    parser.add_argument('--use_time_masking', action='store_true')
    parser.add_argument('--redis_host', default='redis')
    parser.add_argument('--redis_port', type=int, default=6379)
    parser.add_argument('--pull_list_name', default='preprocessed_list')
    parser.add_argument('--lrw', action='store_true')
    parser.add_argument('--group_by_user', action='store_true')
    parser.add_argument('--group_by_word', action='store_true')
    parser.add_argument('--debug', action='store_true')

    main(parser.parse_args())
