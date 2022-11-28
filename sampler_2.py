import argparse
import math
import multiprocessing
import random
import time
from pathlib import Path

import cv2
import numpy as np

from detectors import get_mouth_frames_wrapper, smooth_landmarks
from sampler import MEL_STEP_SIZE, NUM_TIMESTEPS, crop_audio_window, get_window
from sample_pool import SamplePool
from video_utils import get_video_frames, run_frame_augmentation


def process_samples(process_index, args):
    sample_pool = SamplePool(location=args.sample_pool_location)

    # get preprocessed samples
    processed_directory = Path(args.processed_directory)
    processed_sample_paths = list(processed_directory.glob('*.npz'))

    while True:
        random.shuffle(processed_sample_paths)

        for processed_sample_path in processed_sample_paths:
            try:
                start_time = time.time()

                video_path, detections, speaker_embeddings, mel_spec = \
                    np.load(processed_sample_path, allow_pickle=True)['sample']

                video_frames = get_video_frames(video_path=video_path, greyscale=args.greyscale)
                assert len(video_frames) == len(detections), f'{video_path}: {len(video_frames)} != {len(detections)}'

                if args.smooth_frames:
                    detections = smooth_landmarks(detections=detections, window_length=7)

                mouth_frames = get_mouth_frames_wrapper(video_frames,
                                                        use_old_method=False,
                                                        use_perspective_warp=args.use_perspective_warp,
                                                        face_stats=detections)

                speaker_embedding = speaker_embeddings[0].astype(np.float32)

                # only apply time masking to training data
                mean_frame = np.mean(mouth_frames, axis=0) if args.use_time_masking else None

                try:
                    center_ids = random.sample(range(0, len(video_frames)), 5)  # without replacement i.e. unique
                except ValueError:
                    # failed to select 5 ints, sample size may not be big enough
                    continue

                initial_time = time.time() - start_time
                generated_count, sample_time = 0, 0
                for center_id in center_ids:
                    start_time = time.time()

                    # grab video window
                    window = get_window(mouth_frames, center_id=center_id)
                    if not window:
                        continue
                    mouth_frames_window, start_frame_id = window

                    # crop mel spec
                    mel_spec_window = crop_audio_window(mel_spec.T, start_frame_id).astype(np.float32)
                    if mel_spec_window.shape[0] != MEL_STEP_SIZE:
                        continue

                    # time masking augmentation
                    # - https://arxiv.org/pdf/2202.13084v1.pdf
                    # - https://arxiv.org/pdf/2205.02058.pdf
                    if mean_frame is not None:
                        mask_duration_secs = np.random.uniform(0, 0.4)
                        mask_duration_frames = math.ceil(NUM_TIMESTEPS * mask_duration_secs)  # choose num frames to mask
                        frame_index = random.randint(0, NUM_TIMESTEPS - mask_duration_frames)  # choose start index of mask
                        for i in range(frame_index, frame_index + mask_duration_frames):
                            mouth_frames_window[i] = mean_frame

                    # only apply augmentation if training data and randomly at 50%
                    apply_frame_augmentation = args.frame_augmentation and random.random() < args.augmentation_prob

                    # apply augmentation to window
                    if apply_frame_augmentation:
                        mouth_frames_window = run_frame_augmentation(mouth_frames_window, method='mouth',
                                                                     intensity_aug=args.intensity_augmentation)

                    if args.normalise_frames:
                        # normalize frames to 0-255 uint8 dtype
                        mouth_frames_window = [
                            cv2.normalize(src=frame, dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
                            for frame in mouth_frames_window
                        ]

                    mouth_frames_window = (np.asarray(mouth_frames_window) / 255.).astype(np.float32)

                    if args.debug and process_index == 0:
                        for frame in mouth_frames_window:
                            cv2.imshow('Mouth', frame)
                            cv2.waitKey(50)

                    sample = [video_path, mouth_frames_window, mel_spec_window, speaker_embedding, len(mel_spec_window)]
                    sample_pool.write(sample, max_size=args.max_sample_pool_size)

                    generated_count += 1
                    sample_time += (time.time() - start_time)

                final_time = round(initial_time + sample_time, 2)
                print(f'Process {process_index} generated {generated_count} samples in {final_time}s')
            except Exception as e:
                print(f'Process {process_index} error: {e}')


def main(args):
    SamplePool(location=args.sample_pool_location, redo=args.redo_sample_pool)

    if args.debug:
        process_samples(0, args)
    else:
        tasks = [[i, args] for i in range(args.num_processes)]
        with multiprocessing.Pool(processes=args.num_processes) as pool:
            pool.starmap(process_samples, tasks)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('processed_directory')
    parser.add_argument('sample_pool_location')
    parser.add_argument('max_sample_pool_size', type=int)
    parser.add_argument('--redo_sample_pool', action='store_true')
    parser.add_argument('--frame_augmentation', action='store_true')
    parser.add_argument('--use_time_masking', action='store_true')
    parser.add_argument('--use_perspective_warp', action='store_true')
    parser.add_argument('--intensity_augmentation', action='store_true')
    parser.add_argument('--augmentation_prob', type=float, default=0.5)
    parser.add_argument('--num_processes', type=int, default=5)
    parser.add_argument('--greyscale', action='store_true')
    parser.add_argument('--smooth_frames', action='store_true')
    parser.add_argument('--normalise_frames', action='store_true')
    parser.add_argument('--debug', action='store_true')

    main(parser.parse_args())
