"""
Preprocesses videos and pushes them in a Redis Queue for feeding to the synthesiser
"""
import argparse
import multiprocessing
import os
import pickle
import random
import shutil
import time
from pathlib import Path

import cv2
import numpy as np
import redis

from detectors import get_mouth_frames
from audio_utils import extract_mel_spectrogram, get_audio_embeddings, play_audio, preprocess_audio
from video_utils import convert_fps, extract_audio, run_frame_augmentation, get_fps, get_video_frames, \
    get_video_rotation, show_frames, run_video_augmentation

UP = "\x1B[4A"
CLR = "\x1B[0K"
print('\n\n')


def get_selection_weights(redis_server, pull_list_name):
    speaker_counts = {}
    speaker_ids = []
    for index in range(redis_server.llen(pull_list_name)):
        video_path = redis_server.lindex(pull_list_name, index).decode('utf-8')
        speaker_id = video_path.split('/')[-2]
        speaker_counts[speaker_id] = speaker_counts.get(speaker_id, 0) + 1
        speaker_ids.append(speaker_id)
    speaker_weight = 1. / len(speaker_counts)
    selection_weights = [speaker_weight / speaker_counts[speaker_id]
                         for speaker_id in speaker_ids]
    assert len(selection_weights) == len(speaker_ids)

    return selection_weights


def process_video(process_index, video_path, fps=None, is_training=False, is_training_data=False, output_directory=None,
                  augmentation_prob=0.5, video_augmentation=False, frame_augmentation=False, audio_preprocessing=False,
                  audio_file=None, use_old_ffmpeg=False, debug=False):
    debug = debug and process_index == 0

    # only apply augmentation if training data and randomly at 50%
    apply_video_augmentation = is_training_data and random.random() < augmentation_prob

    if apply_video_augmentation and video_augmentation:
        video_path = run_video_augmentation(video_path=video_path,
                                            new_video_path=f'/tmp/video_augmentation_output_{process_index}.mp4')

    # convert FPS if applicable
    _fps = get_fps(video_path=video_path)
    if fps and fps != _fps:
        new_video_path = f'/tmp/video_fps_conversion_output_{process_index}_{is_training}_{is_training_data}.mp4'
        video_path = convert_fps(video_path=video_path, new_video_path=new_video_path, fps=fps)
        video_rotation = 0  # ffmpeg auto rotates the video via metadata
    else:
        fps = _fps
        video_rotation = get_video_rotation(video_path=video_path)

    video_frames = get_video_frames(video_path=video_path, rotation=video_rotation)

    if debug:
        show_frames(video_frames, delay=fps, title=f'{process_index} - Original')

    # apply full frame augmentation if applicable
    if apply_video_augmentation and frame_augmentation:
        video_frames = run_frame_augmentation(frames=video_frames, method='full')

    if debug:
        show_frames(video_frames, delay=fps, title=f'{process_index} - Augmentation')

    # find and crop mouth ROI
    mouth_frames = []
    for method in ['dlib']:
        try:
            mouth_frames = get_mouth_frames(frames=video_frames, method=method)
        except Exception as e:
            print(e)
            break
        if mouth_frames:
            break
    if not mouth_frames:  # if failed dlib and s3fd detectors
        return
    video_frames = mouth_frames

    if debug:
        show_frames(video_frames, delay=fps, title=f'{process_index} - Mouth Cropping')

    # extract audio from video
    if not audio_file:
        audio_file = extract_audio(video_path=video_path, use_old_ffmpeg=use_old_ffmpeg)

    if output_directory:
        # write files for inference
        shutil.copyfile(video_path, str(output_directory.joinpath('original_video.mp4')))
        shutil.copyfile(audio_file.name, str(output_directory.joinpath('audio.wav')))
        for i, frame in enumerate(video_frames):
            cv2.imwrite(str(output_directory.joinpath(f'{i}.jpg')), frame)

    if debug:
        play_audio(audio_file)

    # preprocess audio
    if audio_preprocessing:
        preprocessed_audio_file = preprocess_audio(audio_file=audio_file)
        audio_file.close()
        audio_file = preprocessed_audio_file
        if output_directory:
            # write preprocessed audio for inference
            shutil.copyfile(audio_file.name, str(output_directory.joinpath('audio_preprocessed.wav')))

    if debug:
        play_audio(audio_file)

    # extract speaker embeddings
    speaker_embeddings = get_audio_embeddings(audio_file=audio_file)
    if not speaker_embeddings:
        return
    speaker_embeddings = np.asarray(speaker_embeddings)

    if output_directory:
        # write embeddings for inference
        np.savez_compressed(str(output_directory.joinpath('ref.npz')), ref=speaker_embeddings)

    # extract mel-spectrogram
    mel_spec = extract_mel_spectrogram(audio_file=audio_file)

    audio_file.close()

    if is_training:
        return video_frames, mel_spec, speaker_embeddings
    else:
        return not None


def main(args):
    redis_server = redis.Redis(host=args.redis_host, port=args.redis_port)
    num_preprocessed_videos = 0

    # check if redis key exists
    if not redis_server.exists(args.pull_list_name):
        raise Exception(f'Redis Key "{args.pull_list_name}" does not exist')

    # grab selection weights for pulling video paths from the list
    selection_weights = get_selection_weights(redis_server, args.pull_list_name)
    all_indexes = list(range(redis_server.llen(args.pull_list_name)))
    assert len(all_indexes) == len(selection_weights)

    num_processes = args.num_processes

    if args.clear_push_list:
        redis_server.delete(args.push_list_name)

    try:
        while True:
            # wait for queue puller to catch up
            num_preprocessed_videos_in_queue = redis_server.llen(args.push_list_name)
            if num_preprocessed_videos_in_queue >= args.push_list_max:
                if args.debug:
                    redis_server.delete(args.push_list_name)
                time.sleep(1)
                continue

            # get the random indexes using selection weights (for imbalanced datasets)
            random_indexes = random.choices(all_indexes, k=num_processes, weights=selection_weights)
            video_paths = [redis_server.lindex(args.pull_list_name, index) for index in random_indexes]
            if not video_paths:
                time.sleep(1)
                continue
            video_paths = [video_path.decode('utf-8') for video_path in video_paths]

            # each process gets a video path to preprocess
            tasks = []
            for i, video_path in enumerate(video_paths):
                tasks.append([i, video_path, args.fps, args.is_training, args.is_training_data, None,
                              args.augmentation_prob, args.video_augmentation, args.frame_augmentation,
                              args.audio_preprocessing, args.use_old_ffmpeg, args.debug])

            # run the process pool, collect the results and push to the queue
            with multiprocessing.Pool(processes=num_processes) as pool:
                results = pool.starmap(process_video, tasks)
            for video_path, result in zip(video_paths, results):
                if result is None:
                    continue

                # push onto list for the sampler
                binary_obj = pickle.dumps([video_path, *result])
                redis_server.rpush(args.push_list_name, binary_obj)  # pushes to tail of list

                num_preprocessed_videos += 1

            stats = f'{UP}Num video paths in {args.pull_list_name}: {redis_server.llen(args.pull_list_name)}{CLR}\n' \
                    f'Num preprocessed videos: {num_preprocessed_videos}{CLR}\n' \
                    f'Num preprocessed videos in {args.push_list_name}: {redis_server.llen(args.push_list_name)}{CLR}\n'
            print(stats)

    except KeyboardInterrupt:
        pass

    cv2.destroyAllWindows()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--is_training', action='store_true')
    parser.add_argument('--is_training_data', action='store_true')
    parser.add_argument('--augmentation_prob', type=float, default=0.5)
    parser.add_argument('--audio_preprocessing', action='store_true')
    parser.add_argument('--video_augmentation', action='store_true')
    parser.add_argument('--frame_augmentation', action='store_true')
    parser.add_argument('--redis_host', default='redis')
    parser.add_argument('--redis_port', type=int, default=6379)
    parser.add_argument('--pull_list_name', default='feed_list')
    parser.add_argument('--push_list_name', default='preprocessed_list')
    parser.add_argument('--push_list_max', type=int, default=100)
    parser.add_argument('--use_old_ffmpeg', action='store_true')
    parser.add_argument('--num_processes', type=int, default=5)
    parser.add_argument('--fps', type=int)
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--clear_push_list', action='store_true')

    main(parser.parse_args())
