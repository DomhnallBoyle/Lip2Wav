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

from detectors import get_face_landmarks, get_mouth_frames_wrapper as get_mouth_frames, smooth_landmarks
from audio_utils import extract_mel_spectrogram, get_audio_embeddings, play_audio, preprocess_audio
from similarity_search import query as lip_query
from video_utils import convert_fps, extract_audio, get_fps, get_video_frames, get_video_rotation, run_cfe_cropper, \
    run_video_augmentation, show_frames


UP = "\x1B[4A"
CLR = "\x1B[0K"
print('\n\n')


def get_selection_weights(redis_server, pull_list_name, by_user, by_word):
    # for users, it assumes the video paths are in folders with the speaker ID as the folder name
    # for words, it assumes the video name is like {phrase}_{word}_{uuid}.mp4. 'sil' counts as a word
    counts = {}
    ids = []
    for index in range(redis_server.llen(pull_list_name)):
        video_path = redis_server.lindex(pull_list_name, index).decode('utf-8')

        if by_user:
            _id = video_path.split('/')[-2]
        elif by_word:
            _id = video_path.split('/')[-1].split('_')[1]

        counts[_id] = counts.get(_id, 0) + 1
        ids.append(_id)

    weight = 1. / len(counts)  # equal weight among ids
    selection_weights = [weight / counts[_id] for _id in ids]  # bigger weight = the more likely it will be picked
    assert len(selection_weights) == len(ids)

    return selection_weights


def get_speaker_and_content(video_path, speaker_id_index):
    # returns the speaker id and content of a video path
    video_path = Path(video_path)

    # e.g. ['123', 'XYZ', 'abc.mp4'], speaker_id_index = -3, content = XYZ_abc
    video_path_wo_suffix = video_path.with_suffix('')
    video_path_parts = video_path_wo_suffix.parts
    # speaker_id, content = video_path_parts[speaker_id_index], '_'.join(video_path_parts[speaker_id_index + 1:])
    speaker_id = video_path_parts[speaker_id_index]
    content = video_path_parts[-1].split('_')[0]

    return speaker_id, content


def generate_speaker_video_mapping(redis_server, pull_list_name, speaker_id_index=-2):
    # this mapping is for getting a random audio from the same speaker that doesn't have the same content as the video
    # used for extracting the speaker embedding
    mapping = {}
    for index in range(redis_server.llen(pull_list_name)):
        video_path = redis_server.lindex(pull_list_name, index).decode('utf-8')

        speaker_id, content = get_speaker_and_content(Path(video_path), speaker_id_index)

        speaker_content_mapping = mapping.get(speaker_id, {})
        content_videos = speaker_content_mapping.get(content, [])
        content_videos.append(str(video_path))
        speaker_content_mapping[content] = content_videos
        mapping[speaker_id] = speaker_content_mapping

    return mapping


def process_video(process_index, video_path, fps=None, is_training=False, is_training_data=False, output_directory=None,
                  augmentation_prob=0.5, video_augmentation=False, audio_preprocessing=False,
                  speaker_embedding_audio_file=None, use_old_ffmpeg=False, use_old_mouth_extractor=False,
                  use_perspective_warp=False, use_cfe_cropper=False, greyscale=False, blur_frames=False, cfe_host=None,
                  smooth_frames=False, debug=False):
    debug = debug and process_index == 0
    if debug:
        print('Processing', video_path)

    # only apply augmentation if training data and randomly at 50%
    apply_video_augmentation = is_training_data and random.random() < augmentation_prob

    if apply_video_augmentation and video_augmentation:
        if debug:
            print('Running video augmentation...')
        video_path = run_video_augmentation(video_path=video_path,
                                            new_video_path=f'/tmp/video_augmentation_output_{process_index}.mp4')

    if use_cfe_cropper:
        # the cfe converts the video to 25 FPS, returns black and white and does a perspective warp crop
        video_frames = run_cfe_cropper(video_path=video_path, host=cfe_host)
        if not video_frames:
            if debug:
                print('No mouth frames for', video_path)
            return

        if debug:
            # NOTE: usually the cropped videos are 1 frame more than the original
            print('Num frames between original and cropped', len(get_video_frames(video_path)), len(video_frames))
    else:
        # convert FPS if applicable
        if debug:
            print('Checking FPS conversion..')
        _fps = get_fps(video_path=video_path)
        if fps and fps != _fps:
            if debug:
                print('Running FPS conversion...')
            new_video_path = f'/tmp/video_fps_conversion_output_{process_index}_{is_training}_{is_training_data}.mp4'
            video_path = convert_fps(video_path=video_path, new_video_path=new_video_path, fps=fps)
            video_rotation = 0  # ffmpeg auto rotates the video via metadata
        else:
            fps = _fps
            video_rotation = get_video_rotation(video_path=video_path)

        video_frames = get_video_frames(video_path=video_path, rotation=video_rotation)
        if not video_frames:
            if debug:
                print('No video frames for', video_path)
            return

        if debug:
            show_frames(video_frames, delay=fps, title=f'{process_index} - Original')

        # find and crop mouth ROI
        detections = None
        if smooth_frames:
            detections = {}
            for i, frame in enumerate(video_frames):
                face_stats = get_face_landmarks(frame=frame)
                if not face_stats:
                    return
                face_coords, landmarks = face_stats
                detections[i] = {'c': face_coords, 'l': landmarks}
            assert len(detections) == len(video_frames)
            detections = smooth_landmarks(detections=detections)

        mouth_frames = []
        try:
            mouth_frames = get_mouth_frames(frames=video_frames, use_old_method=use_old_mouth_extractor,
                                            use_perspective_warp=use_perspective_warp, face_stats=detections)
        except Exception as e:
            print('Exception when getting mouth frames:', e)
        if not mouth_frames:
            if debug:
                print('No mouth frames for', video_path)
            return
        video_frames = mouth_frames

        if greyscale:
            # creates 2D frames i.e. (50 x 100), not (3 x 50 x 100) like in BGR
            video_frames = [cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) for frame in video_frames]

        if blur_frames:
            video_frames = [cv2.GaussianBlur(frame, (7, 7), 0) for frame in video_frames]

    if debug:
        show_frames(video_frames, delay=fps, title=f'{process_index} - Mouth Cropping')

    # extract audio from video
    if debug:
        print('Extracting audio...')
    audio_file = extract_audio(video_path=video_path, use_old_ffmpeg=use_old_ffmpeg)

    if output_directory:
        # write files for inference
        shutil.copyfile(video_path, str(output_directory.joinpath('original_video.mp4')))
        if speaker_embedding_audio_file is None:
            shutil.copyfile(audio_file.name, str(output_directory.joinpath('audio.wav')))
        else:
            shutil.copyfile(speaker_embedding_audio_file.name, str(output_directory.joinpath('audio.wav')))
        for i, frame in enumerate(video_frames):
            # save as .png for lossless compression
            cv2.imwrite(str(output_directory.joinpath(f'{i}.png')), frame)

    if debug:
        play_audio(audio_file.name)

    # preprocess audio
    if audio_preprocessing:
        if debug:
            print('Preprocessing audio...')
        preprocessed_audio_file = preprocess_audio(audio_file=audio_file)
        audio_file.close()
        audio_file = preprocessed_audio_file
        if output_directory and speaker_embedding_audio_file is None:
            # write preprocessed audio for inference
            shutil.copyfile(audio_file.name, str(output_directory.joinpath('audio_preprocessed.wav')))

    if debug:
        play_audio(audio_file.name)

    # extract speaker embeddings
    if debug:
        print('Getting speaker embeddings...')
    if speaker_embedding_audio_file is None:
        if debug: 
            print('Using audio file from video as embeddings')
        speaker_embedding_audio_file = audio_file
    else:
        if debug: 
            print('Using given audio file as embeddings')
        if audio_preprocessing:
            preprocessed_speaker_embedding_audio_file = preprocess_audio(audio_file=speaker_embedding_audio_file)
            speaker_embedding_audio_file.close()
            speaker_embedding_audio_file = preprocessed_speaker_embedding_audio_file
            if output_directory:
                shutil.copyfile(speaker_embedding_audio_file.name, str(output_directory.joinpath('audio_preprocessed.wav')))

    speaker_embeddings = get_audio_embeddings(audio_file=speaker_embedding_audio_file)
    if not speaker_embeddings:
        if debug:
            print('No speaker embeddings for', video_path)
        return
    speaker_embeddings = np.asarray(speaker_embeddings)

    if output_directory:
        # write embeddings for inference
        np.savez_compressed(str(output_directory.joinpath('ref.npz')), ref=speaker_embeddings)

    # extract mel-spectrogram
    if debug:
        print('Getting mel-spec...')
    mel_spec = extract_mel_spectrogram(audio_file=audio_file)

    speaker_embedding_audio_file.close()
    audio_file.close()

    if is_training:
        return video_frames, mel_spec, speaker_embeddings
    else:
        return not None


def process_wrapper(process_index, args, video_path):
    speaker_embedding_audio_file = None  # default to using the audio from the video as speaker embedding

    if args.speaker_content_mapping:
        # use a different audio content for the speaker embedding (same speaker)
        speaker_id, content = get_speaker_and_content(Path(video_path), args.speaker_id_index)

        random_content = random.choice(list(set(args.speaker_content_mapping[speaker_id].keys()) - {content}))
        assert content != random_content

        random_video_path = random.choice(args.speaker_content_mapping[speaker_id][random_content])
        assert str(video_path) != random_video_path
        speaker_embedding_audio_file = extract_audio(random_video_path, use_old_ffmpeg=args.use_old_ffmpeg)

        if args.debug:
            print('Same speaker diff content', process_index, video_path, random_video_path)

    if args.use_closest_audio_embedding:
        # use closest embedding by lips and ros - exclude speaker from selection
        results = lip_query(video_path=video_path, by_rate_of_speech=True, audio_preprocessing=args.audio_preprocessing,
                            exclude_speaker=True, use_old_ffmpeg=args.use_old_ffmpeg)
        if results is not None:
            closest_video_path = results[0][0]
            speaker_embedding_audio_file = extract_audio(closest_video_path)

            if args.debug:
                print('Closest audio embedding', video_path, results)

    return process_video(process_index, str(video_path),
                         fps=args.fps,
                         is_training=args.is_training,
                         is_training_data=args.is_training_data,
                         output_directory=None,
                         augmentation_prob=args.augmentation_prob,
                         video_augmentation=args.video_augmentation,
                         audio_preprocessing=args.audio_preprocessing,
                         speaker_embedding_audio_file=speaker_embedding_audio_file,
                         use_old_ffmpeg=args.use_old_ffmpeg,
                         use_old_mouth_extractor=args.use_old_mouth_extractor,
                         use_perspective_warp=args.use_perspective_warp,
                         use_cfe_cropper=args.use_cfe_cropper,
                         greyscale=args.greyscale,
                         cfe_host=args.cfe_host,
                         debug=args.debug)


def main(args):
    redis_server = redis.Redis(host=args.redis_host, port=args.redis_port)
    num_preprocessed_videos, num_failed_videos = 0, 0

    # check if redis key exists
    if not redis_server.exists(args.pull_list_name):
        raise Exception(f'Redis Key "{args.pull_list_name}" does not exist')

    selection_weights = None
    all_indexes = list(range(redis_server.llen(args.pull_list_name)))
    weighted_selection = args.weighted_user_selection or args.weighted_word_selection
    if weighted_selection:
        # grab selection weights for pulling video paths from the list
        selection_weights = get_selection_weights(redis_server, args.pull_list_name,
                                                  by_user=args.weighted_user_selection,
                                                  by_word=args.weighted_word_selection)
        assert len(all_indexes) == len(selection_weights)

    if args.clear_push_list:
        redis_server.delete(args.push_list_name)

    speaker_content_mapping = None
    if args.use_different_content_embedding:
        speaker_content_mapping = generate_speaker_video_mapping(redis_server, args.pull_list_name,
                                                                 args.speaker_id_index)
    args.speaker_content_mapping = speaker_content_mapping

    num_to_select = args.num_processes if args.num_processes else 1

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
            random_indexes = random.choices(all_indexes, k=num_to_select, weights=selection_weights)
            video_paths = [redis_server.lindex(args.pull_list_name, index) for index in random_indexes]
            if not video_paths:
                time.sleep(1)
                continue

            video_paths = [video_path.decode('utf-8') for video_path in video_paths]

            if args.num_processes:
                # run the process pool, collect the results and push to the queue
                # each process gets a video path to preprocess
                tasks = [[i, args, video_path] for i, video_path in enumerate(video_paths)]

                with multiprocessing.Pool(processes=args.num_processes) as pool:
                    results = pool.starmap(process_wrapper, tasks)
            else:
                results = [process_wrapper(0, args, video_paths[0])]

            for video_path, result in zip(video_paths, results):
                if result is None:
                    num_failed_videos += 1
                    continue

                # push onto list for the sampler
                binary_obj = pickle.dumps([video_path, *result])
                redis_server.rpush(args.push_list_name, binary_obj)  # pushes to tail of list

                num_preprocessed_videos += 1

            if not args.debug:
                stats = f'{UP}Num video paths in {args.pull_list_name}: {redis_server.llen(args.pull_list_name)}{CLR}\n' \
                        f'Num preprocessed videos: {num_preprocessed_videos}{CLR}\n' \
                        f'Num failed videos: {num_failed_videos}{CLR}\n' \
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
    parser.add_argument('--redis_host', default='redis')
    parser.add_argument('--redis_port', type=int, default=6379)
    parser.add_argument('--pull_list_name', default='feed_list')
    parser.add_argument('--push_list_name', default='preprocessed_list')
    parser.add_argument('--push_list_max', type=int, default=100)
    parser.add_argument('--use_old_ffmpeg', action='store_true')
    parser.add_argument('--num_processes', type=int)
    parser.add_argument('--fps', type=int)
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--clear_push_list', action='store_true')
    parser.add_argument('--weighted_user_selection', action='store_true')
    parser.add_argument('--weighted_word_selection', action='store_true')
    parser.add_argument('--use_old_mouth_extractor', action='store_true')
    parser.add_argument('--use_perspective_warp', action='store_true')
    parser.add_argument('--use_cfe_cropper', action='store_true')
    parser.add_argument('--greyscale', action='store_true')
    parser.add_argument('--cfe_host')
    parser.add_argument('--use_closest_audio_embedding', action='store_true')  # i.e. by lips and rate-of-speech
    parser.add_argument('--use_different_content_embedding', action='store_true')  # i.e. speaker audio from a different video of diff content
    parser.add_argument('--speaker_id_index', type=int, default=-2)  # i.e. <speaker_id>/<video_id>.mp4 = -2

    main(parser.parse_args())
