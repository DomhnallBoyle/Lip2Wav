import argparse
import os
import multiprocessing
import pickle
import random
import shutil
import sys
import traceback
import uuid
from pathlib import Path

import numpy as np
import redis
from tqdm import tqdm

from audio_utils import extract_mel_spectrogram, get_audio_embeddings, play_audio, preprocess_audio as denoise_audio
from detectors import get_face_landmarks, get_face_landmarks_fan, get_mouth_frames_perspective_warp_2, \
    get_mouth_frames_wrapper, smooth_landmarks
from preprocessor import generate_speaker_video_mapping, get_speaker_and_content
from synthesizer.hparams import hparams
from video_utils import convert_fps, extract_audio, get_fps, get_video_frames, \
    run_video_augmentation, save_video_frames, show_frames

SPEAKER_EMBEDDING_SAMPLE_RATE = 16000


def interpolate_2d(m, s):
    from scipy import interpolate

    x = np.linspace(0, 1, m.shape[0])
    y = np.linspace(0, 1, m.shape[1])
    f = interpolate.interp2d(y, x, m, kind='cubic')

    x2 = np.linspace(0, 1, s[0])
    y2 = np.linspace(0, 1, s[1])

    return f(y2, x2)


def process_video_2(process_index, video_path, speaker_embedding_audio_file, landmarks_directory, denoise=False, skip_mel_spec=False, debug=False):
    video_fps_conversion_path = f'/tmp/video_fps_conversion_output_{process_index}.mp4'

    # get audio from video (convert fps if required)
    if not skip_mel_spec:
        audio_file = extract_audio(video_path=convert_fps(video_path=video_path, new_video_path=video_fps_conversion_path, fps=hparams.fps), sr=hparams.sample_rate)
        if denoise:
            denoised_audio_file = denoise_audio(audio_file=audio_file, sr=hparams.sample_rate)
            audio_file.close()
            audio_file = denoised_audio_file
        if debug:
            play_audio(audio_file.name)

    # get video landmarks
    # NOTE: landmarks from method and landmarks from file produce same landmarks
    video_frames = get_video_frames(video_path=video_path)  # FAN requires RGB frames
    if landmarks_directory:
        landmarks_id = '/'.join(Path(video_path).parts[-3:]).replace('.mp4', '.pkl')
        landmarks_path = Path(landmarks_directory).joinpath(landmarks_id)
        with landmarks_path.open('rb') as f:
            landmarks = pickle.load(f)
    else:
        landmarks = []
        for frame in video_frames:
            try:
                landmarks.append(get_face_landmarks_fan(frame=frame)[1])
            except Exception as e:
                print('Failed to extract landmarks:', video_path, e)
                return
    assert len(landmarks) == len(video_frames)

    # get mouth frames
    try:
        mouth_frames = get_mouth_frames_perspective_warp_2(frames=video_frames, landmarks=landmarks, greyscale=True)[0]
    except Exception as e:
        print('Failed to get mouth frames:', video_path, e)
        return
    assert len(mouth_frames) == len(landmarks) == len(video_frames)
    if debug:
        show_frames(video_frames, 50, 'Video Frames')
        show_frames(mouth_frames, 50, 'Mouth Frames')

    # convert fps is applicable
    video_fps = get_fps(video_path=video_path)
    if video_fps != hparams.fps:
        mouth_frames_path = f'/tmp/mouth_frames_{process_index}.mp4'
        save_video_frames(video_frames=mouth_frames, video_path=mouth_frames_path, fps=video_fps, colour=False)
        mouth_frames = get_video_frames(video_path=convert_fps(video_path=mouth_frames_path, new_video_path=video_fps_conversion_path, fps=hparams.fps), greyscale=True)
    mouth_frames = np.asarray(mouth_frames).astype(np.uint8)
    assert mouth_frames.shape[1:] == (96, 96)

    # extract speaker embedding - was trained on 16kHz audios (Librispeech + Vox 1 & 2)
    # NOTE: this has a memory leak somewhere - on-going issue on github
    if denoise:
        denoised_speaker_embedding_audio_file = denoise_audio(audio_file=speaker_embedding_audio_file, sr=SPEAKER_EMBEDDING_SAMPLE_RATE)
        speaker_embedding_audio_file.close()
        speaker_embedding_audio_file = denoised_speaker_embedding_audio_file
    if debug:
        play_audio(speaker_embedding_audio_file.name)
    speaker_embeddings = get_audio_embeddings(audio_file=speaker_embedding_audio_file)
    if speaker_embeddings is None:
        raise Exception('Failed to get speaker embeddings:', video_path)
    speaker_embedding = np.asarray(speaker_embeddings[0]).astype(np.float32)
    speaker_embedding_audio_file.close()

    sample = [mouth_frames, speaker_embedding]

    # extract mel-spec (interpolate if required)
    # NOTE: there is some slight difference between mel-spec and video frame length - use interpolation to fix
    if not skip_mel_spec:
        mel_spec = extract_mel_spectrogram(audio_file=audio_file).T
        duration = len(mouth_frames) / hparams.fps
        if mel_spec.shape[0] != int(duration * 80):
            mel_spec = interpolate_2d(mel_spec, (int(duration * 80), mel_spec.shape[1]))
        assert mel_spec.shape[0] == int(duration * 80) == mouth_frames.shape[0] * 4, f'{mel_spec.shape[0]}, {int(duration * 80)}, {mouth_frames.shape[0] * 4}'
        mel_spec = mel_spec.astype(np.float32)
        audio_file.close()
        sample += [mel_spec]

    return sample


def process_video(process_index, video_path, save_video_path, output_directory, speaker_embedding_audio_file, fps, greyscale, 
                  interpolate, landmarks_directory, convert_fps_after_landmarks, use_fan_landmarks, debug):
    original_video_path = video_path
    original_fps = get_fps(video_path=video_path)
    video_fps_conversion_path = f'/tmp/video_fps_conversion_output_{process_index}.mp4'

    # convert fps if applicable
    video_rotation = None
    if original_fps != fps:
        video_path = convert_fps(video_path=video_path, new_video_path=video_fps_conversion_path, fps=fps)
        video_rotation = 0

    # extract and preprocess audio from video
    audio_file = extract_audio(video_path=video_path, sr=hparams.sample_rate)
    if debug:
        play_audio(audio_file.name)
    preprocessed_audio_file = preprocess_audio(audio_file=audio_file, sr=hparams.sample_rate)
    audio_file.close()
    audio_file = preprocessed_audio_file

    if landmarks_directory: 
        video_path = original_video_path
        video_rotation = None

    # get video frames
    greyscale = greyscale and not use_fan_landmarks  # only read in greyscale if not using fan landmarks
    video_frames = get_video_frames(video_path=video_path, rotation=video_rotation, greyscale=greyscale)
    if not video_frames: 
        if debug: 
            print('No video frames', video_path)
        return

    # using already processed landmarks
    all_landmarks = None
    if landmarks_directory: 
        landmarks_id = '/'.join(Path(video_path).parts[-3:]).replace('.mp4', '.pkl')
        landmarks_path = Path(landmarks_directory).joinpath(landmarks_id)
        if not landmarks_path.exists():
            if debug: 
                print('Landmarks path does not exist')
            return
        with open(str(landmarks_path), 'rb') as f:
            all_landmarks = pickle.load(f)
        assert len(video_frames) == len(all_landmarks)

    # use FAN landmark predictor
    if use_fan_landmarks:
        all_landmarks = []
        iterator = video_frames
        if debug: 
            iterator = tqdm(iterator)
        for frame in iterator: 
            try:
                landmarks = get_face_landmarks_fan(frame)[1]
            except IndexError as e:
                print('Failed to extract landmarks:', original_video_path, e) 
                return
            all_landmarks.append(landmarks)
        assert len(video_frames) == len(all_landmarks)

    # get mouth frames
    if all_landmarks is not None:
        try:
            mouth_frames = get_mouth_frames_perspective_warp_2(video_frames, all_landmarks, greyscale=True)[0]
        except Exception as e:
            return
    else:
        detections = {}
        for i, frame in enumerate(video_frames):
            results = get_face_landmarks(frame=frame)
            if results is None:
                return
            face_coords, landmarks = results
            detections[i] = {'c': face_coords, 'l': landmarks}
        try:
            detections = smooth_landmarks(detections, window_length=12)
        except Exception as e:
            return
        mouth_frames = get_mouth_frames_wrapper(video_frames, use_perspective_warp=True, face_stats=detections,
                                                reshape_height=96, reshape_width=96)

    # convert FPS of cropped mouth frames 
    if convert_fps_after_landmarks and original_fps != fps:
        video_path = f'/tmp/mouth_frames_{process_index}.mp4'

        # save frames to file
        save_video_frames(video_frames=mouth_frames, video_path=video_path, fps=original_fps, colour=False)

        # convert fps and reload mouth frames
        video_path = convert_fps(video_path=video_path, new_video_path=video_fps_conversion_path, fps=fps)
        mouth_frames = get_video_frames(video_path=video_path, greyscale=greyscale)
    
    mouth_frames = np.asarray(mouth_frames).astype(np.uint8)
    assert mouth_frames.shape[1:] == (96, 96)

    if debug:
        print('Mouth frames', mouth_frames.shape)
        show_frames(mouth_frames, 50, 'Mouth Frames')

    # preprocess audio for speaker embedding
    preprocessed_speaker_embedding_audio_file = preprocess_audio(audio_file=speaker_embedding_audio_file, sr=SPEAKER_EMBEDDING_SAMPLE_RATE)
    speaker_embedding_audio_file.close()
    speaker_embedding_audio_file = preprocessed_speaker_embedding_audio_file

    # extract speaker embedding - was trained on 16kHz audios (Librispeech + Vox 1 & 2)
    # NOTE: this has a memory leak somewhere - on-going issue on github
    speaker_embeddings = get_audio_embeddings(audio_file=speaker_embedding_audio_file)
    if speaker_embeddings is None:
        raise Exception(f'{process_index} failed to get speaker embeddings')
    speaker_embedding = np.asarray(speaker_embeddings[0]).astype(np.float32)

    # extract mel-spec
    mel_spec = extract_mel_spectrogram(audio_file=audio_file).T
    duration = len(mouth_frames) / hparams.fps
    while mel_spec.shape[0] != int(duration * 80):
        if interpolate:
            mel_spec = interpolate_2d(mel_spec, (int(duration * 80), mel_spec.shape[1]))
            interpolate = False
            continue
        print(f'{process_index}, {mel_spec.shape[0]} != {int(duration * 80)}, duration = {duration}')  # 80 per second
        return
    mel_spec = mel_spec.astype(np.float32)

    # play audio files
    if debug:
        play_audio(audio_file.name)
        play_audio(speaker_embedding_audio_file.name)

    # clean-up
    speaker_embedding_audio_file.close()
    audio_file.close()

    # change video path
    path_d = {
        '/shared/HDD': '/media/alex/Storage/Domhnall',
        '/shared/SSD': '/media/SSD'
    }
    for k, v in path_d.items():
        original_video_path = original_video_path.replace(k, v)

    # save to disk
    obj = [save_video_path, mouth_frames, speaker_embedding, mel_spec]
    save_name = f'{str(uuid.uuid4())}.npz'
    np.savez_compressed(str(output_directory.joinpath(save_name)), sample=obj)


def get_speaker_embedding_audio_file(args, video_path, debug):
    # speaker embedding audio should be sampled at 16kHz
    if args.speaker_content_mapping:
        # get speaker embedding of same speaker, different content
        speaker_id, content = get_speaker_and_content(Path(video_path), args.speaker_id_index)
        try:
            random_content = random.choice(list(set(args.speaker_content_mapping[speaker_id].keys()) - {content}))
            assert (content != random_content) and (content.split('_')[0] != random_content.split('_')[0])

            random_video_path = random.choice(args.speaker_content_mapping[speaker_id][random_content])
            assert video_path != random_video_path
            if debug:
                print('Same speaker diff content:', video_path, random_video_path)
            speaker_embedding_audio_file = extract_audio(video_path=random_video_path, sr=SPEAKER_EMBEDDING_SAMPLE_RATE)
        except IndexError as e:
            # no content to choose from i.e. there is only 1 video from this person
            print('Failed to select different speaker embedding content:', e, video_path)

            # use speaker embedding from same video
            speaker_embedding_audio_file = extract_audio(video_path=video_path, sr=SPEAKER_EMBEDDING_SAMPLE_RATE)
    elif args.use_extracted_audio_path:
        audio_path = Path(video_path).with_suffix('.wav')
        speaker_embedding_audio_file = audio_path.open('r')
    elif args.default_audio_path: 
        speaker_embedding_audio_file = Path(args.default_audio_path).open('r')
    else:
        speaker_embedding_audio_file = extract_audio(video_path=video_path, sr=SPEAKER_EMBEDDING_SAMPLE_RATE)

    return speaker_embedding_audio_file


def process_wrapper(process_index, args, indexes):
    debug = args.debug and process_index == 0

    redis_server = redis.Redis(host=args.redis_host, port=6379)

    # get already preprocessed videos
    processed_videos_path = args.output_directory.joinpath('processed.txt')
    processed_videos = []
    if processed_videos_path.exists():
        with processed_videos_path.open('r') as f:
            processed_videos = f.read().splitlines()

    for i in tqdm(indexes):
        # grab video by index from redis list
        video_path = redis_server.lindex(args.pull_list_name, i).decode('utf-8')
        if video_path in processed_videos:
            continue

        original_video_path = video_path
        video_paths = [original_video_path]
        if args.speed_augmentation:
            for _ in range(2):
                new_video_path = f'/tmp/{str(uuid.uuid4())}.mp4'
                run_video_augmentation(video_path=original_video_path, new_video_path=new_video_path, random_prob=1)
                video_paths.append(new_video_path)
                
        # change video path
        path_d = {
            '/shared/HDD': '/media/alex/Storage/Domhnall',
            '/shared/SSD': '/media/SSD'
        }
        save_video_path = original_video_path
        for k, v in path_d.items():
            save_video_path = save_video_path.replace(k, v)

        for video_path in video_paths:
            # a different speaker embedding for every video
            speaker_embedding_audio_file = get_speaker_embedding_audio_file(args, original_video_path, debug)
            try:
                sample = process_video_2(
                    process_index=process_index,
                    video_path=video_path,
                    speaker_embedding_audio_file=speaker_embedding_audio_file,
                    landmarks_directory=args.landmarks_directory,
                    denoise=args.denoise,
                    skip_mel_spec=args.skip_mel_spec,
                    debug=debug
                )
                if sample is None:
                    continue
            except Exception:
                print(video_path, traceback.format_exc())
                break

            np.savez_compressed(str(args.output_directory.joinpath(f'{str(uuid.uuid4())}.npz')), sample=[save_video_path, *sample])

        # clean-up to save space
        for video_path in video_paths:
            if '/tmp/' in video_path:
                os.remove(video_path)

        # record video as processed, whether it was successful or not
        with processed_videos_path.open('a') as f:
            f.write(f'{original_video_path}\n')


def process(args):
    redis_server = redis.Redis(host=args.redis_host, port=6379)
    if not redis_server.exists(args.pull_list_name): 
        raise Exception(f'Redis Key "{args.pull_list_name}" does not exist')

    # create output directory
    output_directory = Path(args.output_directory)
    if output_directory.exists() and args.redo: 
        shutil.rmtree(str(output_directory))
    output_directory.mkdir(exist_ok=True)
    args.output_directory = output_directory

    # save CLI args for reference later
    with output_directory.joinpath('log.txt').open('w') as f:
        for arg in sys.argv:
            f.write(f'{arg} \\\n')

    args.speaker_content_mapping = None
    if args.same_speaker_different_content:
        # generate the mapping for selecting embeddings of same speaker, different content
        args.speaker_content_mapping = generate_speaker_video_mapping(redis_server, args.pull_list_name,
                                                                      args.speaker_id_index)

    # get already preprocessed videos
    processed_videos_path = args.output_directory.joinpath('processed.txt')
    processed_videos = []
    if processed_videos_path.exists():
        with processed_videos_path.open('r') as f:
            processed_videos = f.read().splitlines()

    # get all videos indexes left to preprocess
    indexes_left = []
    for index in list(range(redis_server.llen(args.pull_list_name))):
        video_path = redis_server.lindex(args.pull_list_name, index).decode('utf-8')
        if video_path in processed_videos:
            continue
        indexes_left.append(index)

    print('Num processed videos:', len(processed_videos))
    print('Num videos left:', len(indexes_left))

    # divide indexes left over between processes
    num_tasks_per_process = len(indexes_left) // args.num_processes
    tasks = []
    for i in range(args.num_processes): 
        start = i * num_tasks_per_process
        if i == args.num_processes - 1:
            end = len(indexes_left)
        else:
            end = start + num_tasks_per_process
        tasks.append([i, args, indexes_left[start:end]])

    # NOTE: to prevent deadlock on multiprocessing w/ fan predictor, export OMP_NUM_THREADS=1
    with multiprocessing.Pool(processes=args.num_processes) as pool: 
        pool.starmap(process_wrapper, tasks)


def server(args): 
    from flask import Flask, request, send_file

    assert Path(args.default_speaker_embedding_video_path).exists()

    app = Flask(__name__)

    output_directory = Path(args.output_directory)
    if output_directory.exists():
        shutil.rmtree(str(output_directory))
    output_directory.mkdir()

    @app.route('/process', methods=['POST'])
    def server_process():
        video_path = str(output_directory.joinpath('video.mp4'))
        output_path = str(output_directory.joinpath('output.npz'))

        video_file = request.files['video']
        video_file.save(video_path)

        if bool(int(request.args.get('use_audio_from_video', 0))):
            speaker_embedding_video_path = video_path
        else:
            speaker_embedding_video_path = args.default_speaker_embedding_video_path
    
        sample = process_video_2(
            process_index=0, 
            video_path=video_path, 
            speaker_embedding_audio_file=extract_audio(video_path=speaker_embedding_video_path, sr=SPEAKER_EMBEDDING_SAMPLE_RATE),
            landmarks_directory=None,
            denoise=args.denoise,
            skip_mel_spec=True
        )
        if sample is None:
            raise Exception('Failed to preprocess video')

        np.savez_compressed(output_path, sample=sample)

        return send_file(output_path, attachment_filename='processed.npz')

    app.run('0.0.0.0')


def main(args):
    # set params from the paper
    hparams.set_hparam('fps', 20)
    if args.sv2s_params:
        hparams.set_hparam('sample_rate', 24000)
        hparams.set_hparam('n_fft', 2048)
        hparams.set_hparam('win_size', 1200)
        hparams.set_hparam('hop_size', 300)

    f = {
        'process': process,
        'server': server
    }[args.run_type](args)


if __name__ == '__main__': 
    parser = argparse.ArgumentParser()
    parser.add_argument('--sv2s_params', action='store_true')
    parser.add_argument('--denoise', action='store_true')
    sub_parsers = parser.add_subparsers(dest='run_type')

    parser_1 = sub_parsers.add_parser('process')
    parser_1.add_argument('pull_list_name')
    parser_1.add_argument('output_directory')
    parser_1.add_argument('--redis_host', default='redis')
    parser_1.add_argument('--num_processes', type=int, default=5)
    parser_1.add_argument('--default_audio_path')
    parser_1.add_argument('--same_speaker_different_content', action='store_true')
    parser_1.add_argument('--use_extracted_audio_path', action='store_true')
    parser_1.add_argument('--speaker_id_index', type=int, default=-2)
    parser_1.add_argument('--speed_augmentation', action='store_true')
    parser_1.add_argument('--landmarks_directory')
    parser_1.add_argument('--skip_mel_spec', action='store_true')
    parser_1.add_argument('--redo', action='store_true')
    parser_1.add_argument('--debug', action='store_true')

    parser_2 = sub_parsers.add_parser('server')
    parser_2.add_argument('output_directory')
    parser_2.add_argument('default_speaker_embedding_video_path')

    main(parser.parse_args())
