import argparse
import os
import multiprocessing
import pickle
import random
import shutil
import tempfile
import uuid
from pathlib import Path

import numpy as np
import redis
from tqdm import tqdm

from audio_utils import extract_mel_spectrogram, get_audio_embeddings, play_audio, preprocess_audio
from detectors import get_face_landmarks, get_face_landmarks_fan, get_mouth_frames_perspective_warp_2, \
    get_mouth_frames_wrapper, smooth_landmarks
from preprocessor import generate_speaker_video_mapping, get_speaker_and_content
from synthesizer.hparams import hparams
from video_utils import convert_fps, extract_audio, get_fps, get_video_duration, get_video_frames, get_video_rotation, \
    run_video_augmentation, save_video_frames, show_frames

FPS = 20
# SAMPLE_RATE = 24000
SAMPLE_RATE = 16000


def interpolate_2d(m, s):
    from scipy import interpolate

    x = np.linspace(0, 1, m.shape[0])
    y = np.linspace(0, 1, m.shape[1])
    f = interpolate.interp2d(y, x, m, kind='cubic')

    x2 = np.linspace(0, 1, s[0])
    y2 = np.linspace(0, 1, s[1])

    return f(y2, x2)


def process_video(process_index, video_path, output_directory, speaker_embedding_audio_file, fps, greyscale, interpolate,
                  landmarks_directory, convert_fps_after_landmarks, use_fan_landmarks, device, debug):
    # TODO: 
    #  fix sample rate bug, output is still 16000Hz not matter what SAMPLE RATE is
    #  deal with speaker encoder memory leak failing, we don't want videos continuing processing

    original_video_path = video_path
    original_fps = get_fps(video_path=video_path)
    video_fps_conversion_path = f'/tmp/video_fps_conversion_output_{process_index}.mp4'

    # convert fps if applicable
    if original_fps != fps:
        video_path = convert_fps(video_path=video_path, new_video_path=video_fps_conversion_path, fps=fps)
        video_rotation = 0

    # extract and preprocess audio from video
    audio_file = extract_audio(video_path=video_path, sr=SAMPLE_RATE)
    if debug:
        play_audio(audio_file.name)
    preprocessed_audio_file = preprocess_audio(audio_file=audio_file)
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
    # expects 
    if use_fan_landmarks:
        all_landmarks = []
        iterator = video_frames
        if debug: 
            iterator = tqdm(iterator)
        for frame in iterator: 
            landmarks = get_face_landmarks_fan(frame=frame, device=device)[1]
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
    
    assert mouth_frames.shape == (len(video_frames), 96, 96)
    mouth_frames = np.asarray(mouth_frames).astype(np.uint8)

    if debug:
        print('Mouth frames', mouth_frames.shape)
        show_frames(mouth_frames, 50, 'Mouth Frames')

    # preprocess audio for speaker embedding
    preprocessed_speaker_embedding_audio_file = preprocess_audio(audio_file=speaker_embedding_audio_file)
    speaker_embedding_audio_file.close()
    speaker_embedding_audio_file = preprocessed_speaker_embedding_audio_file

    # extract speaker embedding
    speaker_embeddings = get_audio_embeddings(audio_file=speaker_embedding_audio_file)
    if speaker_embeddings is None:
        raise Exception(f'{process_index} failed to get speaker embeddings')
    speaker_embedding = np.asarray(speaker_embeddings[0]).astype(np.float32)

    # extract mel-spec
    mel_spec = extract_mel_spectrogram(audio_file=audio_file).T
    duration = len(mouth_frames) / FPS
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

    # save to disk
    obj = [original_video_path, mouth_frames, speaker_embedding, mel_spec]
    save_name = f'{str(uuid.uuid4())}.npz'
    np.savez_compressed(str(output_directory.joinpath(save_name)), sample=obj)


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
                speaker_embedding_audio_file = extract_audio(video_path=random_video_path, sr=SAMPLE_RATE)
            except IndexError as e:
                # no content to choose from i.e. there is only 1 video from this person
                print('Failed to select different speaker embedding content:', e, video_path)

                # use speaker embedding from same video
                speaker_embedding_audio_file = extract_audio(video_path=video_path, sr=SAMPLE_RATE)
        else:
            speaker_embedding_audio_file = extract_audio(video_path=video_path, sr=SAMPLE_RATE)

        original_video_path = video_path
        video_paths = [original_video_path]
        if args.speed_augmentation:
            for _ in range(2):
                # use original stem because need phrase for calculating weights correctly
                new_video_path = f'/tmp/{Path(original_video_path).stem}_{uuid.uuid4()}.mp4'
                run_video_augmentation(video_path=original_video_path, new_video_path=new_video_path, random_prob=1)
                video_paths.append(new_video_path)
        for video_path in video_paths:
            # another file object because it's closed in methods
            _speaker_embedding_audio_file = tempfile.NamedTemporaryFile(suffix='.wav')
            shutil.copyfile(speaker_embedding_audio_file.name, _speaker_embedding_audio_file.name)
            _speaker_embedding_audio_file.seek(0)
            
            try:
                process_video(
                    process_index=process_index,
                    video_path=video_path,
                    output_directory=args.output_directory,
                    speaker_embedding_audio_file=_speaker_embedding_audio_file,
                    fps=FPS,
                    greyscale=args.greyscale,
                    interpolate=args.interpolate,
                    landmarks_directory=args.landmarks_directory,
                    convert_fps_after_landmarks=args.convert_fps_after_landmarks,
                    use_fan_landmarks=args.use_fan_landmarks,
                    device='cpu' if args.cpu else 'cuda:0',
                    debug=debug
                )
            except Exception as e:
                print(e)
                exit()
            if '/tmp/' in video_path:
                os.remove(video_path)  # to save space
        speaker_embedding_audio_file.close()

        # record video as processed, whether it was successful or not
        with processed_videos_path.open('a') as f:
            f.write(f'{original_video_path}\n')


def main(args):
    redis_server = redis.Redis(host=args.redis_host, port=6379)
    if not redis_server.exists(args.pull_list_name): 
        raise Exception(f'Redis Key "{args.pull_list_name}" does not exist')

    # # from the paper
    # hparams.set_hparam('sample_rate', SAMPLE_RATE)
    # hparams.set_hparam('n_fft', 2048)
    # hparams.set_hparam('win_size', 1200)
    # hparams.set_hparam('hop_size', 300)

    # create output directory
    output_directory = Path(args.output_directory)
    if output_directory.exists() and args.redo: 
        shutil.rmtree(str(output_directory))
    output_directory.mkdir(exist_ok=True)
    args.output_directory = output_directory

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

    with multiprocessing.Pool(processes=args.num_processes) as pool: 
        pool.starmap(process_wrapper, tasks)


if __name__ == '__main__': 
    parser = argparse.ArgumentParser()
    parser.add_argument('pull_list_name')
    parser.add_argument('output_directory')
    parser.add_argument('--redis_host', default='redis')
    parser.add_argument('--greyscale', action='store_true')
    parser.add_argument('--num_processes', type=int, default=5)
    parser.add_argument('--same_speaker_different_content', action='store_true')
    parser.add_argument('--speaker_id_index', type=int, default=-2)
    parser.add_argument('--interpolate', action='store_true')
    parser.add_argument('--speed_augmentation', action='store_true')
    parser.add_argument('--landmarks_directory')
    parser.add_argument('--convert_fps_after_landmarks', action='store_true')
    parser.add_argument('--use_fan_landmarks', action='store_true')
    parser.add_argument('--cpu', action='store_true')
    parser.add_argument('--redo', action='store_true')
    parser.add_argument('--debug', action='store_true')

    main(parser.parse_args())
