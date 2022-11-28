import argparse
import multiprocessing
import random
import shutil
import uuid
from pathlib import Path

import numpy as np
import redis
from tqdm import tqdm

from audio_utils import extract_mel_spectrogram, get_audio_embeddings, preprocess_audio
from detectors import get_face_landmarks
from preprocessor import generate_speaker_video_mapping, get_speaker_and_content
from video_utils import convert_fps, extract_audio, get_fps, get_video_frames, get_video_rotation, run_video_augmentation

# TODO:
#  convert speaker embeddings and mel-spec to float32 - save space
#  don't save c = rectangle and l = np.array
#  transpose mel-spec
#  video augmentation


def process_video(process_index, video_path, output_directory, speaker_embedding_audio_file, fps):
    # convert fps if applicable
    if get_fps(video_path=video_path) != fps: 
        new_video_path = f'/tmp/video_fps_conversion_output_{process_index}.mp4'
        video_path = convert_fps(video_path=video_path, new_video_path=new_video_path, fps=fps)
        video_rotation = 0
    else:
        video_rotation = get_video_rotation(video_path=video_path)
    
    # get video frames
    video_frames = get_video_frames(video_path=video_path, rotation=video_rotation)
    if not video_frames: 
        return

    # get coords and landmarks
    # TODO: pass coords to get_face_landmarks to make quicker
    #  not for voxceleb though because lots of head movement
    detections = {}
    for i, frame in enumerate(video_frames): 
        face_stats = get_face_landmarks(frame=frame)
        if not face_stats:
            return
        face_coords, landmarks = face_stats
        detections[i] = {'c': face_coords, 'l': landmarks}

    # extract and preprocess audio from video
    audio_file = extract_audio(video_path=video_path)
    preprocessed_audio_file = preprocess_audio(audio_file=audio_file)
    audio_file.close()
    audio_file = preprocessed_audio_file

    # preprocess audio for speaker embedding
    preprocessed_speaker_embedding_audio_file = preprocess_audio(audio_file=speaker_embedding_audio_file)
    speaker_embedding_audio_file.close()
    speaker_embedding_audio_file = preprocessed_speaker_embedding_audio_file

    # extract speaker embedding
    speaker_embeddings = get_audio_embeddings(audio_file=speaker_embedding_audio_file)
    if not speaker_embeddings: 
        return
    speaker_embeddings = np.asarray(speaker_embeddings)

    # extract mel-spec
    mel_spec = extract_mel_spectrogram(audio_file=audio_file)

    # clean-up
    speaker_embedding_audio_file.close()
    audio_file.close()

    # save to disk
    obj = [video_path, detections, speaker_embeddings, mel_spec]
    save_name = f'{str(uuid.uuid4())}.npz'
    np.savez_compressed(str(output_directory.joinpath(save_name)), sample=obj)


def process_wrapper(process_index, args, start_index, end_index):
    debug = args.debug and process_index == 0

    redis_server = redis.Redis(host=args.redis_host, port=6379)

    # get already preprocessed videos
    processed_videos_path = args.output_directory.joinpath('processed.txt')
    processed_videos = []
    if processed_videos_path.exists():
        with processed_videos_path.open('r') as f:
            processed_videos = f.read().splitlines()

    for i in tqdm(range(start_index, end_index)):
        # grab video by index from redis list
        video_path = redis_server.lindex(args.pull_list_name, i).decode('utf-8')
        if video_path in processed_videos: 
            continue

        if args.speaker_content_mapping:
            # get speaker embedding of same speaker, different content
            speaker_id, content = get_speaker_and_content(Path(video_path), args.speaker_id_index)
            random_content = random.choice(list(set(args.speaker_content_mapping[speaker_id].keys()) - {content}))
            assert (content != random_content) and (content.split('_')[0] != random_content.split('_')[0])

            random_video_path = random.choice(args.speaker_content_mapping[speaker_id][random_content])
            assert video_path != random_video_path
            speaker_embedding_audio_file = extract_audio(video_path=random_video_path)
        else:
            speaker_embedding_audio_file = extract_audio(video_path=video_path)

        process_video(
            process_index=process_index,
            video_path=video_path, 
            output_directory=args.output_directory,
            speaker_embedding_audio_file=speaker_embedding_audio_file,
            fps=args.fps
        )

        # record video as processed, whether it was successful or not
        with processed_videos_path.open('a') as f:
            f.write(f'{video_path}\n')


def main(args):
    redis_server = redis.Redis(host=args.redis_host, port=6379)
    if not redis_server.exists(args.pull_list_name): 
        raise Exception(f'Redis Key "{args.pull_list_name}" does not exist')

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

    # split video indexes between the number of processes
    all_indexes = list(range(redis_server.llen(args.pull_list_name)))
    tasks = []
    num_tasks_per_process = len(all_indexes) // args.num_processes
    for i in range(args.num_processes): 
        start = i * num_tasks_per_process
        end = start + num_tasks_per_process
        if i == args.num_processes - 1:
            end = len(all_indexes)
        tasks.append([i, args, start, end])

    with multiprocessing.Pool(processes=args.num_processes) as pool: 
        pool.starmap(process_wrapper, tasks)


if __name__ == '__main__': 
    parser = argparse.ArgumentParser()
    parser.add_argument('pull_list_name')
    parser.add_argument('output_directory')
    parser.add_argument('--redis_host', default='redis')
    parser.add_argument('--fps', type=int, default=25)
    parser.add_argument('--num_processes', type=int, default=5)
    parser.add_argument('--same_speaker_different_content', action='store_true')
    parser.add_argument('--speaker_id_index', type=int, default=-2)
    parser.add_argument('--redo', action='store_true')
    parser.add_argument('--debug', action='store_true')

    main(parser.parse_args())
