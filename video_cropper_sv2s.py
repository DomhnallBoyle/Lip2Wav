"""Run forced alignment on video transcripts to crop out at the word level
Supply specific keywords to crop out from LRS3 or voxceleb2
"""
import argparse
import os
import random
import re
import shutil
import subprocess
import uuid

import numpy as np
import requests
import pandas as pd
from pathlib import Path
from tqdm import tqdm


def extract_audio(video_path, audio_path, ffmpeg_path='ffmpeg'):
    command = f'{ffmpeg_path} -hide_banner -loglevel error -threads 1 -y -i {video_path} -async 1 -ac 1 -vn -acodec pcm_s16le -ar 16000 {audio_path}'
    subprocess.call(command, shell=True)


def get_pose(video_path, pose_host):
    with open(video_path, 'rb') as f:
        response = requests.post(f'http://{pose_host}/estimate/', files={'video': f.read()})
        results = response.json()

    return results['angles']


def clean_up(video_path):
    os.remove(str(video_path))
    parent_directory = video_path.parents[0]
    if len(list(parent_directory.glob('*'))) == 0:  # delete empty directory
        shutil.rmtree(str(parent_directory))


def crop(word, video_path, transcript, crop_duration, fa_host, fa_port, output_directory, pose_host, pose_angle, 
         audio_path='/tmp/audio.wav', se_audio_path='/tmp/se_audio.wav', cropped_video_path='/tmp/cropped_video.mp4', 
         speaker_embedding_audio=False, debug=False):
    from audio_utils import forced_alignment
    from video_utils import crop as crop_video, get_video_duration

    ffmpeg_path = '/opt/lip2wav/ffmpeg-4.4.1-i686-static/ffmpeg'

    video_duration = get_video_duration(video_path)
    if video_duration < crop_duration: 
        return 0

    # run forced alignment
    extract_audio(video_path, audio_path, ffmpeg_path=ffmpeg_path)
    alignment = forced_alignment(audio_path, transcript, fa_host, port=fa_port)
    if alignment is None:
        return 0

    if debug:
        print(video_path, video_duration, audio_path, se_audio_path, cropped_video_path, alignment)

    # could be multiple occurrences of the same word
    word_occurrences = [x for x in alignment if x[0].lower() == word]

    # crop the video around the word start/end times
    count = 0
    for _, start_time, end_time, _ in word_occurrences:
        word_duration = end_time - start_time
        if word_duration > crop_duration:
            continue
        padding_duration = crop_duration - word_duration

        # try to place word in the middle
        pad = padding_duration / 2
        left_pad, right_pad = pad, pad
        crop_start_time = start_time - left_pad
        crop_end_time = end_time + right_pad

        # if crop times aren't valid, fix padding
        failed = crop_start_time < 0 or crop_end_time > video_duration
        if failed:
            if crop_start_time < 0: 
                shift = abs(crop_start_time)
                crop_start_time = 0
                crop_end_time += shift
            elif crop_end_time > video_duration:
                shift = crop_end_time - video_duration
                crop_start_time -= shift
                crop_end_time = video_duration

            # if crop_start_time < 0:
            #     left_pad = 0
            #     right_pad = padding_duration
            # elif crop_end_time > video_duration:
            #     left_pad = padding_duration
            #     right_pad = 0
            # crop_start_time = start_time - left_pad
            # crop_end_time = end_time + right_pad

        assert crop_start_time >= 0 and crop_end_time <= video_duration, f'{crop_start_time}, {crop_end_time}, {video_duration}, {word_occurrences}'
        calculated_crop_duration = round(crop_end_time - crop_start_time, 1)
        assert calculated_crop_duration == crop_duration, f'{calculated_crop_duration}, {crop_duration}'

        if debug:
            print(video_path, crop_start_time, crop_end_time)
        crop_video(video_path, crop_start_time, crop_end_time, cropped_video_path)
        if get_video_duration(cropped_video_path) != crop_duration:
            continue

        # save cropped video to directory
        word_stripped = word.replace('\'', '')
        video_name = f'{word_stripped}_{uuid.uuid4()}.mp4'  # remove apostrophes from file paths
        user_id = Path(video_path).parents[0].name
        output_user_directory = output_directory.joinpath(user_id)
        output_user_directory.mkdir(exist_ok=True, parents=True)
        output_video_path = output_user_directory.joinpath(video_name)
        shutil.copyfile(cropped_video_path, str(output_video_path))

        # save pose
        if pose_host:
            try:
                angles = get_pose(str(output_video_path), pose_host=pose_host)
            except Exception as e:
                print(f'Failed:', e)
                clean_up(output_video_path)
                continue
            
            num_frames = len(angles)
            acceptable_majority = int(num_frames * 0.9)
            yaws, pitches, _ = zip(*angles)
            num_valid_frames = sum([-pose_angle <= yaw <= pose_angle and 
                                    -pose_angle <= pitch <= pose_angle 
                                    for yaw, pitch in zip(yaws, pitches)])
            if num_valid_frames < acceptable_majority:
                clean_up(output_video_path)
                continue

        # save separate audio for speaker embedding?
        if speaker_embedding_audio:
            video_path_parent_directory = Path(video_path).parents[0]
            other_video_paths = [str(p) for p in video_path_parent_directory.glob('*.mp4') if str(p) != video_path]
            if not other_video_paths:
                # no other video paths to extract audio for speaker embedding
                clean_up(output_video_path)
                continue
            se_video_path = random.choice(other_video_paths)
            assert video_path != se_video_path
            extract_audio(se_video_path, se_audio_path, ffmpeg_path=ffmpeg_path)
            shutil.copyfile(se_audio_path, output_video_path.with_suffix('.wav'))

        count += 1

    return count


def whisper_qa(directory):
    # whisper ASR to check quality of FA cropping
    # NOTE: run this separately in ipython
    import whisper

    assert directory.exists()

    model = whisper.load_model('medium')
    decode_options = whisper.DecodingOptions(fp16=False, language='en')

    video_paths = list(directory.rglob('*.mp4'))
    audio_path = '/tmp/audio.wav'
    total_count = len(video_paths)
    invalid_video_paths = []

    # check if WOI in whisper ASR results
    for video_path in tqdm(video_paths):
        word = video_path.name.split('_')[0]
        extract_audio(str(video_path), audio_path)
        audio = whisper.load_audio(audio_path)
        audio = whisper.pad_or_trim(audio)
        mel = whisper.log_mel_spectrogram(audio).to(model.device)
        transcript = whisper.decode(model, mel, decode_options).text
        transcript = re.sub(r"[^\w\d\s-]+", '', transcript.lower())  # remove punctuation including apostrophes
        if word not in transcript:
            invalid_video_paths.append((str(video_path), transcript))

    print(f'{len(invalid_video_paths)}/{total_count} videos from {directory} are invalid')

    return invalid_video_paths


def _list(x):
    if os.path.exists(x):
        with open(x, 'r') as f:
            return f.read().splitlines()
    else:
        print('Path doesn\'t exist, using CLI args')

    return x.split(',')


def get_occurrences_lrs3(dataset_directory, words):
    word_path_d = {word.lower(): set() for word in words}
    for video_path in tqdm(list(dataset_directory.rglob('*.mp4'))):
        _set = Path(video_path).parents[1].name
        if _set not in ['pretrain', 'trainval', 'test']: 
            continue

        transcript_path = video_path.with_suffix('.txt')
        if not transcript_path.exists():
            continue

        with transcript_path.open('r') as f:
            transcript = f.read().splitlines()[0].split('Text:')[1].lower().strip()

        for word in transcript.split(' '):
            # transcript could have the same keyword multiple times
            if word in word_path_d:
                word_path_d[word].add((str(video_path), transcript))  # set = no duplicates

    return word_path_d


def get_occurrences_voxceleb(dataset_directory, words, asr_csv_path):
    word_path_d = {word.lower(): set() for word in words}
    df = pd.read_csv(asr_csv_path)

    for index, row in tqdm(df.iterrows()):
        video_id, transcript = row['ID'], row['Transcript']
        video_path = dataset_directory.joinpath(video_id)

        if pd.isna(transcript): 
            continue

        try:
            for word in transcript.split(' '):
                # transcript could have the same keyword multiple times
                if word in word_path_d:
                    word_path_d[word].add((str(video_path), transcript))  # set = no duplicates
        except Exception as e:
            print(video_id, transcript)
            raise e

    return word_path_d


def get_occurrences_sravi(dataset_directory, words, phrases, user_id=None):
    word_path_d = {word.lower(): set() for word in words}
    phrases_d = {p.lower().replace(' ', '').replace('?', '').replace('\'', ''): p.lower() for p in phrases}

    for video_path in dataset_directory.rglob('*.mp4'):
        if user_id and video_path.parents[0].name != user_id: 
            continue

        phrase = video_path.stem.split('_')[0].lower()
        phrase = phrases_d.get(phrase)
        if not phrase:
            continue
        
        for word in phrase.split(' '):
            if word in word_path_d:
                word_path_d[word].add((str(video_path), phrase))  # set = no duplicates

    return word_path_d


def main(args):
    dataset_directory = Path(args.dataset_directory)
    output_directory = Path(args.output_directory)
    if args.redo and output_directory.exists():
        shutil.rmtree(output_directory)
    output_directory.mkdir(exist_ok=True)

    print('Getting occurrences...')
    if args.run_type == 'lrs3': 
        word_occurrences = get_occurrences_lrs3(dataset_directory, args.words)
    elif args.run_type == 'voxceleb':
        word_occurrences = get_occurrences_voxceleb(dataset_directory, args.words, args.asr_csv_path)
    else:
        word_occurrences = get_occurrences_sravi(dataset_directory, args.words, args.phrases, args.user_id)
    word_occurrences = {k: list(v) for k, v in word_occurrences.items()}
    assert len(word_occurrences) == len(args.words)

    # debug word occurrences
    for word, occurrences in word_occurrences.items():
        print(word, len(occurrences))
    response = input('Are you happy to continue? y/n: ')
    if response.lower() != 'y': 
        return

    audio_path, se_audio_path = f'/tmp/{uuid.uuid4()}.wav', f'/tmp/{uuid.uuid4()}.wav'
    cropped_video_path = f'/tmp/{uuid.uuid4()}.mp4'

    for word, occurrences in word_occurrences.items():
        print(f'\nCropping for "{word}"...')
        random.shuffle(occurrences)
        word_crop_count = 0

        for video_path, transcript in occurrences:
            if word_crop_count >= args.min_num_occurrences:
                break
            print(f'{word_crop_count}/{args.min_num_occurrences}\r', end='')
            try:
                crop_count = crop(
                    word=word,
                    video_path=video_path,
                    transcript=transcript,
                    crop_duration=args.crop_duration,
                    fa_host=args.fa_host,
                    fa_port=args.fa_port,
                    output_directory=output_directory,
                    pose_host=args.pose_host,
                    pose_angle=args.pose_angle,
                    audio_path=audio_path,
                    se_audio_path=se_audio_path,
                    cropped_video_path=cropped_video_path,
                    speaker_embedding_audio=True,
                    debug=args.debug
                )
            except UnicodeEncodeError as e:
                print(e, video_path)
                continue
            except Exception as e:
                print(video_path, transcript)
                raise e
            word_crop_count += crop_count


if __name__ == '__main__': 
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset_directory')
    parser.add_argument('output_directory')
    parser.add_argument('words', type=_list)
    parser.add_argument('--crop_duration', type=int, default=1)
    parser.add_argument('--fa_host', default='0.0.0.0')
    parser.add_argument('--fa_port', type=int, default=8082)
    parser.add_argument('--min_num_occurrences', type=int)
    parser.add_argument('--pose_host', default='0.0.0.0:8085')
    parser.add_argument('--pose_angle', type=int, default=45)
    parser.add_argument('--redo', action='store_true')
    parser.add_argument('--debug', action='store_true')

    sub_parsers = parser.add_subparsers(dest='run_type')
    
    parser_1 = sub_parsers.add_parser('lrs3')
 
    parser_2 = sub_parsers.add_parser('voxceleb')
    parser_2.add_argument('asr_csv_path')

    parser_3 = sub_parsers.add_parser('sravi')
    parser_3.add_argument('phrases', type=_list)
    parser_3.add_argument('--user_id')

    main(parser.parse_args())
