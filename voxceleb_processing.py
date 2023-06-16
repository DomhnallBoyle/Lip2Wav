"""
Process VoxCeleb to filter out poor quality, non-english videos
Also filter by pose
"""
import argparse
import math
import multiprocessing
import os
import re
import sqlite3
import subprocess
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import redis
import requests
import torch
import whisper
from tqdm import tqdm


def ffprobe(args):
    command = '/opt/lip2wav/ffmpeg-4.4.1-i686-static/ffprobe -hide_banner {video_path} > {output_path} 2>&1'
    output_directory = Path(args.output_directory)
    output_directory.mkdir(exist_ok=True)

    for video_path in Path(args.voxceleb_directory).glob('*/*/*.mp4'):
        user_id, video_id, sample_id = video_path.with_suffix('').parts[-3:]
        output_path = output_directory.joinpath(f'{user_id}_{video_id}_{sample_id}_ffprobe.txt')
        if output_path.exists():
            continue

        subprocess.call(command.format(video_path=str(video_path), output_path=str(output_path)), shell=True)


def ffprobe_select(args):
    total = 0
    filtered_paths = []
    failed_paths = set()

    print('Filtering...')
    try:
        for probe_path in tqdm(Path(args.processed_directory).glob('*_ffprobe.txt')):
            total += 1

            with probe_path.open('r') as f:
                contents = f.read()

            if args.duration:
                duration = re.findall(r'Duration: (\d+:\d+:\d+.\d+).+', contents)[0]
                _datetime = datetime.strptime(duration, '%H:%M:%S.%f')
                if _datetime.second > args.duration:
                    continue

            if args.fps:
                try:
                    fps = int(re.findall(r'(\d+) fps', contents)[0])
                except IndexError:
                    failed_paths.add(probe_path.name)
                    continue
                if fps != args.fps:
                    continue

            if args.height and args.width:
                try:
                    width, height = re.findall(r'(\d+)x(\d+) ', contents)[0]
                except IndexError:
                    failed_paths.add(probe_path.name)
                    continue
                width, height = int(width), int(height)
                if height != args.height or width != args.width:
                    continue

            filtered_paths.append(probe_path.name)
    except KeyboardInterrupt:
        pass

    print('Total videos:', total)
    print('Filtered videos:', len(filtered_paths))
    print('Failed videos:', len(failed_paths))
    for path in filtered_paths:
        path = path.replace('_ffprobe.txt', '.mp4').replace('_', '/')
        print(path)


def pose_process(i, video_paths, pose_host, args):
    for video_path in tqdm(video_paths):
        video_path = Path(video_path)
        user_id, video_id, sample_id = video_path.with_suffix('').parts[-3:]
        output_path = args.output_directory.joinpath(f'{user_id}_{video_id}_{sample_id}_pose.npy')
        if output_path.exists():
            continue

        with video_path.open('rb') as f:
            try:
                response = requests.post(f'http://{pose_host}/estimate/', files={'video': f.read()})
            except Exception as e:
                print(f'Failed process {i}:', e)
                exit()
            if response.status_code != 200:
                print(video_path, response.__dict__)
                continue
            results = response.json()

        angles = np.asarray(results['angles'])
        np.save(str(output_path), angles)


def pose(args):
    output_directory = Path(args.output_directory)
    output_directory.mkdir(exist_ok=True)
    args.output_directory = output_directory

    num_processes_per_host = math.ceil(args.num_processes / len(args.pose_hosts))
    pose_hosts = []
    host_index = 0
    for i in range(args.num_processes):
        pose_hosts.append(args.pose_hosts[host_index])
        if (i + 1) % num_processes_per_host == 0:
            host_index += 1
    print('Pose hosts per process:', pose_hosts)

    tasks = []
    i = 0
    for video_path in tqdm(Path(args.voxceleb_directory).glob('*/*/*.mp4')):
        user_id, video_id, sample_id = video_path.with_suffix('').parts[-3:]
        output_path = args.output_directory.joinpath(f'{user_id}_{video_id}_{sample_id}_pose.npy')
        if output_path.exists():
            continue
        if args.multiprocessing:
            tasks.append([i, video_path, pose_hosts[i], args])
            if len(tasks) < args.num_processes:
                i += 1  # continue adding tasks
                continue

            assert len(tasks) == args.num_processes
            with multiprocessing.Pool(processes=args.num_processes) as p:
                p.starmap(pose_process, tasks)
            tasks = []
            i = 0
        else:
            pose_process(i, video_path, args.pose_hosts[0], args)

    # finish off any remaining
    if args.multiprocessing and len(tasks) > 0:
        with multiprocessing.Pool(processes=len(tasks)) as p:
            p.starmap(pose_process, tasks)


def pose_2(args):
    assert args.num_processes == len(args.pose_hosts)

    output_directory = Path(args.output_directory)
    output_directory.mkdir(exist_ok=True)
    args.output_directory = output_directory

    video_paths = [os.path.join(root, f) for root, dirs, files in os.walk(args.voxceleb_directory)
                   for f in files if f[-4:] == '.mp4']
    print('Total video paths:', len(video_paths))

    video_paths_to_process = []
    for video_path in video_paths:
        user_id, video_id, sample_id = Path(video_path).with_suffix('').parts[-3:]
        output_path = args.output_directory.joinpath(f'{user_id}_{video_id}_{sample_id}_pose.npy')
        if output_path.exists():
            continue
        video_paths_to_process.append(video_path)
    print('Video paths left:', len(video_paths_to_process))

    num_paths_per_processes = len(video_paths_to_process) // args.num_processes
    tasks = []
    for i in range(args.num_processes):
        start = i * num_paths_per_processes
        if i == args.num_processes - 1:
            end = len(video_paths_to_process)
        else:
            end = start + num_paths_per_processes
        video_paths = video_paths_to_process[start:end]
        tasks.append([i, video_paths, args.pose_hosts[i], args])

    with multiprocessing.Pool(processes=args.num_processes) as p:
        p.starmap(pose_process, tasks)


def pose_sqlite_process(pose_paths, pose_ids, db_path, voxceleb_path): 
    con = sqlite3.connect(str(db_path))
    cursor = con.cursor()
    cursor.execute('PRAGMA main.auto_vacuum = 1;')  # full vacuum - free up space

    batch_size = 1_000_000
    video_batch, frame_batch = [], []
    name_regex = r'(id\d+)_(.+)_(\d+)_pose'

    video_batch, frame_batch = [], []
    try:
        for pose_path, pose_id in tqdm(zip(pose_paths, pose_ids), total=len(pose_paths)):
            user_id, video_id, sample_id = re.match(name_regex, pose_path.stem).groups()

            video_name = Path(user_id).joinpath(video_id, f'{sample_id}.mp4')
            # assert voxceleb_path.joinpath(video_name).exists()  # to ensure the regex works
            # assert os.path.isfile(str(voxceleb_path.joinpath(video_name)))
            
            count = cursor.execute('SELECT COUNT(*) FROM video WHERE path = ?', (str(video_name),)).fetchone()[0]
            if int(count) == 1: 
                continue  # already exists

            # limited by the numpy loads
            angles = np.load(str(pose_path)).tolist()
            yaws, pitches, rolls = zip(*angles)

            video_batch.append((pose_id, str(video_name), len(angles)))
            frame_batch.extend([(pose_id, yaw, pitch, roll) for yaw, pitch, roll in zip(yaws, pitches, rolls)])

            # batch inserts - faster
            if len(video_batch) >= batch_size or len(frame_batch) >= batch_size: 
                cursor.executemany('INSERT INTO video VALUES (?, ?, ?)', video_batch)
                cursor.executemany('INSERT INTO frame (video_id, yaw, pitch, roll) VALUES (?, ?, ?, ?)', frame_batch)
                con.commit()
                video_batch, frame_batch = [], []

        if len(video_batch) > 0 and len(frame_batch) > 0: 
            cursor.executemany('INSERT INTO video VALUES (?, ?, ?)', video_batch)
            cursor.executemany('INSERT INTO frame (video_id, yaw, pitch, roll) VALUES (?, ?, ?, ?)', frame_batch)
            con.commit()

        con.close()
    except KeyboardInterrupt:
        con.commit()
        con.close()  


def pose_sqlite_db_create(args): 
    voxceleb_path = Path(args.voxceleb_directory)

    db_path = Path(args.db_path)
    if db_path.exists():
        os.remove(str(db_path))
    
    con = sqlite3.connect(str(db_path))
    cursor = con.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS video (
            id INTEGER PRIMARY KEY,
            path TEXT,
            num_frames INTEGER
        )
    """)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS frame (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            video_id INTEGER,
            pitch REAL,
            yaw REAL,
            roll REAL,
            FOREIGN KEY (video_id) REFERENCES video (id)
        )
    """)
    cursor.execute('CREATE INDEX video_id_index ON video(id);')  # faster selects
    cursor.execute('CREATE INDEX video_path_index ON video(path);')
    cursor.execute('CREATE INDEX frame_id_index ON frame(id);')
    cursor.execute('CREATE INDEX frame_video_id_index ON frame(video_id);')
    con.commit()
    con.close()

    processed_pose_paths = list(Path(args.processed_directory).glob('*_pose.npy'))
    pose_ids = list(range(1, len(processed_pose_paths) + 1))
    assert len(processed_pose_paths) == len(pose_ids)
    num_paths_per_process = len(processed_pose_paths) // args.num_processes
    tasks = []
    for i in range(args.num_processes): 
        start = i * num_paths_per_process
        if i == args.num_processes - 1: 
            end = len(processed_pose_paths)
        else:
            end = start + num_paths_per_process
        tasks.append([processed_pose_paths[start:end], pose_ids[start:end], db_path, voxceleb_path])
            
    with multiprocessing.Pool(processes=args.num_processes) as p:
        p.starmap(pose_sqlite_process, tasks)  
        

def to_redis(arr, video_path):
    arr_dtype = bytearray(str(arr.dtype), 'utf-8')
    arr_shape = bytearray(','.join([str(a) for a in arr.shape]), 'utf-8')
    sep = bytearray('|', 'utf-8')
    arr_bytes = arr.ravel().tobytes()
    video_path_bytes = bytearray(str(video_path), 'utf-8')

    to_return = video_path_bytes + sep + arr_dtype + sep + arr_shape + sep + arr_bytes

    return to_return


def from_redis(serialized_arr):
    sep = '|'.encode('utf-8')
    i_0 = serialized_arr.find(sep)
    i_1 = serialized_arr.find(sep, i_0 + 1)
    i_2 = serialized_arr.find(sep, i_1 + 1)

    video_path = serialized_arr[:i_0].decode('utf-8')
    arr_dtype = serialized_arr[i_0 + 1:i_1].decode('utf-8')
    arr_shape = tuple([int(a) for a in serialized_arr[i_1 + 1:i_2].decode('utf-8').split(',')])
    arr_str = serialized_arr[i_2 + 1:]
    arr = np.frombuffer(arr_str, dtype = arr_dtype).reshape(arr_shape)

    return arr, video_path


def language_detection_whisper(args): 
    r = redis.Redis(host='0.0.0.0')
    r.delete(args.queue_name)

    model = whisper.load_model('medium')  # ~5GB VRAM

    while True: 
        print('Queue size:', r.llen(args.queue_name))
        encoded = r.rpop(args.queue_name)  # removes from tail
        if encoded is None: 
            time.sleep(1)
            continue
        mel, video_path = from_redis(encoded)

        _, probs = model.detect_language(torch.from_numpy(mel).to(model.device))
        lang = max(probs, key=probs.get)
        conf = probs[lang]  # softmax score

        with Path(args.output_path).open('a') as f:
            f.write(f'{video_path} {lang} {conf}\n')


def extract_audio(video_path, audio_path):
    command = f'ffmpeg -hide_banner -loglevel error -threads 1 -y -i {video_path} -async 1 -ac 1 -vn -acodec pcm_s16le -ar 16000 {audio_path}'
    subprocess.call(command, shell=True)


def language_detection_worker(i, args, video_paths):
    r = redis.Redis(host='0.0.0.0')

    audio_path = f'/tmp/audio_{i}.wav'
    for video_path in tqdm(video_paths):
        while r.llen(args.queue_name) > 100:  # don't overload redis
            time.sleep(1)
        extract_audio(video_path, audio_path)
        
        audio = whisper.load_audio(audio_path)
        audio = whisper.pad_or_trim(audio)  # trimmed to 30 seconds
        mel = whisper.log_mel_spectrogram(audio).numpy()
        encoded = to_redis(mel, video_path)

        r.lpush(args.queue_name, bytes(encoded))  # push to the head
        

def language_detection_workers(args):
    video_paths = [os.path.join(root, f) for root, dirs, files in os.walk(args.voxceleb_directory)
                   for f in files if f[-4:] == '.mp4']
    # video_paths = [str(p) for p in list(Path('/media/alex/Storage/Domhnall/datasets/vox_celeb/2/videos/dev/samples').glob('*.mp4'))]
    print('Total video paths:', len(video_paths))

    # check if video paths already done
    output_path = Path(args.output_path)
    if output_path.exists():
        with output_path.open('r') as f:
            processed_video_paths = [l.split(' ')[0].strip() for l in f.read().splitlines()]
        video_paths = list(set(video_paths) - set(processed_video_paths))
        if not video_paths:
            print('No videos left to process')
            exit()

    print('Video paths left to process:', len(video_paths))

    tasks = []
    num_paths_per_process = len(video_paths) // args.num_workers
    for i in range(args.num_workers): 
        start = i * num_paths_per_process
        if i == args.num_workers - 1:
            end = len(video_paths)
        else:
            end = start + num_paths_per_process
        tasks.append([i, args, video_paths[start:end]])

    with multiprocessing.Pool(processes=args.num_workers) as p:
        p.starmap(language_detection_worker, tasks)


def language_csv_create(args):
    results_path = Path(args.results_path)
    assert results_path.exists() and '.txt' in results_path.name
    with results_path.open('r') as f:
        results = f.read().splitlines()

    csv_path = Path(args.csv_path)
    assert str(results_path) != str(csv_path)
    if csv_path.exists():
        os.remove(csv_path)
    with csv_path.open('w') as f:
        f.write(f'ID,Language,Confidence\n')
        for line in results:
            video_path, lang, conf = line.split(' ')
            video_id = '/'.join(Path(video_path).parts[-3:])
            f.write(f'{video_id},{lang},{conf}\n')


def language_filtering_deepspeech(args):
    from nltk.corpus import words

    word_pool = set(words.words())

    us_2_gb_d = requests.get('https://raw.githubusercontent.com/hyperreality/American-British-English-Translator/master/data/american_spellings.json').json()
    gb_2_us_d = {v: k for k, v in us_2_gb_d.items()}

    # deepspeech trained on librispeech - mix of US and GB english
    test_transcripts = ['my name is domhnall']
    num_transcripts = len(test_transcripts)
    valid_transcripts = []
    for transcript in test_transcripts: 
        transcript = transcript.strip().lower()
        words = transcript.split(' ')
        num_valid_words = sum([word in word_pool for word in words]) 
        score = num_valid_words / len(words)
        if score >= args.valid_utt_threshold:
            valid_transcripts.append(transcript)
        print(f'~ no. English transcripts: {len(valid_transcripts)} / {len(test_transcripts)}')

    # TODO: 
    #  include running approx filtered dataset size - should be 1326 hours
    #  create csv with video id, transcript and score


def asr_whisper(args):
    voxceleb_directory = Path(args.voxceleb_directory)

    with open(args.video_paths_text_file, 'r') as f:
        video_paths = f.read().splitlines()

    model = whisper.load_model(args.whisper_model)
    decode_options = whisper.DecodingOptions(fp16=False, language='en')  # no need for language detection

    output_csv_path = Path(args.output_csv_path)
    audio_path = '/tmp/audio.wav'

    processed_video_ids = []
    if output_csv_path.exists():
        with output_csv_path.open('r') as f:
            processed_video_ids = [l.split(',')[0] for l in f.read().splitlines()]
    else:
        with output_csv_path.open('w') as f:
            f.write('ID,Transcript\n')

    print(f'Already processed {len(processed_video_ids)} videos...')
    processed_video_ids = set(processed_video_ids)

    for video_path in tqdm(video_paths):
        video_path = voxceleb_directory.joinpath(video_path)
        assert video_path.exists()
        video_id = '/'.join(video_path.parts[-3:])
        if video_id in processed_video_ids: 
            continue

        extract_audio(str(video_path), audio_path)
        audio = whisper.load_audio(audio_path)
        audio = whisper.pad_or_trim(audio)
        mel = whisper.log_mel_spectrogram(audio).to(model.device)
        transcript = whisper.decode(model, mel, decode_options).text
        transcript = re.sub(r"[^\w\d'\s-]+", '', transcript.lower())  # remove punctuation except apostrophes
    
        with output_csv_path.open('a') as f:
            f.write(f'{video_id},{transcript}\n')
 

def main(args):
    {
        'ffprobe': ffprobe,
        'ffprobe_select': ffprobe_select,
        'pose': pose,
        'pose_2': pose_2,
        'pose_sqlite_db_create': pose_sqlite_db_create, 
        'language_detection_whisper': language_detection_whisper,
        'language_detection_workers': language_detection_workers,
        'language_csv_create': language_csv_create,
        'language_filtering_deepspeech': language_filtering_deepspeech,
        'asr_whisper': asr_whisper
    }[args.run_type](args)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    sub_parsers = parser.add_subparsers(dest='run_type')

    parser_1 = sub_parsers.add_parser('ffprobe')
    parser_1.add_argument('voxceleb_directory')
    parser_1.add_argument('output_directory')

    parser_2 = sub_parsers.add_parser('ffprobe_select')
    parser_2.add_argument('processed_directory')
    parser_2.add_argument('--fps', type=int)
    parser_2.add_argument('--duration', type=float)
    parser_2.add_argument('--height', type=int)
    parser_2.add_argument('--width', type=int)

    parser_3 = sub_parsers.add_parser('pose')
    parser_3.add_argument('voxceleb_directory')
    parser_3.add_argument('output_directory')
    parser_3.add_argument('pose_hosts', type=lambda s: s.split(','))
    parser_3.add_argument('--multiprocessing', action='store_true')
    parser_3.add_argument('--num_processes', type=int, default=5)

    parser_4 = sub_parsers.add_parser('pose_2')
    parser_4.add_argument('voxceleb_directory')
    parser_4.add_argument('output_directory')
    parser_4.add_argument('pose_hosts', type=lambda s: s.split(','))
    parser_4.add_argument('--num_processes', type=int, default=5)

    parser_5 = sub_parsers.add_parser('pose_sqlite_db_create')
    parser_5.add_argument('voxceleb_directory')
    parser_5.add_argument('processed_directory')
    parser_5.add_argument('db_path')
    parser_5.add_argument('--num_processes', type=int, default=5)

    parser_6 = sub_parsers.add_parser('language_detection_whisper')
    parser_6.add_argument('queue_name')
    parser_6.add_argument('output_path')

    parser_7 = sub_parsers.add_parser('language_detection_workers')
    parser_7.add_argument('voxceleb_directory')
    parser_7.add_argument('queue_name')
    parser_7.add_argument('output_path')
    parser_7.add_argument('--num_workers', type=int, default=5)

    parser_8 = sub_parsers.add_parser('language_csv_create')
    parser_8.add_argument('results_path')
    parser_8.add_argument('csv_path')

    parser_9 = sub_parsers.add_parser('language_filtering_deepspeech')
    parser_9.add_argument('--valid_utt_threshold', type=float, default=0.6)

    parser_10 = sub_parsers.add_parser('asr_whisper')
    parser_10.add_argument('voxceleb_directory')
    parser_10.add_argument('video_paths_text_file')
    parser_10.add_argument('output_csv_path')
    parser_10.add_argument('--whisper_model', default='medium')

    main(parser.parse_args())
