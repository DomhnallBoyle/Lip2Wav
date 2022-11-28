"""Process VoxCeleb to filter out poor quality, non-english videos"""
import argparse
import math
import multiprocessing
import os
import random
import re
import sqlite3
import subprocess
import uuid
from datetime import datetime
from pathlib import Path

import numpy as np
import requests
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
        

def main(args):
    {
        'ffprobe': ffprobe,
        'ffprobe_select': ffprobe_select,
        'pose': pose,
        'pose_2': pose_2,
        'pose_sqlite_db_create': pose_sqlite_db_create, 
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

    main(parser.parse_args())
