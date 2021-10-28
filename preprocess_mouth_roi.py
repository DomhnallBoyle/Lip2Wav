"""Based on LipNet cropping of the mouth ROI"""
import sys

if sys.version_info[0] < 3 and sys.version_info[1] < 2:
    raise Exception("Must be using >= Python 3.2")

from os import listdir, path

if not path.isfile('face_detection/detection/sfd/s3fd.pth'):
    raise FileNotFoundError('Save the s3fd model to face_detection/sfd/s3fd.pth \
							before running this script!')

import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
import argparse, os, cv2, traceback, subprocess
from tqdm import tqdm
from glob import glob
from synthesizer import audio
from synthesizer.hparams import hparams as hp

import face_alignment
# import face_detection
import dlib
import shutil
import torch
import time

parser = argparse.ArgumentParser()

parser.add_argument('--ngpu', help='Number of GPUs across which to run in parallel', default=1, type=int)
parser.add_argument('--batch_size', help='Single GPU Face detection batch size', default=32, type=int)
parser.add_argument("--data_root", help="Root folder of the LRW dataset", required=True)
parser.add_argument("--preprocessed_root", help="Root folder of the preprocessed dataset", required=True)
parser.add_argument("--split", help="* | train | val | test (* will preprocess all splits)", default="test",
                    choices=["*", "train", "val", "test"])


args = parser.parse_args()

# fa = [face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, flip_input=False,
#                                    device='cpu') for _ in range(args.ngpu)]  # S3FD face detectors CPU
# fa = [face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, flip_input=False,
#                                    device='cuda:{}'.format(id)) for id in range(args.ngpu)]  # S3FD face detectors multiple GPU
fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, flip_input=False, device='cuda:{}'.format(0))  # S3FD face detectors 1 GPU

# fa = [face_detection.FaceAlignment(face_detection.LandmarksType._2D, flip_input=False,
#                                    device='cuda:{}'.format(id)) for id in range(args.ngpu)]  # S3FD face detectors GPU

# face_detectors = [dlib.get_frontal_face_detector() for _ in range(args.ngpu)]  # CPU HOG Face detectors
# shape_predictors = [dlib.shape_predictor('shape_predictor_68_face_landmarks.dat') for _ in range(args.ngpu)]  # CPU shape predictors based on landmarks
face_detector = dlib.get_frontal_face_detector()
shape_predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

# template = 'ffmpeg -loglevel panic -y -i {} -ar {} -f wav {}'
template2 = 'ffmpeg -hide_banner -loglevel panic -threads 1 -y -i {} -async 1 -ac 1 -vn -acodec pcm_s16le -ar 16000 {}'

"""3 ways of face and landmark detection:
1: DLIB detector (CPU) and DLIB landmarks (CPU) - very fast w/ multiple cores
2: S3FD face detector (GPU) and DLIB landmarks (CPU) - 1.3 videos per second, not as fast as 1
3: S3FD face detector (GPU) and Pytorch landmarks (GPU) - ~20FPS
"""


def get_frames_mouth(detector, predictor, frames):
    """
    1: DLIB face and landmark detector
    """
    MOUTH_WIDTH = 100
    MOUTH_HEIGHT = 50
    HORIZONTAL_PAD = 0.19
    normalize_ratio = None
    mouth_frames = []
    for frame in frames:
        dets = detector(frame, 1)
        # pred = detector.get_detections_for_batch(np.array([frame]))[0]
        # dets = [dlib.rectangle(*pred[1:])]
        shape = None
        for k, d in enumerate(dets):
            shape = predictor(frame, d)
            i = -1
        if shape is None: # Detector doesn't detect face, just return as is
            return frames
        mouth_points = []
        for part in shape.parts():
            i += 1
            if i < 48: # Only take mouth region
                continue
            mouth_points.append((part.x,part.y))
        np_mouth_points = np.array(mouth_points)

        mouth_centroid = np.mean(np_mouth_points[:, -2:], axis=0)

        if normalize_ratio is None:
            mouth_left = np.min(np_mouth_points[:, :-1]) * (1.0 - HORIZONTAL_PAD)
            mouth_right = np.max(np_mouth_points[:, :-1]) * (1.0 + HORIZONTAL_PAD)

            normalize_ratio = MOUTH_WIDTH / float(mouth_right - mouth_left)

        new_img_shape = (int(frame.shape[0] * normalize_ratio), int(frame.shape[1] * normalize_ratio))
        # resized_img = imresize(frame, new_img_shape)
        resized_img = cv2.resize(frame, new_img_shape)

        mouth_centroid_norm = mouth_centroid * normalize_ratio

        mouth_l = int(mouth_centroid_norm[0] - MOUTH_WIDTH / 2)
        mouth_r = int(mouth_centroid_norm[0] + MOUTH_WIDTH / 2)
        mouth_t = int(mouth_centroid_norm[1] - MOUTH_HEIGHT / 2)
        mouth_b = int(mouth_centroid_norm[1] + MOUTH_HEIGHT / 2)

        mouth_crop_image = resized_img[mouth_t:mouth_b, mouth_l:mouth_r]
        if mouth_crop_image.shape != (50, 100, 3):
            mouth_crop_image = cv2.resize(mouth_crop_image, (50, 100))

        mouth_frames.append(mouth_crop_image)
    return mouth_frames


def s3fd_detector_and_pytorch_landmarks(_frames):
        """
        # 3: S3FD detector and Pytorch landmarks 
        # this works but is quite slow because only 1 GPU
        # takes approx 1 second per video
        # takes longer on CPU OMG!!!
        """
        all_landmarks = []
        for frame in _frames:
            landmarks = fa.get_landmarks_from_image(np.asarray(frame))
            if landmarks is not None:
                all_landmarks.append(landmarks[0])

        if len(_frames) != len(all_landmarks):
            return []

        MOUTH_WIDTH = 100
        MOUTH_HEIGHT = 50
        HORIZONTAL_PAD = 0.19
        normalize_ratio = None
        mouth_frames = []
        for frame, landmarks in zip(_frames, all_landmarks):
            mouth_points = []
            for i in range(len(landmarks)):
                if i < 48:
                    continue
                mouth_points.append((landmarks[i][0], landmarks[i][1]))

            np_mouth_points = np.array(mouth_points)
            mouth_centroid = np.mean(np_mouth_points[:, -2:], axis=0)

            if normalize_ratio is None:
                mouth_left = np.min(np_mouth_points[:, :-1]) * (1.0 - HORIZONTAL_PAD)
                mouth_right = np.max(np_mouth_points[:, :-1]) * (1.0 + HORIZONTAL_PAD)

                normalize_ratio = MOUTH_WIDTH / float(mouth_right - mouth_left)

            new_img_shape = (int(frame.shape[0] * normalize_ratio), int(frame.shape[1] * normalize_ratio))
            # resized_img = imresize(frame, new_img_shape)
            resized_img = cv2.resize(frame, new_img_shape)

            mouth_centroid_norm = mouth_centroid * normalize_ratio

            mouth_l = int(mouth_centroid_norm[0] - MOUTH_WIDTH / 2)
            mouth_r = int(mouth_centroid_norm[0] + MOUTH_WIDTH / 2)
            mouth_t = int(mouth_centroid_norm[1] - MOUTH_HEIGHT / 2)
            mouth_b = int(mouth_centroid_norm[1] + MOUTH_HEIGHT / 2)

            mouth_crop_image = resized_img[mouth_t:mouth_b, mouth_l:mouth_r]
            if mouth_crop_image.shape != (50, 100, 3):
                mouth_crop_image = cv2.resize(mouth_crop_image, (50, 100))
            mouth_frames.append(mouth_crop_image)

        return mouth_frames


def mouth_frames_bad(_mouth_frames):
    return len(_mouth_frames) == 0 or not all([mouth_frame.shape == (50, 100, 3) for mouth_frame in _mouth_frames])


def process_video_file(vfile, args, gpu_id, detector='dlib'):
    video_stream = cv2.VideoCapture(vfile)

    frames = []
    while 1:
        still_reading, frame = video_stream.read()
        if not still_reading:
            video_stream.release()
            break
        frames.append(frame)

    vidname = os.path.basename(vfile).split('.')[0]
    split = vfile.split('/')[-2]
    word = vfile.split('/')[-3]

    fulldir = path.join(args.preprocessed_root, word, split, vidname)
    os.makedirs(fulldir, exist_ok=True)
    #print (fulldir)

    wavpath = path.join(fulldir, 'audio.wav')
    specpath = path.join(fulldir, 'mels.npz')

    command = template2.format(vfile, wavpath)
    result = subprocess.call(command, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)

    # this works and is super fast because of multiple CPU cores
    if detector == 'dlib':
        mouth_frames = get_frames_mouth(face_detector, shape_predictor, frames)
    else:
        mouth_frames = s3fd_detector_and_pytorch_landmarks(frames)
    
    if mouth_frames_bad(mouth_frames):
        shutil.rmtree(fulldir)
        return False

    for i, mouth_frame in enumerate(mouth_frames):
        cv2.imwrite(path.join(fulldir, '{}.jpg'.format(i)), mouth_frame)

    return True

def process_audio_file(vfile, args, gpu_id):
    vidname = os.path.basename(vfile).split('.')[0]
    split = vfile.split('/')[-2]
    word = vfile.split('/')[-3]

    fulldir = path.join(args.preprocessed_root, word, split, vidname)
    os.makedirs(fulldir, exist_ok=True)

    wavpath = path.join(fulldir, 'audio.wav')
    specpath = path.join(fulldir, 'mels.npz')

    wav = audio.load_wav(wavpath, hp.sample_rate)
    spec = audio.melspectrogram(wav, hp)
    lspec = audio.linearspectrogram(wav, hp)
    np.savez_compressed(specpath, spec=spec, lspec=lspec)


def mp_handler(job):
    vfile, args, gpu_id = job
    try:
        success = process_video_file(vfile, args, gpu_id)
        if success:
            process_audio_file(vfile, args, gpu_id)
    except Exception as e:
        print(e, flush=True)


def process(process_id, jobs):
    failed_videos = []
    for vfile in tqdm(jobs):
        try:
            success = process_video_file(vfile, args, process_id, detector='dlib')
            if success:
                process_audio_file(vfile, args, process_id)
            else:
                failed_videos.append(vfile)
        except Exception as e:
            print(e, flush=True)
            failed_videos.append(vfile)

    return failed_videos


def main(args):
    print('Started processing for {} with {} GPUs'.format(args.data_root, args.ngpu))

    filelist = list(glob(path.join(args.data_root, '*/{}/*.mp4'.format(args.split))))

    """
    jobs = [(vfile, args, i%args.ngpu) for i, vfile in enumerate(filelist)]
    p = ThreadPoolExecutor(args.ngpu)
    futures = [p.submit(mp_handler, j) for j in jobs]
    _ = [r.result() for r in tqdm(as_completed(futures), total=len(futures))]
    """

    """
    for i, vfile in tqdm(enumerate(filelist)):
        mp_handler((vfile, args, i%args.ngpu))
    """

    """
    # run dlib face and landmark detectors first on majority
    num_processes = args.ngpu
    num_tasks_per_process = len(filelist) // num_processes
    jobs = []
    for i in range(num_processes):
        start = i * num_tasks_per_process
        end = start + num_tasks_per_process
        jobs.append([i+1, filelist[start:end]])
    all_failed_videos = []
    with mp.Pool(processes=num_processes) as p:
        all_results = p.starmap(process, jobs)
        for failed_videos in all_results:
            all_failed_videos.extend(failed_videos)
    """

    # quickly grab the failed DLIB videos
    from pathlib import Path
    all_failed_videos = []
    for vfile in filelist:
        vidname = os.path.basename(vfile).split('.')[0]
        split = vfile.split('/')[-2]
        word = vfile.split('/')[-3]
        processed_path = Path(args.preprocessed_root).joinpath(word).joinpath(split).joinpath(vidname)
        if not processed_path.exists():
            all_failed_videos.append(vfile)

    print('Num failed so far:', len(all_failed_videos))

    # try GPU face/landmark detector on failed
    failed_videos = []
    final_fail_counter = 0
    for vfile in tqdm(all_failed_videos):
        try:
            success = process_video_file(vfile, args, 0, detector='gpu')
            if success:
                process_audio_file(vfile, args, 0)
            else:
                final_fail_counter += 1
        except Exception as e:
            print(e, flush=True)
            final_fail_counter += 1 
    print('Num failed final:', final_fail_counter)


if __name__ == '__main__':
    main(args)
