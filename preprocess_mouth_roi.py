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

parser = argparse.ArgumentParser()

parser.add_argument('--ngpu', help='Number of GPUs across which to run in parallel', default=1, type=int)
parser.add_argument('--batch_size', help='Single GPU Face detection batch size', default=32, type=int)
parser.add_argument("--data_root", help="Root folder of the LRW dataset", required=True)
parser.add_argument("--preprocessed_root", help="Root folder of the preprocessed dataset", required=True)
parser.add_argument("--split", help="* | train | val | test (* will preprocess all splits)", default="test",
                    choices=["*", "train", "val", "test"])


args = parser.parse_args()

# fa = [face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, flip_input=False,
#                                    device='cpu') for _ in range(args.ngpu)]  # S3FD face detectors GPU
# fa = [face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, flip_input=False,
#                                    device='cuda:{}'.format(id)) for id in range(args.ngpu)]  # S3FD face detectors GPU
fa = [face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, flip_input=False,
                                   device='cuda:{}'.format(0)) for _ in range(args.ngpu)]  # S3FD face detectors GPU
# fa = [face_detection.FaceAlignment(face_detection.LandmarksType._2D, flip_input=False,
#                                    device='cuda:{}'.format(id)) for id in range(args.ngpu)]  # S3FD face detectors GPU
# face_detectors = [dlib.get_frontal_face_detector() for _ in range(args.ngpu)]  # CPU HOG Face detectors
shape_predictors = [dlib.shape_predictor('shape_predictor_68_face_landmarks.dat') for _ in range(args.ngpu)]  # CPU shape predictors based on landmarks

# template = 'ffmpeg -loglevel panic -y -i {} -ar {} -f wav {}'
template2 = 'ffmpeg -hide_banner -loglevel panic -threads 1 -y -i {} -async 1 -ac 1 -vn -acodec pcm_s16le -ar 16000 {}'

"""3 ways of face and landmark detection:
1: DLIB detector (CPU) and DLIB landmarks (CPU) - very fast w/ multiple cores
2: S3FD face detector (GPU) and DLIB landmarks (CPU) - 1.3 videos per second, not as fast as 1
3: S3FD face detector (GPU) and Pytorch landmarks (GPU) - ~20FPS
"""

def get_frames_mouth(detector, predictor, frames):
    MOUTH_WIDTH = 100
    MOUTH_HEIGHT = 50
    HORIZONTAL_PAD = 0.19
    normalize_ratio = None
    mouth_frames = []
    for frame in frames:
        # dets = detector(frame, 1)
        pred = detector.get_detections_for_batch(np.array([frame]))[0]
        dets = [dlib.rectangle(*pred[1:])]
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

        mouth_frames.append(mouth_crop_image)
    return mouth_frames


def process_video_file(vfile, args, gpu_id):
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
    subprocess.call(command, shell=True)

    # this works but is quite slow because only 1 GPU
    # takes approx 1 second per video
    # takes longer on CPU OMG!!!
    all_landmarks = []
    for frame in frames:
        landmarks = fa[gpu_id].get_landmarks_from_image(np.asarray(frame))[0]
        all_landmarks.append(landmarks)

    if len(frames) != len(all_landmarks):
        shutil.rmtree(fulldir)
        return False

    MOUTH_WIDTH = 100
    MOUTH_HEIGHT = 50
    HORIZONTAL_PAD = 0.19
    normalize_ratio = None
    mouth_frames = []
    for frame, landmarks in zip(frames, all_landmarks):
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
        mouth_frames.append(mouth_crop_image)

    # # this works and is super fast because of multiple CPU cores
    # mouth_frames = get_frames_mouth(fa[gpu_id], shape_predictors[gpu_id], frames)

    if len(mouth_frames) == 0 or not all([mouth_frame.shape == (50, 100, 3) for mouth_frame in mouth_frames]):
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
        process_video_file(vfile, args, gpu_id)
        process_audio_file(vfile, args, gpu_id)
    except KeyboardInterrupt:
        exit(0)
    except:
        traceback.print_exc()

def main(args):
    print('Started processing for {} with {} GPUs'.format(args.data_root, args.ngpu))

    filelist = glob(path.join(args.data_root, '*/{}/*.mp4'.format(args.split)))

    jobs = [(vfile, args, i%args.ngpu) for i, vfile in enumerate(filelist)]
    p = ThreadPoolExecutor(args.ngpu)
    futures = [p.submit(mp_handler, j) for j in jobs]
    _ = [r.result() for r in tqdm(as_completed(futures), total=len(futures))]

if __name__ == '__main__':
    main(args)