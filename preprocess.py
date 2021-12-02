import glob
import sys
import tempfile
import threading
import time

if sys.version_info[0] < 3 and sys.version_info[1] < 2:
	raise Exception("Must be using >= Python 3.2")

from os import listdir, path

if not path.isfile('face_detection/detection/sfd/s3fd.pth'):
	raise FileNotFoundError('Save the s3fd model to face_detection/sfd/s3fd.pth \
							before running this script!')

import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
import requests
import argparse, os, cv2, traceback, subprocess, re, sys, shutil
from tqdm import tqdm
from synthesizer import audio
from synthesizer.hparams import hparams as hp
from pathlib import Path

import face_detection
from video_inference import *

parser = argparse.ArgumentParser()

parser.add_argument('--ngpu', help='Number of GPUs across which to run in parallel', default=1, type=int)
parser.add_argument('--batch_size', help='Single GPU Face detection batch size', default=32, type=int)
parser.add_argument("--data_root", help="Root folder of the LRW dataset", required=True)
parser.add_argument("--preprocessed_root", help="Root folder of the preprocessed dataset", required=True)
parser.add_argument("--split", help="* | train | val | test (* will preprocess all splits)", default="test", 
										choices=["*", "train", "val", "test"])
parser.add_argument('--extract_lip_movement_embeddings', action='store_true')
parser.add_argument('--copy_original_video', action='store_true')
parser.add_argument('--fps', type=int)
parser.add_argument('file_pattern')
parser.add_argument('starting_parent_index', type=int)
parser.add_argument('--preprocess_audio', action='store_true')

args = parser.parse_args()

fa = []
for i in range(args.ngpu):
	fa.append(face_detection.FaceAlignment(face_detection.LandmarksType._2D, flip_input=False, device='cuda:0'))
	time.sleep(5)
	print(f'Started detector {i+1}...')


def process_video_file(vfile, args, gpu_id):
	vidname = os.path.basename(vfile).split('.')[0]
	parent_dirs = '/'.join(vfile.split('/')[args.starting_parent_index:-1])

	fulldir = path.join(args.preprocessed_root, parent_dirs, vidname)
	if not os.path.exists(fulldir):
		os.makedirs(fulldir)

	# convert FPS is applicable
	convert_fps = args.fps is not None and args.fps != get_fps(vfile)
	if convert_fps:
		print('Original FPS:', get_fps(vfile))
		tmp_vfile = tempfile.NamedTemporaryFile(suffix='.mp4')
		subprocess.call(VIDEO_CONVERT_FPS_COMMAND.format(
			input_video_path=vfile,
			output_video_path=tmp_vfile.name,
			fps=args.fps
		), shell=True)
		vfile = tmp_vfile.name
		print('New FPS:', get_fps(vfile))
		# ffmpeg > v2.7 auto rotates video sources with rotation metadata
		# this is done in the fps conversion
		video_rotation = 0
	else:
		video_rotation = get_video_rotation(vfile)

	# copy original video
	if args.copy_original_video:
		shutil.copyfile(vfile, path.join(fulldir, 'original_video.mp4'))

	# ----------------------------- lip2wav preprocessing steps -----------------------------

	try:
		wavpath = path.join(fulldir, 'audio.wav')
		subprocess.call(VIDEO_TO_AUDIO_COMMAND.format(input_video_path=vfile, output_audio_path=wavpath), shell=True)
		if args.preprocess_audio:
			preprocessed_wavpath = preprocess_audio(audio_path=wavpath)
			shutil.move(preprocessed_wavpath, wavpath)
	except Exception as e:
		shutil.rmtree(fulldir)
		raise e

	video_stream = cv2.VideoCapture(vfile)

	frames = []
	while 1:
		still_reading, frame = video_stream.read()
		if not still_reading:
			video_stream.release()
			break
		frame = fix_frame_rotation(frame, video_rotation)
		frames.append(frame)

	batches = [frames[i:i + args.batch_size] for i in range(0, len(frames), args.batch_size)]
	i = -1
	for fb in batches:
		preds = fa[gpu_id].get_detections_for_batch(np.asarray(fb))

		for j, f in enumerate(preds):
			i += 1
			if f is None:
				continue

			cv2.imwrite(path.join(fulldir, '{}.jpg'.format(i)), f[0])

	# get speaker embedding for video file
	if args.extract_lip_movement_embeddings:
		lip_movement_embedding = get_lip_movement_embedding(vfile)
		if lip_movement_embedding is not None:
			np.savez_compressed(path.join(fulldir, 'lip_ref.npz'), ref=lip_movement_embedding)

	if convert_fps:
		tmp_vfile.close()

	
def mp_handler(job):
	vfile, args, gpu_id = job
	try:
		process_video_file(vfile, args, gpu_id)
	except KeyboardInterrupt:
		exit(0)
	except:
		traceback.print_exc()


def main(args):
	print('Started processing for {} with {} GPUs'.format(args.data_root, args.ngpu))

	filelist = glob.glob(os.path.join(args.data_root, args.file_pattern))

	print('Num videos:', len(list(filelist)))

	jobs = [(vfile, args, i % args.ngpu) for i, vfile in enumerate(filelist)]
	p = ThreadPoolExecutor(args.ngpu)
	futures = [p.submit(mp_handler, j) for j in jobs]
	_ = [r.result() for r in tqdm(as_completed(futures), total=len(futures))]


if __name__ == '__main__':
	main(args)
