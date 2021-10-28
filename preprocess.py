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
import requests
import argparse, os, cv2, traceback, subprocess, re, sys, shutil
from tqdm import tqdm
from glob import glob
from synthesizer import audio
from synthesizer.hparams import hparams as hp
from pathlib import Path
# from lip_movement_encoder_utils import get_lip_movement_embedding, get_cfe_features

import face_detection
from video_inference import VIDEO_CONVERT_FPS_COMMAND

# sys.path.append('/shared/Repos/video_harvesting/app/main/services/sync_net')
# from main.utils.preprocessing import extract_audio, reset_scale_and_frame_rate, preprocess_video_and_audio

parser = argparse.ArgumentParser()

parser.add_argument('--ngpu', help='Number of GPUs across which to run in parallel', default=1, type=int)
parser.add_argument('--batch_size', help='Single GPU Face detection batch size', default=32, type=int)
parser.add_argument("--data_root", help="Root folder of the LRW dataset", required=True)
parser.add_argument("--preprocessed_root", help="Root folder of the preprocessed dataset", required=True)
parser.add_argument("--split", help="* | train | val | test (* will preprocess all splits)", default="test", 
										choices=["*", "train", "val", "test"])
parser.add_argument('--extract_lip_movement_embeddings', action='store_true')
parser.add_argument('--lrw_preprocessing', action='store_true')
parser.add_argument('--copy_original_video', action='store_true')
parser.add_argument('--fps', type=int)

args = parser.parse_args()

fa = [face_detection.FaceAlignment(face_detection.LandmarksType._2D, flip_input=False,
								   device='cuda:{}'.format(id)) for id in range(args.ngpu)]

# template = 'ffmpeg -loglevel panic -y -i {} -ar {} -f wav {}'
template2 = 'ffmpeg -hide_banner -loglevel panic -threads 1 -y -i {} -async 1 -ac 1 -vn -acodec pcm_s16le -ar 16000 {}'


def get_video_rotation(video_path, debug=False):
	cmd = f'ffmpeg -i {video_path}'

	p = subprocess.Popen(
		cmd.split(' '),
		stderr=subprocess.PIPE,
		close_fds=True
	)
	stdout, stderr = p.communicate()

	try:
		reo_rotation = re.compile('rotate\s+:\s(\d+)')
		match_rotation = reo_rotation.search(str(stderr))
		rotation = match_rotation.groups()[0]
	except AttributeError:
		if debug:
			print(f'Rotation not found: {video_path}')
		return 0

	return int(rotation)


def fix_frame_rotation(image, rotation):
	if rotation == 90:
		image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
	elif rotation == 180:
		image = cv2.rotate(image, cv2.ROTATE_180)
	elif rotation == 270:
		image = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)

	return image


def get_fps(video_path):
	video_capture = cv2.VideoCapture(video_path)
	fps = int(video_capture.get(cv2.CAP_PROP_FPS))
	video_capture.release()

	return fps


def process_video_file(vfile, args, gpu_id):
	vidname = os.path.basename(vfile).split('.')[0]
	user_id = vfile.split('/')[-2]
	# split = vfile.split('/')[-2]
	# word = vfile.split('/')[-3]
	# fulldir = path.join(args.preprocessed_root, word, split, vidname)
	fulldir = path.join(args.preprocessed_root, user_id, vidname)
	os.makedirs(fulldir, exist_ok=True)

	# convert FPS is applicable
	convert_fps = args.fps and args.fps != get_fps(vfile)
	if convert_fps:
		print('Original FPS:', get_fps(vfile))
		tmp_vfile = vfile.replace('.mp4', f'-{args.fps}fps.mp4')
		subprocess.call(VIDEO_CONVERT_FPS_COMMAND.format(
			input_video_path=vfile,
			output_video_path=tmp_vfile,
			fps=args.fps
		), shell=True)
		vfile = tmp_vfile
		print('New FPS:', get_fps(vfile))

	# copy original video
	if args.copy_original_video:
		shutil.copyfile(vfile, path.join(fulldir, 'original_video.mp4'))

	if args.lrw_preprocessing:
		# ----------------------------- detection and tracking -----------------------------
		with open(vfile, 'rb') as f:
			response = requests.post('http://172.20.0.2:8083/detect', files={'video': f.read()})
			assert response.status_code == 200
		frame_detections = response.json()
		track = []
		for frame_id, detections in frame_detections.items():
			for person_id, detection in detections.items():
				track.append({
					k: v for k, v in zip(['x1', 'y1', 'x2', 'y2'], detection)
				})
		track = [[d['x1'], d['y1'], d['x2'], d['y2']] for d in track]

		# ----------------------------- sync preprocessing -----------------------------
		# TODO: Follow this entire process using Liam's "Tell me about your day" video
		#  i.e. watch /tmp directory .mp4 .avi files

		tmp_dir = '/tmp/syncnet_preprocessed'
		if os.path.exists(tmp_dir):
			shutil.rmtree(tmp_dir)
		os.makedirs(tmp_dir)
		scaled_video_path = os.path.join(tmp_dir, 'video_scaled.avi')
		audio_path = os.path.join(tmp_dir, 'extracted_audio.wav')
		preprocessed_video = os.path.join(tmp_dir, 'video_preprocessed.avi')
		preprocessed_audio = os.path.join(tmp_dir, 'audio_preprocessed.wav')
		combined_output_path = os.path.join(tmp_dir, 'video_combined.avi')
		combined_output_path_mp4 = os.path.join(tmp_dir, 'video_combined.mp4')
		reset_scale_and_frame_rate(vfile, scaled_video_path)
		extract_audio(scaled_video_path, audio_path)
		preprocess_video_and_audio(
			video_input_path=scaled_video_path,
			video_output_path=preprocessed_video,
			track=track,
			audio_input_path=audio_path,
			audio_output_path=preprocessed_audio,
			combined_output_path=combined_output_path,
			height=256,
			width=256,
			crop_scale=0.6
		)
		# convert .avi to .mp4
		subprocess.call(f'ffmpeg -hide_banner -loglevel panic -i {combined_output_path} -strict -2 {combined_output_path_mp4}', shell=True)

		# copy preprocessed video
		vfile = path.join(fulldir, 'preprocessed_video.mp4')
		shutil.copyfile(combined_output_path_mp4, vfile)

	# ----------------------------- lip2wav preprocessing steps -----------------------------
	video_stream = cv2.VideoCapture(vfile)
	video_rotation = get_video_rotation(vfile)

	frames = []
	while 1:
		still_reading, frame = video_stream.read()
		if not still_reading:
			video_stream.release()
			break
		frame = fix_frame_rotation(frame, video_rotation)
		frames.append(frame)

	wavpath = path.join(fulldir, 'audio.wav')
	specpath = path.join(fulldir, 'mels.npz')

	command = template2.format(vfile, wavpath)
	subprocess.call(command, shell=True)

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
		# NOTE: this is the DTW CFE which uses an old AE model
		# the lip movement encoder was trained using arks from this
		ark_matrix = get_cfe_features(vfile, 'https://51.144.138.184', '/shared/Repos/visual-dtw/azure_cfe.pem')
		if ark_matrix is None:
			return
		lip_movement_embeddings = get_lip_movement_embedding(ark_matrix)
		np.savez_compressed(path.join(fulldir, 'video_ref.npz'), ref=lip_movement_embeddings)

	if convert_fps:
		os.remove(vfile)


def process_audio_file(vfile, args, gpu_id):
	vidname = os.path.basename(vfile).split('.')[0]
	user_id = vfile.split('/')[-2]
	# split = vfile.split('/')[-2]
	# word = vfile.split('/')[-3]

	# fulldir = path.join(args.preprocessed_root, word, split, vidname)
	# fulldir = args.preprocessed_root
	fulldir = path.join(args.preprocessed_root, user_id, vidname)
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
		# process_audio_file(vfile, args, gpu_id)
	except KeyboardInterrupt:
		exit(0)
	except:
		traceback.print_exc()
		
def main(args):
	print('Started processing for {} with {} GPUs'.format(args.data_root, args.ngpu))

	# filelist = glob(path.join(args.data_root, '*/{}/*.mp4'.format(args.split)))
	# filelist = glob(path.join(args.data_root, '*.mp4'))
	# filelist = glob(path.join(args.data_root, '*/*.mpg'))
	filelist = list(glob(path.join(args.data_root, '*/*.mp4')))

	print('Num videos:', len(list(filelist)))

	jobs = [(vfile, args, i%args.ngpu) for i, vfile in enumerate(filelist)]
	p = ThreadPoolExecutor(args.ngpu)
	futures = [p.submit(mp_handler, j) for j in jobs]
	_ = [r.result() for r in tqdm(as_completed(futures), total=len(futures))]

if __name__ == '__main__':
	main(args)
