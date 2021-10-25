import soundfile as sf

import synthesizer
from synthesizer import inference as sif
import numpy as np
import sys, cv2, os, pickle, argparse, time, subprocess
from tqdm import tqdm
from shutil import copy, copyfile
from glob import glob
from pathlib import Path

from similarity_search import query

from audio_encoder.audio import preprocess_wav

from vocoder.inference import load_model as load_vocoder_model, infer_waveform
load_vocoder_model('_vocoder/saved_models/vocoder_1159k.pt')  # best trained vocoder


class Generator(object):
	def __init__(self):
		super(Generator, self).__init__()

		self.synthesizer = sif.Synthesizer(verbose=False)

	def read_window(self, window_fnames):
		window = []
		for fname in window_fnames:
			img = cv2.imread(fname)
			if img is None:
				raise FileNotFoundError('Frames maybe missing in {}.' 
						' Delete the video to stop this exception!'.format(sample['folder']))

			img = cv2.resize(img, (sif.hparams.img_size, sif.hparams.img_size))
			window.append(img)

		images = np.asarray(window) / 255. # T x H x W x 3
		return images

	def vc(self, args, sample, outfile, vidpath):
		hp = sif.hparams
		id_windows = [range(i, i + hp.T) for i in range(0, (sample['till'] // hp.T) * hp.T, 
					hp.T - hp.overlap) if (i + hp.T <= (sample['till'] // hp.T) * hp.T)]

		all_windows = [[sample['folder'].format(id) for id in window] for window in id_windows]
		last_segment = [sample['folder'].format(id) for id in range(sample['till'])][-hp.T:]
		all_windows.append(last_segment)

		# ref.npz is the speaker embeddings from the sample audio (created by preprocess_speakers.py)
		audio_embeddings_path = os.path.join(os.path.dirname(sample['folder']), 'ref.npz')
		video_embeddings_path = os.path.join(os.path.dirname(sample['folder']), 'video_ref.npz')
		if args.audio_embeddings_path:
			audio_embeddings_path = args.audio_embeddings_path

		if args.find_closest_embeddings and os.path.exists(video_embeddings_path):
			lip_movement_embedding = np.load(video_embeddings_path)['ref'][0]
			audio_embeddings_path, distance = query(lip_movement_embedding)
			# print(sample['folder'].split('/')[-2], audio_embeddings_path.split('/')[-2], distance)
			word, sample_video = audio_embeddings_path.split('/')[-4], audio_embeddings_path.split('/')[-2]
			actual_video_embeddings_path = '/shared/lrw/lipread_mp4/' + word + '/test/' + sample_video + '.mp4'
			if os.path.exists(actual_video_embeddings_path):
				copyfile(actual_video_embeddings_path, vidpath + 'video_embedding.mp4')

		if not os.path.exists(audio_embeddings_path):
			return

		ref = np.load(audio_embeddings_path)['ref'][0]

		# use all the wav samples in the reference file, might be 1, might be 5...
		ref = np.expand_dims(ref, 0)
		# ref = np.zeros([1, 256], dtype=np.float32)

		for window_idx, window_fnames in enumerate(all_windows):
			images = self.read_window(window_fnames)

			s = self.synthesizer.synthesize_spectrograms(images, ref)[0]
			if window_idx == 0:  # if first frame
				mel = s
			elif window_idx == len(all_windows) - 1:  # if last frame
				remaining = ((sample['till'] - id_windows[-1][-1] + 1) // 5) * 16
				if remaining == 0:
					continue
				mel = np.concatenate((mel, s[:, -remaining:]), axis=1)
			else:
				mel = np.concatenate((mel, s[:, hp.mel_overlap:]), axis=1)

		if args.output_mel_specs:
			np.save(vidpath + 'mel-spec.npy', mel)

		wav = self.synthesizer.griffin_lim(mel)
		sif.audio.save_wav(wav, outfile, sr=hp.sample_rate)

		def combine_generated_audio_and_video(audio_path, output_video_path):
			video_path = vidpath + 'preprocessed_video.mp4'
			silent_video_path = vidpath + 'video_no_audio.mp4'
			subprocess.call(f'ffmpeg -hide_banner -loglevel panic -y -i {video_path} -c copy -an {silent_video_path}', shell=True)  # remove audio from video
			subprocess.call(f'ffmpeg -hide_banner -loglevel panic -y -i {silent_video_path} -i {audio_path} -strict -2 -c:v copy -c:a aac {output_video_path}', shell=True)  # combine generated audio and video
			os.remove(silent_video_path)

		if args.combine_audio_video_output:
			combine_generated_audio_and_video(outfile, vidpath + 'generated_video.mp4')

		if args.use_neural_vocoder:
			# generate neural vocoder output
			vocoder_audio_output = vidpath + 'vocoder_generated_audio.wav'
			vocoder_wav = infer_waveform(mel, target=16000)
			vocoder_wav = np.pad(vocoder_wav, (0, 16000), mode='constant')
			vocoder_wav = preprocess_wav(vocoder_wav)
			sf.write(vocoder_audio_output, vocoder_wav.astype(np.float32), 16000)

			combine_generated_audio_and_video(vocoder_audio_output, vidpath + 'vocoder_generated_video.mp4')


def get_vidlist(data_root):
	"""
	test = synthesizer.hparams.get_image_list('test', data_root)
	test_vids = {}
	for x in test:
		x = x[:x.rfind('/')]
		if len(os.listdir(x)) < 30: continue
		test_vids[x] = True
	return list(test_vids.keys())
    """
	import glob

	return glob.glob(os.path.join(data_root, '*.mp4'))


def complete(folder):
	# first check if ref file present
	# if not os.path.exists(os.path.join(folder, 'ref.npz')):
	# 	return False

	frames = glob(os.path.join(folder, '*.jpg'))
	if len(frames) < 25:
		return False

	ids = [int(os.path.splitext(os.path.basename(f))[0]) for f in frames]
	sortedids = sorted(ids)
	if sortedids[0] != 0: return False
	for i, s in enumerate(sortedids):
		if i != s:
			return False
	return True


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('-d', "--data_root", help="Speaker folder path", required=True)
	parser.add_argument('-r', "--results_root", help="Speaker folder path")
	parser.add_argument('--checkpoint', help="Path to trained checkpoint", required=True)
	parser.add_argument('--audio_embeddings_path')
	parser.add_argument('--hide_gt', action='store_true')
	parser.add_argument('--phrases_path')
	parser.add_argument('--output_mel_specs', action='store_true')
	parser.add_argument('--find_closest_embeddings', action='store_true')
	parser.add_argument('--use_neural_vocoder', action='store_true')
	parser.add_argument('--combine_audio_video_output', action='store_true')
	args = parser.parse_args()

	if args.phrases_path:
		with open(args.phrases_path, 'r') as f:
			phrases = f.read().splitlines()
			phrases = sorted([phrase.upper().replace(' ', '_') for phrase in phrases])
			print(phrases)

	sif.hparams.set_hparam('eval_ckpt', args.checkpoint)

	# videos = get_vidlist(args.data_root); print(videos)
	data_root = Path(args.data_root)
	videos = [str(p) for p in data_root.glob('*/test/*')]
	# videos = [str(data_root)]

	RESULTS_ROOT = args.results_root
	if RESULTS_ROOT and not os.path.isdir(RESULTS_ROOT):
		os.mkdir(RESULTS_ROOT)

		GTS_ROOT = os.path.join(RESULTS_ROOT, 'gts/')
		WAVS_ROOT = os.path.join(RESULTS_ROOT, 'wavs/')
		files_to_delete = []
		if not os.path.isdir(GTS_ROOT):
			os.mkdir(GTS_ROOT)
		else:
			files_to_delete = list(glob(GTS_ROOT + '*'))
		if not os.path.isdir(WAVS_ROOT):
			os.mkdir(WAVS_ROOT)
		else:
			files_to_delete.extend(list(glob(WAVS_ROOT + '*')))
		for f in files_to_delete: os.remove(f)

	hp = sif.hparams
	g = Generator()

	average_time, counter = 0, 0
	print('Num videos:', len(videos))
	for vid in tqdm(videos):
		if not complete(vid):
			print(vid, 'not complete')
			continue

		sample = {}
		vidpath = vid + '/'

		sample['folder'] = vidpath + '{}.jpg'

		images = glob(vidpath + '*.jpg')
		sample['till'] = (len(images) // 5) * 5

		vidname = vid.split('/')[-3] + '_' + vid.split('/')[-1]
		if args.hide_gt:
			vidname = f'{phrases.index(vid.split("/")[-3])}_{vid.split("/")[-1]}'
		# outfile = WAVS_ROOT + vidname + '.wav'
		outfile = vidpath + 'generated_audio.wav'

		start_time = time.time()
		g.vc(args, sample, outfile, vidpath)
		took = time.time() - start_time
		average_time += took
		counter += 1

		# don't copy the groundtruth audio to the folder if applicable
		# if not args.hide_gt:
		# 	copy(vidpath + 'audio.wav', GTS_ROOT + vidname + '.wav')

	average_time /= counter
	print(f'Inference took on av. {average_time} seconds')
