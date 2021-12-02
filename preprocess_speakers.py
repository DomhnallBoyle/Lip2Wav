import sys

if sys.version_info[0] < 3 and sys.version_info[1] < 2:
	raise Exception("Must be using >= Python 3.2")

import multiprocessing as mp
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed
from os import listdir, path
import numpy as np
import argparse, os, traceback
from tqdm import tqdm
from glob import glob
import audio_encoder, subprocess
from audio_encoder import inference as eif
from synthesizer import audio as sa
from synthesizer import hparams as hp
from webrtc_vad import *

encoder_weights = 'audio_encoder/saved_models/pretrained.pt'
eif.load_model(encoder_weights)
secs = 1
k = 1


def process_audio_file(afile, args):
	# this preprocessing seems to do multiple things:
	# 1: resample audio
	# 2: normalise volume
	# 3: trim long silences using WebRTC VAD
	#
	# this ensures sampling speaker embeddings from periods of clear speech
	wav = audio_encoder.audio.preprocess_wav(afile)

	if len(wav) < secs * audio_encoder.audio.sampling_rate:
		shutil.rmtree(path.dirname(afile))
		return

	# get speaker embedding for a portion of the audio
	if not args.custom_preprocessing:
		indices = np.random.choice(len(wav) - audio_encoder.audio.sampling_rate * secs, k)  # gets 1 random number since k == 1
		wavs = [wav[idx: idx + audio_encoder.audio.sampling_rate * secs] for idx in indices]  # 1 random sampled wav
		embeddings = np.asarray([eif.embed_utterance(wav) for wav in wavs])  # 1 embedding
	else:
		# custom preprocessing with VAD to select periods of speech
		audio, sample_rate = read_wave(afile)
		vad = webrtcvad.Vad(3)  # most aggressive
		frames = list(frame_generator(30, audio, sample_rate))
		segments = vad_collector(sample_rate, 30, 300, vad, frames)
		for segment in segments:
			with tempfile.NamedTemporaryFile('.wav') as f:
				write_wav(f.name, segment, sample_rate)
				audio_path = f.name
				while True:
					wav = audio_encoder.audio.preprocess_wav(audio_path)
					if len(wav) < secs * audio_encoder.audio.sampling_rate:  # ensure long enough
						break
					else:
						# stitch the same audio to make it longer
						output_audio_path = audio_path.replace('.wav', '_stitched.wav')
						subprocess.call(AUDIO_COMBINE_COMMAND.format(
							input_audio_path_1=audio_path,
							input_audio_path_2=audio_path,
							output_audio_path=output_audio_path
						))
						audio_path = output_audio_path

			# get embedding
			embeddings = np.array([eif.embed_utterance(wav)])

			# remove unnecessary files
			for p in os.path.glob('/tmp/*_stitched.wav'):
				os.remove(p)

			break  # only need 1 segment

	np.savez_compressed(afile.replace('audio.wav', 'ref.npz'), ref=embeddings)

	# extract groundtruth mel-specs from FULL audio file
	# conversion to mel-specs involves:
	# 1: short-time-fourier-transform of the signal in the time-frequency domain
	# 2: converting to decibels from amplitude
	# 3: clipping based on min decibel levels i.e. limits values between min and max (values < min become min and values > max become max)
	wav = sa.load_wav(afile, sr=hp.hparams.sample_rate)
	lspec = sa.linearspectrogram(wav, hp.hparams)
	melspec = sa.melspectrogram(wav, hp.hparams)
	np.savez_compressed(afile.replace('audio.wav', 'mels.npz'), lspec=lspec, spec=melspec)


def mp_handler(job):
	vfile, args = job
	try:
		process_audio_file(vfile, args)
	except KeyboardInterrupt:
		exit(0)
	except:
		traceback.print_exc()


def dump(args):
	print('Started processing for with {} CPU cores'.format(args.num_workers))

	filelist = list(glob(path.join(args.preprocessed_root, args.file_pattern)))

	jobs = [(vfile, args) for vfile in filelist]
	p = ThreadPoolExecutor(args.num_workers)
	futures = [p.submit(mp_handler, j) for j in jobs]
	_ = [r.result() for r in tqdm(as_completed(futures), total=len(futures))]


parser = argparse.ArgumentParser()
parser.add_argument('file_pattern')
parser.add_argument('preprocessed_root', help='Folder where preprocessed files will reside')
parser.add_argument('--num_workers', help='Number of workers to run in parallel', default=8, type=int)
parser.add_argument('--custom_preprocessing', action='store_true')

args = parser.parse_args()

if __name__ == '__main__':
	dump(args)
