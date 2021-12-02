"""
Does speech synthesis on a folder of videos
"""
import argparse
import ast
import itertools
import glob
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import pystoi
import requests
from google.cloud import speech
from jiwer import wer as calculate_wer
from tqdm import tqdm

import audio_encoder.audio
import face_detection
from audio_encoder import inference as eif
from lip_movement_encoder_utils import get_cfe_features, get_lip_movement_embedding as _get_lip_movement_embedding
from preprocess_mouth_roi import s3fd_detector_and_pytorch_landmarks
from similarity_search import query as lip_to_audio_query
from synthesizer import inference as sif

FFMPEG_OPTIONS = '-hide_banner -loglevel panic'
OLD_FFMPEG_VERSION = f'ffmpeg-2.8.15'
# this is a static build from https://johnvansickle.com/ffmpeg/releases/ffmpeg-release-i686-static.tar.xz
# contains x264 codec in build required for clean video frames
NEW_FFMPEG_VERSION = f'/opt/lip2wav/ffmpeg-4.4.1-i686-static/ffmpeg'

# VIDEO_TO_AUDIO_COMMAND = f'{OLD_FFMPEG_VERSION} {FFMPEG_OPTIONS} -threads 1 -y -i {{input_video_path}} -async 1 -ac 1 -vn -acodec pcm_s16le -ar 16000 {{output_audio_path}}'
VIDEO_TO_AUDIO_COMMAND = f'{NEW_FFMPEG_VERSION} {FFMPEG_OPTIONS} -threads 1 -y -i {{input_video_path}} -async 1 -ac 1 -vn -acodec pcm_s16le -ar 16000 {{output_audio_path}}'
VIDEO_CONVERT_FPS_COMMAND = f'{NEW_FFMPEG_VERSION} {FFMPEG_OPTIONS} -y -i {{input_video_path}} -strict -2 -filter:v fps=fps={{fps}} {{output_video_path}}'  # copies original codecs and metadata (rotation)
VIDEO_REMOVE_AUDIO_COMMAND = f'{NEW_FFMPEG_VERSION} {FFMPEG_OPTIONS} -y -i {{input_video_path}} -c copy -an {{output_video_path}}'
VIDEO_ADD_AUDIO_COMMAND = f'{NEW_FFMPEG_VERSION} {FFMPEG_OPTIONS} -y -i {{input_video_path}} -i {{input_audio_path}} -strict -2 -c:v copy -c:a aac {{output_video_path}}'
VIDEO_INFO_COMMAND = f'{NEW_FFMPEG_VERSION} -i {{input_video_path}}'
AUDIO_COMBINE_COMMAND = f'sox {{input_audio_path_1}} {{input_audio_path_2}} {{output_audio_path}}'

# denoising commands
WAV_TO_PCM_COMMAND = f'{NEW_FFMPEG_VERSION} {FFMPEG_OPTIONS} -y -i {{input_audio_path}} -ac 1 -ar 48000 -f s16le -acodec pcm_s16le {{output_audio_path}}'  # 16-bit pcm w/ 48khz
DENOISE_COMMAND = f'/opt/rnnoise/examples/rnnoise_demo {{input_audio_path}} {{output_audio_path}}'
PCM_TO_WAV_COMMAND = f'{NEW_FFMPEG_VERSION} {FFMPEG_OPTIONS} -y -f s16le -ar 48000 -ac 1 -i {{input_audio_path}} {{output_audio_path}}'
WAV_TO_16KHZ_COMMAND = f'{NEW_FFMPEG_VERSION} {FFMPEG_OPTIONS} -y -i {{input_audio_path}} -ac 1 -vn -acodec pcm_s16le -ar 16000 {{output_audio_path}}'

# normalise commands
NORMALISE_AUDIO_COMMAND = f'ffmpeg-normalize -f -q {{input_audio_path}} -o {{output_audio_path}} -ar 16000'
PAD_AUDIO_COMMAND = f'{NEW_FFMPEG_VERSION} {FFMPEG_OPTIONS} -y -i {{input_audio_path}} -af "adelay={{delay}}000|{{delay}}000" {{output_audio_path}}'  # pads audio with delay seconds of silence
REMOVE_AUDIO_PAD_COMMAND = f'{NEW_FFMPEG_VERSION} {FFMPEG_OPTIONS} -y -i {{input_audio_path}} -ss 00:00:{{delay}}.000 -acodec pcm_s16le {{output_audio_path}}'  # removes delay seconds of silence

# lrw cropping commands
CROP_AUDIO_COMMAND = f'{NEW_FFMPEG_VERSION} {FFMPEG_OPTIONS} -y -i {{input_audio_path}} -ss 00:00:00.{{start_millis}} -t 00:00:00.{{duration_millis}} {{output_audio_path}}'

SECS = 1

fa = face_detection.FaceAlignment(face_detection.LandmarksType._2D, flip_input=False)

encoder_weights = 'audio_encoder/saved_models/pretrained.pt'
eif.load_model(encoder_weights)

synthesizer = None


def get_lip_movement_embedding(video_path):
    # NOTE: this is the DTW CFE which uses an old AE model
    # the lip movement encoder was trained using arks from this
    ark_matrix = get_cfe_features(video_path, 'https://51.144.138.184', '/shared/Repos/visual-dtw/azure_cfe.pem')
    if ark_matrix is None:
        return

    return _get_lip_movement_embedding(ark_matrix)


def get_video_rotation(video_path):
    cmd = VIDEO_INFO_COMMAND.format(input_video_path=video_path)

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
        # print(f'Rotation not found: {video_path}')
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


def preprocess_video_file(video_path, output_directory, batch_size, mouth_roi=False, fps=None, _preprocess_audio=False,
                          neutral_audio_path=None, empty_audio_embedding=False, closest_audio_embedding=False):
    if os.path.exists(output_directory):
        shutil.rmtree(output_directory)
    os.makedirs(output_directory)

    # convert FPS if applicable
    new_video_path = os.path.join(output_directory, f'original_video.mp4')
    if fps and fps != get_fps(video_path):
        subprocess.call(VIDEO_CONVERT_FPS_COMMAND.format(
            input_video_path=video_path,
            output_video_path=new_video_path,
            fps=fps
        ), shell=True)
        video_rotation = 0  # ffmpeg auto rotates the video via metadata
    else:
        shutil.copyfile(video_path, new_video_path)
        video_rotation = get_video_rotation(video_path)
    video_path = new_video_path

    video_stream = cv2.VideoCapture(video_path)

    # grab frames and fix rotations
    frames = []
    while 1:
        still_reading, frame = video_stream.read()
        if not still_reading:
            video_stream.release()
            break
        frame = fix_frame_rotation(frame, video_rotation)
        frames.append(frame)

    # don't extract audio if using empty audio embedding
    if not empty_audio_embedding and not closest_audio_embedding:
        audio_path = os.path.join(output_directory, 'audio.wav')
        if neutral_audio_path:
            shutil.copyfile(neutral_audio_path, audio_path)
        else:
            # extract audio from video
            subprocess.call(VIDEO_TO_AUDIO_COMMAND.format(
                input_video_path=video_path,
                output_audio_path=audio_path
            ), shell=True)

        if _preprocess_audio:
            audio_path = preprocess_audio(audio_path=audio_path)

    i = -1
    if mouth_roi:
        # get mouth ROI detections
        mouth_frames = s3fd_detector_and_pytorch_landmarks(frames)
        for mouth_frame in mouth_frames:
            i += 1
            cv2.imwrite(os.path.join(output_directory, '{}.jpg'.format(i)), mouth_frame)
    else:
        # get face detections and crop faces to disk
        batches = [frames[i:i + batch_size] for i in range(0, len(frames), batch_size)]
        for fb in batches:
            preds = fa.get_detections_for_batch(np.asarray(fb))

            for j, f in enumerate(preds):
                i += 1
                if f is None:
                    continue

                cv2.imwrite(os.path.join(output_directory, '{}.jpg'.format(i)), f[0])
    if i == -1:
        print(f'{video_path} has no face detections')
        return False  # no face detections

    embeddings_path = os.path.join(output_directory, 'ref.npz')
    if empty_audio_embedding:
        embeddings = np.asarray([np.zeros(256, dtype=np.float32)])
    elif closest_audio_embedding:
        # get lip movement embedding from lip movement encoder
        lip_movement_embedding = get_lip_movement_embedding(video_path)
        if lip_movement_embedding is None:
            return False

        # perform lookup to find closest audio embedding
        audio_embeddings_path, distance = lip_to_audio_query(lip_movement_embedding=lip_movement_embedding)
        print(video_path, audio_embeddings_path)
        embeddings = np.load(audio_embeddings_path)['ref']
    else:
        # get speaker embedding for audio
        wav = audio_encoder.audio.preprocess_wav(audio_path)  # resamples the audio if required
        if len(wav) < SECS * audio_encoder.audio.sampling_rate:
            print(f'{video_path} audio is too short, {len(wav)} < {SECS * audio_encoder.audio.sampling_rate}')
            return False

        indices = np.random.choice(len(wav) - audio_encoder.audio.sampling_rate * SECS, 1)  # gets 1 random number
        wavs = [wav[idx: idx + audio_encoder.audio.sampling_rate * SECS] for idx in indices]  # 1 random sampled wav
        embeddings = np.asarray([eif.embed_utterance(wav) for wav in wavs])  # 1 embedding
    np.savez_compressed(embeddings_path, ref=embeddings)

    return True


def read_window(window_fnames):
    window = []
    for fname in window_fnames:
        img = cv2.imread(fname)
        if img is None:
            raise FileNotFoundError('Frames maybe missing in {}.'
                                    ' Delete the video to stop this exception!'.format(sample['folder']))

        # resize is (width, height) instead of (height, width)
        img = cv2.resize(img, (sif.hparams.img_width, sif.hparams.img_height))
        window.append(img)

    images = np.asarray(window) / 255.  # T x H x W x 3

    return images


def denoise_audio(audio_path):
    noisy_pcm_path = audio_path.replace('.wav', '.pcm')
    denoised_pcm_path = noisy_pcm_path.replace('.pcm', '_denoised.pcm')
    denoised_wav_path = audio_path.replace('.wav', '_denoised_48khz.wav')
    denoised_wav_path_final = audio_path.replace('.wav', '_denoised.wav')

    subprocess.call(WAV_TO_PCM_COMMAND.format(
        input_audio_path=audio_path,
        output_audio_path=noisy_pcm_path
    ), shell=True)

    subprocess.call(DENOISE_COMMAND.format(
        input_audio_path=noisy_pcm_path,
        output_audio_path=denoised_pcm_path
    ), shell=True)

    subprocess.call(PCM_TO_WAV_COMMAND.format(
        input_audio_path=denoised_pcm_path,
        output_audio_path=denoised_wav_path
    ), shell=True)

    subprocess.call(WAV_TO_16KHZ_COMMAND.format(
        input_audio_path=denoised_wav_path,
        output_audio_path=denoised_wav_path_final
    ), shell=True)

    for path in [noisy_pcm_path, denoised_pcm_path, denoised_wav_path]:
        os.remove(path)

    return denoised_wav_path_final


def pad_audio(audio_path, delay):
    padded_audio_path = audio_path.replace('.wav', '_padded.wav')

    subprocess.call(PAD_AUDIO_COMMAND.format(
        input_audio_path=audio_path,
        delay=delay,
        output_audio_path=padded_audio_path
    ), shell=True)

    return padded_audio_path


def remove_audio_pad(audio_path, delay):
    stripped_audio_path = audio_path.replace('.wav', '_stripped.wav')

    subprocess.call(REMOVE_AUDIO_PAD_COMMAND.format(
        input_audio_path=audio_path,
        delay=delay,
        output_audio_path=stripped_audio_path
    ), shell=True)

    return stripped_audio_path


def normalise_audio(audio_path):
    normalised_audio_path = audio_path.replace('.wav', '_normalised.wav')

    subprocess.call(NORMALISE_AUDIO_COMMAND.format(
        input_audio_path=audio_path,
        output_audio_path=normalised_audio_path
    ), shell=True)

    return normalised_audio_path


def crop_audio(audio_path, start, duration):
    cropped_audio_path = audio_path.replace('.wav', '_cropped.wav')

    subprocess.call(CROP_AUDIO_COMMAND.format(
        input_audio_path=audio_path,
        start_millis=int(start * 1000),
        duration_millis=int(duration * 1000),
        output_audio_path=cropped_audio_path
    ), shell=True)

    return cropped_audio_path


def preprocess_audio(audio_path, delay=3):
    """
    It was found that denoising and then normalising the audio produced louder/more background noise
        - the denoising doesn't work as well on softer audio
        - then the normalising just makes the noise louder

    Normalising and then denoising the audio removed more noise but the sound was only slightly louder
        - normalising first makes the denoising process better
        - normalise again for good measure because the denoising process can make the speaking fainter

    Normalising requires audios >= 3 seconds, pad all with silence and remove after
    See: https://github.com/slhck/ffmpeg-normalize/issues/87
    """
    output_audio_path = audio_path.replace('.wav', '_preprocessed.wav')

    # pad, normalise, denoise, normalise and strip
    padded_audio_path = pad_audio(audio_path, delay=delay)
    normalised_1_audio_path = normalise_audio(padded_audio_path)
    denoised_audio_path = denoise_audio(normalised_1_audio_path)
    normalised_2_audio_path = normalise_audio(denoised_audio_path)
    stripped_audio_path = remove_audio_pad(normalised_2_audio_path, delay=delay)

    shutil.copyfile(stripped_audio_path, output_audio_path)
    for p in [padded_audio_path, normalised_1_audio_path,
              denoised_audio_path, normalised_2_audio_path,
              stripped_audio_path]:
        os.remove(p)

    return output_audio_path


def speech_synthesis(video_directory, video_format='mp4', combine_audio_and_video=False, denoise_output=False,
                     clean_up=False):
    image_paths = glob.glob(os.path.join(video_directory, '*.jpg'))
    till = (len(image_paths) // 5) * 5
    hp = sif.hparams
    image_path = os.path.join(video_directory, '{}.jpg')

    id_windows = [range(i, i + hp.T)
                  for i in range(0, (till // hp.T) * hp.T, hp.T - hp.overlap)
                  if (i + hp.T <= (till // hp.T) * hp.T)]

    all_windows = [[image_path.format(id) for id in window] for window in id_windows]
    last_segment = [image_path.format(id) for id in range(till)][-hp.T:]
    all_windows.append(last_segment)

    embeddings_path = os.path.join(video_directory, 'ref.npz')
    ref = np.load(embeddings_path)['ref'][0]
    ref = np.expand_dims(ref, 0)

    for window_idx, window_fnames in enumerate(all_windows):
        images = read_window(window_fnames)

        s = synthesizer.synthesize_spectrograms(images, ref)[0]
        if window_idx == 0:  # if first frame
            mel = s
        elif window_idx == len(all_windows) - 1:  # if last frame
            remaining = ((till - id_windows[-1][-1] + 1) // 5) * 16
            if remaining == 0:
                continue
            mel = np.concatenate((mel, s[:, -remaining:]), axis=1)
        else:
            mel = np.concatenate((mel, s[:, hp.mel_overlap:]), axis=1)

    generated_output_audio_path = os.path.join(video_directory, 'generated_audio.wav')
    wav = synthesizer.griffin_lim(mel)
    sif.audio.save_wav(wav, generated_output_audio_path, sr=hp.sample_rate)

    if denoise_output:
        # generate denoise for output audio from network
        generated_output_audio_path = denoise_audio(audio_path=generated_output_audio_path)

    if combine_audio_and_video:
        original_video_path = video_directory.joinpath('original_video.mp4')
        tmp_video_path = str(video_directory.joinpath('tmp_video.mp4'))
        subprocess.call(VIDEO_REMOVE_AUDIO_COMMAND.format(
            input_video_path=str(original_video_path),
            output_video_path=tmp_video_path
        ), shell=True)
        subprocess.call(VIDEO_ADD_AUDIO_COMMAND.format(
            input_video_path=tmp_video_path,
            input_audio_path=generated_output_audio_path,
            output_video_path=str(video_directory.joinpath('generated_video.mp4'))
        ), shell=True)
        os.remove(tmp_video_path)

    if clean_up:
        # remove unnecessary files
        os.remove(embeddings_path)
        for image_path in image_paths:
            os.remove(image_path)


def get_stois(directory, denoise_output=False):
    # get STOI and ESTOI
    hp = sif.hparams

    if denoise_output:
        pred_wav_name = 'generated_audio_denoised.wav'
    else:
        pred_wav_name = 'generated_audio.wav'

    gt_wav = sif.audio.load_wav(os.path.join(directory, 'audio.wav'), hp.sample_rate)
    pred_wav = sif.audio.load_wav(os.path.join(directory, pred_wav_name), hp.sample_rate)

    if len(gt_wav) > len(pred_wav):
        gt_wav = gt_wav[:pred_wav.shape[0]]
    elif len(pred_wav) > len(gt_wav):
        pred_wav = pred_wav[:gt_wav.shape[0]]

    stoi = pystoi.stoi(gt_wav, pred_wav, hp.sample_rate, extended=False)
    estoi = pystoi.stoi(gt_wav, pred_wav, hp.sample_rate, extended=True)

    return stoi, estoi


class ASR:

    def __init__(self, name):
        self.name = name
        self.num_candidates = 3

    def recognise(self, audio_content):
        raise NotImplementedException


class GoogleASR(ASR):

    def __init__(self, gcloud_credentials_path, phrases, model, language_code, sample_rate):
        super().__init__(name='Google')
        os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = gcloud_credentials_path
        print(f'Using phrases for ASR speech context:\n{phrases}')
        self.client = speech.SpeechClient()
        self.config = speech.RecognitionConfig(
            encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
            sample_rate_hertz=sample_rate,
            language_code=language_code,
            max_alternatives=self.num_candidates,
            use_enhanced=True,
            model=model,
            speech_contexts=[speech.SpeechContext(phrases=phrases)]
        )

    def recognise(self, audio_content):
        # LINEAR16 = Uncompressed 16-bit signed little-endian samples (Linear PCM).
        # pcm_s16le = PCM signed 16-bit little-endian
        response = self.client.recognize(
            config=self.config,
            audio=speech.RecognitionAudio(content=audio_content)
        )

        return [alternative.transcript.lower().strip()
                for result in response.results
                for alternative in result.alternatives]


class DeepSpeechASR(ASR):

    def __init__(self):
        super().__init__(name='DeepSpeech')
        self.api_endpoint = 'http://speech-recognition:8081/transcribe'

    def recognise(self, audio_content):
        response = requests.post(self.api_endpoint, files={'audio': audio_content},
                                 data={'num_candidates': self.num_candidates})

        return [prediction['transcript'].lower().strip() for prediction in response.json()]


def get_recognition_accuracies(pred_wav_name, parent_output_directory, groundtruth_path, num_synthesised_samples,
                               gcloud_credentials_path=None, deep_speech=False, phrases_path=None,
                               tally_by_keywords=False, lrw=False, language_code='en-GB', model='command_and_search',
                               sample_rate=16000,  calculate_hit_rate=False, asr_excluded_paths=None):
    df = pd.read_csv(groundtruth_path, names=['Video Name', 'Phrase'])
    phrases = list(set(df['Phrase'].values))

    phrases_df = None
    if phrases_path:
        data = []
        with open(phrases_path, 'r') as f:
            for line in f.read().splitlines():
                phrase, key_words = re.match(r'(.+),(\[.+\])', line).groups()
                phrase = phrase.lower().replace('?', '').replace('!', '')
                key_words = ast.literal_eval(key_words)
                data.append([phrase, key_words])
        phrases_df = pd.DataFrame(data=data, columns=['Phrase', 'Key Words'])
        phrases = list(phrases_df['Phrase'].values)

    if gcloud_credentials_path:
        client = GoogleASR(gcloud_credentials_path, phrases, model, language_code, sample_rate)
    elif deep_speech:
        client = DeepSpeechASR()
    else:
        raise Exception('ASR Client not selected')

    tally_by_keywords = phrases_df is not None and tally_by_keywords

    rank_accuracies = [0] * 3
    all_predictions, all_groundtruths, best_wers = [], [], []
    hit_rate = 0

    if lrw:
        pred_wav_name = pred_wav_name.replace('.wav', '_cropped.wav')

    for index, row in tqdm(list(df.iterrows())):
        output_directory = parent_output_directory.joinpath(row['Video Name'])
        if not output_directory.exists():
            continue
        if asr_excluded_paths and row['Video Name'] in asr_excluded_paths:
            num_synthesised_samples -= 1
            continue

        groundtruth = row['Phrase'].lower()
        audio_path = output_directory.joinpath(pred_wav_name)
        asr_results_name = f'{client.name}_asr_results_{audio_path.name.replace(".wav", "")}.txt'
        asr_results_path = output_directory.joinpath(asr_results_name)
        if asr_results_path.exists():
            with asr_results_path.open('r') as f:
                predictions = f.read().splitlines()
            predictions = [prediction for prediction in predictions if prediction]  # check for empty strings
            if not predictions:  # check for empty predictions
                continue
        else:
            if lrw:
                # word appears in middle of audio, crop the audio
                metadata_path = parent_output_directory.parents[0].joinpath(f"{row['Video Name']}.txt")
                with metadata_path.open('r') as f:
                    for line in f.read().splitlines():
                        if not line.lower().startswith('duration'):
                            continue
                        duration = line.split(' ')[1].strip()
                if 'generated' in pred_wav_name:
                    start = 0.99 / 2  # generated audio is ~0.99 seconds
                else:
                    start = 1.16 / 2
                crop_audio(str(audio_path), start, float(duration))

            with open(audio_path, 'rb') as f:
                content = f.read()
                try:
                    predictions = client.recognise(audio_content=content)
                except Exception as e:
                    print(str(audio_path), e)
                    continue

            predictions = [prediction for prediction in predictions if prediction]  # check for empty strings
            # assert 0 <= len(predictions) <= 3, assertion error?
            if not predictions:  # check for empty predictions
                continue

            try:
                with asr_results_path.open('w') as f:
                    for prediction in predictions:
                        f.write(f'{prediction}\n')
            except UnicodeDecodeError as e:
                print(predictions, e)
                continue

        all_predictions.append(predictions[0])  # just use first (most confident) prediction to calculate WER later
        all_groundtruths.append(groundtruth)

        # let's also calculate WER based on all predictions - select the best
        best_wer = np.inf
        for prediction in predictions:
            wer = calculate_wer(groundtruth, prediction)
            if wer < best_wer:
                best_wer = wer
        best_wers.append(best_wer)

        # tally rank accuracies
        if tally_by_keywords:
            try:
                key_word_combos = phrases_df[phrases_df['Phrase'] == groundtruth]['Key Words'].values[0]
            except IndexError as e:
                print(groundtruth, e)
                raise e
            max_score = 0  # capped at 1
            best_prediction_index = None
            for key_words in key_word_combos:
                weight = 1. / len(key_words)  # 1 given for 100% correct
                for i, prediction in enumerate(predictions):
                    score = sum([weight for key_word in key_words if key_word in prediction])
                    if score > max_score:
                        max_score = score
                        best_prediction_index = i
                        if int(max_score) == 1:
                            break
                if int(max_score) == 1:
                    break
            if best_prediction_index is None:
                continue
            for j in range(best_prediction_index, len(rank_accuracies)):
                rank_accuracies[j] += max_score
        else:
            for i, prediction in enumerate(predictions):
                if groundtruth == prediction:
                    for j in range(i, len(rank_accuracies)):
                        rank_accuracies[j] += 1

        if calculate_hit_rate:
            # percentage of times groundtruth appeared anywhere in the predictions
            if any([groundtruth in prediction for prediction in predictions]):
                hit_rate += 1

    assert len(all_groundtruths) == len(all_predictions)

    rank_accuracies = [x / num_synthesised_samples for x in rank_accuracies]  # calculated based on no. synthesised samples
    wer = calculate_wer(all_groundtruths, all_predictions)  # calculated based on synthesised samples with predictions
    av_best_wer = np.mean(best_wers)
    num_samples_with_predictions = len(all_groundtruths)
    num_samples_without_predictions = num_synthesised_samples - num_samples_with_predictions
    asr_success_rate = num_samples_with_predictions / num_synthesised_samples

    results = [
        f'\n------------ {pred_wav_name} ASR --------------',
        f'Rank Accuracies: {rank_accuracies}',
        f'WER: {wer}',
        f'Av. Best WER: {av_best_wer}',
        f'No. Samples w/ ASR predictions: {num_samples_with_predictions}',
        f'No. Samples w/o ASR predictions: {num_samples_without_predictions}',
        f'ASR Success Rate %: {asr_success_rate}'
    ]
    if calculate_hit_rate:
        hit_rate /= num_samples_with_predictions
        results.append(f'Hit Rate %: {hit_rate}')

    return results


def main(args):
    np.random.seed(1234)  # seed the randomizer to extract same audio embeddings

    # set hparams from args
    sif.hparams.set_hparam('eval_ckpt', args.model_checkpoint)
    sif.hparams.set_hparam('img_height', args.image_height)
    sif.hparams.set_hparam('img_width', args.image_width)

    videos_directory = Path(args.videos_directory)
    parent_output_directory = videos_directory.joinpath(args.output_directory)
    parent_output_directory.mkdir(exist_ok=True)

    # save the command used to run this inference script
    with parent_output_directory.joinpath('command.txt').open('w') as f:
        for arg in sys.argv:
            f.write(f'{arg}\n')

    # preprocess videos
    output_directories = []
    print('Preprocessing videos...')
    video_paths_to_process = list(videos_directory.glob(f'*.{args.video_format}'))
    if args.num_samples:
        video_paths_to_process = np.random.choice(video_paths_to_process, args.num_samples, replace=False)
    total_num_samples = len(video_paths_to_process)
    for video_path in tqdm(video_paths_to_process):
        output_directory = parent_output_directory.joinpath(video_path.name.replace(f'.{args.video_format}', ''))
        if not args.redo and os.path.exists(output_directory):
            output_directories.append(output_directory)
            continue

        success = preprocess_video_file(
            video_path=str(video_path),
            output_directory=output_directory,
            batch_size=args.batch_size,
            mouth_roi=args.mouth_roi,
            fps=args.fps,
            _preprocess_audio=args.preprocess_audio,
            neutral_audio_path=args.neutral_audio_path,
            empty_audio_embedding=args.empty_audio_embedding,
            closest_audio_embedding=args.closest_audio_embedding
        )

        if success:
            output_directories.append(output_directory)
        else:
            shutil.rmtree(output_directory)

    num_preprocessed_samples = len(output_directories)
    num_failed_preprocessing_samples = total_num_samples - num_preprocessed_samples

    # run synthesis
    global synthesizer
    synthesizer = sif.Synthesizer(verbose=False)
    print('Running synthesizer...')
    for output_directory in tqdm(output_directories):
        if output_directory.joinpath('generated_audio.wav').exists():
            continue
        speech_synthesis(
            video_directory=output_directory,
            video_format=args.video_format,
            combine_audio_and_video=args.combine_audio_and_video,
            denoise_output=args.denoise_output,
            clean_up=args.clean_up
        )

    stats = [
        f'Total No. Samples: {total_num_samples}',
        f'No. Preprocessed Samples: {num_preprocessed_samples}',
        f'No. Failed Preprocessed Samples: {num_failed_preprocessing_samples}',
    ]

    # don't generate STOI stats if using a different audio embedding from the video
    if not args.neutral_audio_path and not args.empty_audio_embedding and not args.closest_audio_embedding:
        # generate STOI stats for all the videos
        print('Generating STOI stats...')
        av_stoi, av_estoi = 0, 0
        for output_directory in tqdm(output_directories):
            # calculated on samples that exist
            stoi, estoi = get_stois(output_directory, denoise_output=args.denoise_output)
            av_stoi += stoi
            av_estoi += estoi
        av_stoi /= len(output_directories)
        av_estoi /= len(output_directories)
        stats.extend([
            '\n------------ STOI --------------',
            f'Av. STOI: {av_stoi}',
            f'Av. ESTOI: {av_estoi}'
        ])
    else:
        print('Not generating STOI stats because using a different embedding i.e. invalid to compare pred to gt audio')

    # only generate ASR stats if we've groundtruth and google ASR credentials
    groundtruth_path = videos_directory.joinpath('groundtruth.csv')
    run_asr = args.gcloud_credentials_path or args.deep_speech
    if groundtruth_path.exists() and run_asr:

        pred_wav_names = []
        if args.asr_test_groundtruth:
            pred_wav_names += ['audio.wav']
            if args.preprocess_audio:
                pred_wav_names += ['audio_preprocessed.wav']

        if args.denoise_output:
            pred_wav_names += ['generated_audio_denoised.wav']
        else:
            pred_wav_names += ['generated_audio.wav']

        for pred_wav_name in pred_wav_names:
            print(f'Generating ASR stats for {pred_wav_name}...')
            asr_results = get_recognition_accuracies(
                pred_wav_name=pred_wav_name,
                parent_output_directory=parent_output_directory,
                groundtruth_path=groundtruth_path,
                num_synthesised_samples=num_preprocessed_samples,
                gcloud_credentials_path=args.gcloud_credentials_path,
                deep_speech=args.deep_speech,
                phrases_path=args.phrases_path,
                tally_by_keywords=args.tally_by_keywords,
                lrw=args.lrw,
                language_code=args.asr_language_code,
                model=args.asr_model,
                sample_rate=args.asr_sample_rate,
                calculate_hit_rate=args.calculate_hit_rate,
                asr_excluded_paths=args.asr_excluded_paths
            )
            stats.extend(asr_results)

    with parent_output_directory.joinpath('stats.txt').open('w') as f:
        for line in stats:
            f.write(f'{line}\n')

    if args.stats_only:
        # remove all files except stats file
        for file_path in parent_output_directory.glob('*'):
            if file_path.is_file() and file_path.name != 'stats.txt':
                os.remove(file_path)
            elif file_path.is_dir():
                shutil.rmtree(file_path)


if __name__ == '__main__':
    def file_path_contents(p):
        with open(p, 'r') as f:
            return f.read().splitlines()

    parser = argparse.ArgumentParser()
    parser.add_argument('videos_directory')
    parser.add_argument('output_directory')
    parser.add_argument('model_checkpoint')
    parser.add_argument('--mouth_roi', action='store_true')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--fps', type=int, default=25)
    parser.add_argument('--redo', action='store_true')
    parser.add_argument('--video_format', default='mp4')
    parser.add_argument('--combine_audio_and_video', action='store_true')
    parser.add_argument('--neutral_audio_path')  # for extracting a neutral audio embedding
    parser.add_argument('--clean_up', action='store_true')
    parser.add_argument('--empty_audio_embedding', action='store_true')
    parser.add_argument('--preprocess_audio', action='store_true')
    parser.add_argument('--denoise_output', action='store_true')
    parser.add_argument('--closest_audio_embedding', action='store_true')
    parser.add_argument('--stats_only', action='store_true')
    parser.add_argument('--gcloud_credentials_path')
    parser.add_argument('--deep_speech', action='store_true')
    parser.add_argument('--asr_test_groundtruth', action='store_true')
    parser.add_argument('--phrases_path', help='')
    parser.add_argument('--image_height', type=int, default=128, help='Used for mouth ROI')
    parser.add_argument('--image_width', type=int, default=128, help='Used for mouth ROI')
    parser.add_argument('--num_samples', type=int, help='Randomly select this number of samples to generate')
    parser.add_argument('--tally_by_keywords', action='store_true', help='Tally the rank accuracies by keywords')
    parser.add_argument('--lrw', action='store_true')
    parser.add_argument('--asr_language_code', default='en-GB')
    parser.add_argument('--asr_model', default='command_and_search')
    parser.add_argument('--asr_sample_rate', type=int, default=16000)
    parser.add_argument('--calculate_hit_rate', action='store_true')
    parser.add_argument('--asr_excluded_paths', type=file_path_contents)

    # TODO: If 15 fps applied to video, generated_video stays at same pace as original_video
    #  audio is reduced

    main(parser.parse_args())
