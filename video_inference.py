"""
Does speech synthesis on a folder of videos
"""

import argparse
import glob
import os
import re
import shutil
import subprocess
from pathlib import Path

import cv2
import numpy as np
import pystoi
from tqdm import tqdm

import audio_encoder.audio
import face_detection
from audio_encoder import inference as eif
from synthesizer import inference as sif

FFMPEG_OPTIONS = '-hide_banner -loglevel panic'
VIDEO_TO_AUDIO_COMMAND = f'ffmpeg {FFMPEG_OPTIONS} -threads 1 -y -i {{video_path}} -async 1 -ac 1 -vn -acodec pcm_s16le -ar 16000 {{audio_path}}'
VIDEO_CONVERT_FPS_COMMAND = f'ffmpeg {FFMPEG_OPTIONS} -y -i {{input_video_path}} -strict -2 -filter:v fps=fps=25 {{output_video_path}}'
VIDEO_REMOVE_AUDIO_COMMAND = f'ffmpeg {FFMPEG_OPTIONS} -i {{input_video_path}} -c copy -an {{output_video_path}}'
VIDEO_ADD_AUDIO_COMMAND = f'ffmpeg {FFMPEG_OPTIONS} -i {{input_video_path}} -i {{input_audio_path}} -strict -2 -c:v copy -c:a aac {{output_video_path}}'
SECS = 1

fa = face_detection.FaceAlignment(face_detection.LandmarksType._2D, flip_input=False)

encoder_weights = 'audio_encoder/saved_models/pretrained.pt'
eif.load_model(encoder_weights)

synthesizer = None


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


def preprocess_video_file(video_path, output_directory, batch_size, fps=None, neutral_audio_path=None,
                          empty_audio_embedding=False):
    if os.path.exists(output_directory):
        shutil.rmtree(output_directory)
    os.makedirs(output_directory)

    new_video_path = os.path.join(output_directory, f'original_video.mp4')
    if fps:
        subprocess.call(VIDEO_CONVERT_FPS_COMMAND.format(
            input_video_path=video_path,
            output_video_path=new_video_path
        ), shell=True)
    else:
        shutil.copyfile(video_path, new_video_path)
    video_path = new_video_path

    video_stream = cv2.VideoCapture(video_path)
    video_rotation = get_video_rotation(video_path)

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
    if not empty_audio_embedding:
        audio_path = os.path.join(output_directory, 'audio.wav')
        if neutral_audio_path:
            shutil.copyfile(neutral_audio_path, audio_path)
        else:
            # extract audio from video
            subprocess.call(VIDEO_TO_AUDIO_COMMAND.format(
                video_path=video_path,
                audio_path=audio_path
            ), shell=True)

    # get face detections and crop faces to disk
    batches = [frames[i:i + batch_size] for i in range(0, len(frames), batch_size)]
    i = -1
    for fb in batches:
        preds = fa.get_detections_for_batch(np.asarray(fb))

        for j, f in enumerate(preds):
            i += 1
            if f is None:
                continue

            cv2.imwrite(os.path.join(output_directory, '{}.jpg'.format(i)), f[0])
    if i == -1:
        return False  # no face detections

    embeddings_path = os.path.join(output_directory, 'ref.npz')
    if empty_audio_embedding:
        embeddings = np.asarray([np.zeros(256, dtype=np.float32)])
    else:
        # get speaker embedding for audio
        wav = audio_encoder.audio.preprocess_wav(audio_path)  # resamples the audio if required
        if len(wav) < SECS * audio_encoder.audio.sampling_rate:
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

        img = cv2.resize(img, (sif.hparams.img_size, sif.hparams.img_size))
        window.append(img)

    images = np.asarray(window) / 255. # T x H x W x 3

    return images


def speech_synthesis(video_directory, video_format='mp4', combine_audio_and_video=False, debug=False):
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

    if not debug:
        # remove unnecessary files
        os.remove(embeddings_path)
        for image_path in image_paths:
            os.remove(image_path)


def get_stats(video_directory):
    hp = sif.hparams

    gt_wav = sif.audio.load_wav(os.path.join(video_directory, 'audio.wav'), hp.sample_rate)
    pred_wav = sif.audio.load_wav(os.path.join(video_directory, 'generated_audio.wav'), hp.sample_rate)

    if len(gt_wav) > len(pred_wav):
        gt_wav = gt_wav[:pred_wav.shape[0]]
    elif len(pred_wav) > len(gt_wav):
        pred_wav = pred_wav[:gt_wav.shape[0]]

    stoi = pystoi.stoi(gt_wav, pred_wav, hp.sample_rate, extended=False)
    estoi = pystoi.stoi(gt_wav, pred_wav, hp.sample_rate, extended=True)

    return stoi, estoi


def main(args):
    sif.hparams.set_hparam('eval_ckpt', args.model_checkpoint)

    # preprocess videos
    videos_directory = Path(args.videos_directory)
    output_directories = []
    for video_path in tqdm(list(videos_directory.glob(f'*.{args.video_format}'))):
        output_directory = videos_directory.joinpath(video_path.name.replace(f'.{args.video_format}', ''))
        if not args.redo and os.path.exists(output_directory):
            output_directories.append(output_directory)
            continue

        success = preprocess_video_file(str(video_path), output_directory, args.batch_size, args.fps,
                                        args.neutral_audio_path, args.empty_audio_embedding)
        if success:
            output_directories.append(output_directory)
        else:
            shutil.rmtree(output_directory)

    # run synthesis
    global synthesizer
    synthesizer = sif.Synthesizer(verbose=False)
    for output_directory in tqdm(output_directories):
        speech_synthesis(output_directory, args.video_format, args.combine_audio_and_video, args.debug)

    # don't generate stats if using a neutral audio embedding
    if not args.neutral_audio_path and not args.empty_audio_embedding:
        # generate stats for all the videos
        av_stoi, av_estoi = 0, 0
        for output_directory in tqdm(output_directories):
            stoi, estoi = get_stats(output_directory)
            av_stoi += stoi
            av_estoi += estoi
        av_stoi /= len(output_directories)
        av_estoi /= len(output_directories)
        print('Av. STOI:', av_stoi)
        print('Av. ESTOI:', av_estoi)

        with videos_directory.joinpath('stats.txt').open('w') as f:
            f.write(f'Av. STOI: {av_stoi}\n'
                    f'Av. ESTOI: {av_estoi}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('videos_directory')
    parser.add_argument('model_checkpoint')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--fps', type=int, default=None)
    parser.add_argument('--redo', action='store_true')
    parser.add_argument('--video_format', default='mp4')
    parser.add_argument('--combine_audio_and_video', action='store_true')
    parser.add_argument('--neutral_audio_path')  # for extracting a neutral audio embedding
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--empty_audio_embedding', action='store_true')

    main(parser.parse_args())
