"""
Does speech synthesis on a folder of videos
"""
import argparse
import ast
import itertools
import glob
import multiprocessing
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import pystoi
import requests
from jiwer import wer as calculate_wer
from tqdm import tqdm

from asr import DeepSpeechASR, GoogleASR
from audio_utils import crop_audio
from preprocessor import process_video as preprocess_video
from similarity_search import query as lip_query
from synthesizer import inference as sif
from text_utils import expand_contractions
from video_utils import extract_audio, get_video_duration, replace_audio


def read_window(window_fnames, read_option=1, normalise_frames=False):
    window = []
    for fname in window_fnames:
        img = cv2.imread(fname, read_option)  # 0 = greyscale, 1 = colour
        if img is None:
            raise FileNotFoundError('Frames maybe missing in {}.'
                                    ' Delete the video to stop this exception!'.format(sample['folder']))

        # resize is (width, height) instead of (height, width)
        if img.shape[:2] != (sif.hparams.img_height, sif.hparams.img_width):
            img = cv2.resize(img, (sif.hparams.img_width, sif.hparams.img_height))
        window.append(img)

    if normalise_frames:
        # normalize frames to 0-255 uint8 dtype
        window = [
            cv2.normalize(src=frame, dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
            for frame in window
        ]
    images = (np.asarray(window) / 255.).astype(np.float32)  # T x H x W x C
    if read_option == 0:
        images = images[..., np.newaxis]

    return images


def speech_synthesis(synthesizer, video_directory, normalise_frames=False, combine_audio_and_video=False, clean_up=False,
                     save_alignments=False, save_mels=False, save_wavs=False):
    image_paths = glob.glob(os.path.join(video_directory, '*.png'))
    till = (len(image_paths) // 5) * 5
    hp = sif.hparams
    image_path = os.path.join(video_directory, '{}.png')

    id_windows = [range(i, i + hp.T)
                  for i in range(0, (till // hp.T) * hp.T, hp.T - hp.overlap)
                  if (i + hp.T <= (till // hp.T) * hp.T)]

    all_windows = [[image_path.format(id) for id in window] for window in id_windows]
    last_segment = [image_path.format(id) for id in range(till)][-hp.T:]
    all_windows.append(last_segment)

    embeddings_path = os.path.join(video_directory, 'ref.npz')
    ref = np.load(embeddings_path)['ref'][0].astype(np.float32)  # just pick the first embedding
    ref = np.expand_dims(ref, 0)
    
    read_option = 0 if sif.hparams.num_channels == 1 else 1

    for window_idx, window_fnames in enumerate(all_windows):
        images = read_window(window_fnames, read_option=read_option, normalise_frames=normalise_frames)

        result = synthesizer.synthesize_spectrograms(images, ref, return_alignments=save_alignments)
        if save_alignments:
            s, alignments = result
            alignments = alignments[0]
            np.save(os.path.join(video_directory, f'alignments_{window_idx}.npy'), alignments, allow_pickle=False)
        else:
            s = result

        s = s[0]

        if save_mels:
            np.save(os.path.join(video_directory, f'mel_{window_idx}.npy'), s, allow_pickle=False)

        if save_wavs:
            wav = synthesizer.griffin_lim(s)
            sif.audio.save_wav(wav, os.path.join(video_directory, f'wav_{window_idx}.wav'), sr=hp.sample_rate)

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
    generated_output_mel_path = os.path.join(video_directory, 'generated_mel.npy')

    wav = synthesizer.griffin_lim(mel)
    sif.audio.save_wav(wav, generated_output_audio_path, sr=hp.sample_rate)
    np.save(generated_output_mel_path, mel, allow_pickle=False)

    if combine_audio_and_video:
        replace_audio(
            video_path=str(video_directory.joinpath('original_video.mp4')),
            audio_path=generated_output_audio_path,
            output_video_path=str(video_directory.joinpath('generated_video.mp4'))
        )

    if clean_up:
        # remove unnecessary files
        os.remove(embeddings_path)
        for image_path in image_paths:
            os.remove(image_path)


def get_stois(directory):
    # get STOI and ESTOI
    hp = sif.hparams

    gt_wav = sif.audio.load_wav(os.path.join(directory, 'audio.wav'), hp.sample_rate)
    pred_wav = sif.audio.load_wav(os.path.join(directory, 'generated_audio.wav'), hp.sample_rate)

    if len(gt_wav) > len(pred_wav):
        gt_wav = gt_wav[:pred_wav.shape[0]]
    elif len(pred_wav) > len(gt_wav):
        pred_wav = pred_wav[:gt_wav.shape[0]]

    stoi = pystoi.stoi(gt_wav, pred_wav, hp.sample_rate, extended=False)
    estoi = pystoi.stoi(gt_wav, pred_wav, hp.sample_rate, extended=True)

    return stoi, estoi


def _round(x):
    return round(x * 100, 1)


def asr_metric(sr, wer):
    return round(((sr * wer) + ((100 - sr) * 100)) / 100, 1)


def get_recognition_accuracies(pred_wav_name, parent_output_directory, groundtruth_path, num_synthesised_samples,
                               gcloud_credentials_path=None, deep_speech=False, phrases_path=None,
                               tally_by_keywords=False, lrw=False, language_code='en-GB', model='command_and_search',
                               sample_rate=16000,  calculate_hit_rate=False, asr_included_paths=None,
                               asr_excluded_paths=None):
    use_alternative_gts = False
    data = []
    with open(groundtruth_path, 'r') as f:
        for line in f.read().splitlines():
            video_name, phrase, alternatives = re.match(r'([a-zA-Z0-9-_]+),([a-z0-9 \']+),?(\[.+\])?', line).groups()
            if alternatives is not None:
                alternatives = ast.literal_eval(alternatives)
                use_alternative_gts = True
            data.append([video_name, phrase, alternatives])
    df = pd.DataFrame(data=data, columns=['Video Name', 'Phrase', 'Alternatives'])
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
    best_wers_top_1, best_wers_top_3 = [], []

    if lrw:
        new_pred_wav_name = pred_wav_name.replace('.wav', '_cropped.wav')
    else:
        new_pred_wav_name = pred_wav_name

    asr_times = []
    for index, row in tqdm(list(df.iterrows())):
        output_directory = parent_output_directory.joinpath(row['Video Name'])
        if not output_directory.exists():
            print(output_directory, 'doesn\'t exist')
            continue
        if (asr_included_paths and row['Video Name'] not in asr_included_paths) or \
                (asr_excluded_paths and row['Video Name'] in asr_excluded_paths):
            num_synthesised_samples -= 1
            continue

        groundtruth = row['Phrase'].lower()
        audio_path = output_directory.joinpath(pred_wav_name)

        asr_results_name = f'{client.name}_asr_results_{new_pred_wav_name.replace(".wav", "")}.txt'
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
                cropped_audio_path = output_directory.joinpath(new_pred_wav_name)
                if cropped_audio_path.exists():
                    audio_path = str(cropped_audio_path)
                else:
                    metadata_path = parent_output_directory.parents[0].joinpath(f"{row['Video Name']}.txt")
                    with metadata_path.open('r') as f:
                        for line in f.read().splitlines():
                            if not line.lower().startswith('duration'):
                                continue
                            duration = float(line.split(' ')[1].strip())

                    offset = 0.06

                    if 'generated' in new_pred_wav_name:
                        start = (0.99 / 2) - offset  # generated audio is ~0.99 seconds
                    else:
                        start = (1.22 / 2) - offset

                    if duration <= 0.353:
                        duration = (duration * 1.77) - offset

                    audio_path = crop_audio(str(audio_path), start, duration)

            start_time = time.time()
            try:
                predictions = client.run(audio_path=audio_path)
            except Exception as e:
                print(str(audio_path), e)
                continue
            asr_times.append(time.time() - start_time)

            predictions = [prediction for prediction in predictions if prediction]  # check for empty strings
            # assert 0 <= len(predictions) <= 3, assertion error?
            if not predictions:  # check for empty predictions
                continue

            with asr_results_path.open('w') as f:
                for prediction in predictions:
                    try:
                        f.write(f'{prediction}\n')
                    except UnicodeEncodeError as e:
                        prediction_converted = prediction.encode('utf-8').decode('utf-8')
                        print(prediction_converted, e)
                        f.write(f'{prediction_converted}\n')

        groundtruth_contractions = expand_contractions(groundtruth)
        predictions = [expand_contractions(pred) for pred in predictions]

        all_predictions.append(predictions[0])  # just use first (most confident) prediction to calculate WER later
        all_groundtruths.append(groundtruth_contractions)

        # let's also calculate WER based on all predictions - select the best
        best_wer = np.inf
        for prediction in predictions:
            wer = calculate_wer(groundtruth_contractions, prediction)
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
                if groundtruth_contractions == prediction:
                    for j in range(i, len(rank_accuracies)):
                        rank_accuracies[j] += 1

        if calculate_hit_rate:
            # percentage of times groundtruth appeared anywhere in the predictions
            if any([groundtruth_contractions in prediction for prediction in predictions]):
                hit_rate += 1

        if use_alternative_gts:
            # calculate Top 1 and 3 WERs based on alternative groundtruths too (if any)
            best_wer_top_1, best_wer_top_3 = np.inf, np.inf
            gts = [groundtruth_contractions] + row['Alternatives']
            for gt in gts:
                for j, pred in enumerate(predictions):
                    wer = calculate_wer(gt, pred)
                    if j == 0 and wer < best_wer_top_1:
                        best_wer_top_1 = wer
                    if wer < best_wer_top_3:
                        best_wer_top_3 = wer
            best_wers_top_1.append(best_wer_top_1)
            best_wers_top_3.append(best_wer_top_3)

    assert len(all_groundtruths) == len(all_predictions)

    if len(all_groundtruths) == len(all_predictions) == 0:
        return [
            f'\n------------ {new_pred_wav_name} {client.name} ASR --------------',
            'No predictions from audio'
        ]

    rank_accuracies = [_round(x / num_synthesised_samples) for x in rank_accuracies]  # calculated based on no. synthesised samples
    wer = _round(calculate_wer(all_groundtruths, all_predictions)) if all_predictions else None  # calculated based on synthesised samples with predictions
    av_best_wer = _round(np.mean(best_wers)) if all_predictions else None
    num_samples_with_predictions = len(all_groundtruths)
    num_samples_without_predictions = num_synthesised_samples - num_samples_with_predictions
    asr_success_rate = _round(num_samples_with_predictions / num_synthesised_samples)
    asr_score_top_1 = asr_metric(asr_success_rate, wer)
    asr_score_top_3 = asr_metric(asr_success_rate, av_best_wer)

    print('Av. ASR Time (secs):', round(np.mean(asr_times), 1))

    results = [
        f'\n------------ {new_pred_wav_name} {client.name} ASR --------------',
        f'Rank Accuracies: {rank_accuracies}',
        f'WER: {wer} / {av_best_wer}',
        f'No. Samples w/ ASR predictions: {num_samples_with_predictions}',
        f'No. Samples w/o ASR predictions: {num_samples_without_predictions}',
        f'ASR Success Rate %: {asr_success_rate}',
        f'ASR Score %: {asr_score_top_1} / {asr_score_top_3}'
    ]
    if calculate_hit_rate:
        hit_rate = _round(hit_rate / num_samples_with_predictions) if all_predictions else None
        results.append(f'Hit Rate %: {hit_rate}')
    if use_alternative_gts:
        av_wer_top_1 = _round(np.mean(best_wers_top_1))
        av_wer_top_3 = _round(np.mean(best_wers_top_3))
        asr_score_top_1 = asr_metric(asr_success_rate, av_wer_top_1)
        asr_score_top_3 = asr_metric(asr_success_rate, av_wer_top_3)
        results.append(f'WER (w/ alternatives): {av_wer_top_1} / {av_wer_top_3}')
        results.append(f'ASR Score % (w/ alternatives): {asr_score_top_1} / {asr_score_top_3}')

    return results


def preprocess_wrapper(process_index, args, tasks):
    output_directories, video_durations, preprocessing_times = [], [], []
    for output_directory, video_path in tqdm(tasks):
        speaker_embedding_audio_file = None

        if args.neutral_audio_path:
            speaker_embedding_audio_file = tempfile.NamedTemporaryFile(suffix='.wav')
            with open(args.neutral_audio_path, 'rb') as f:
                speaker_embedding_audio_file.write(f.read())
            speaker_embedding_audio_file.seek(0)

        if args.neutral_audio_from_video_path:
            speaker_embedding_audio_file = extract_audio(video_path=args.neutral_audio_from_video_path,
                                                         use_old_ffmpeg=args.use_old_ffmpeg)

        if args.closest_audio_embedding:
            results = lip_query(video_path=str(video_path),
                                by_content=args.by_content,
                                by_rate_of_speech=args.by_rate_of_speech,
                                audio_preprocessing=args.audio_preprocessing,
                                exclude_speaker=True,
                                use_old_ffmpeg=args.use_old_ffmpeg)
            if results is None:
                shutil.rmtree(str(output_directory))
                continue
            closest_video_path = results[0][0]
            # print(process_index, str(video_path), results)
            speaker_embedding_audio_file = extract_audio(closest_video_path)

        start_time = time.time()
        result = preprocess_video(
            process_index=process_index,
            video_path=str(video_path),
            fps=args.fps,
            output_directory=output_directory,
            audio_preprocessing=args.audio_preprocessing,
            speaker_embedding_audio_file=speaker_embedding_audio_file,
            use_old_ffmpeg=args.use_old_ffmpeg,
            use_old_mouth_extractor=args.use_old_mouth_extractor,
            use_perspective_warp=args.use_perspective_warp,
            use_cfe_cropper=args.use_cfe_cropper,
            greyscale=args.greyscale,
            blur_frames=args.blur_frames,
            cfe_host=args.cfe_host,
            smooth_frames=args.smooth_frames,
            debug=args.debug
        )
        if result is None:
            shutil.rmtree(str(output_directory))
            continue

        preprocessing_times.append(time.time() - start_time)
        video_durations.append(get_video_duration(str(video_path)))
        output_directories.append(output_directory)

    return output_directories, video_durations, preprocessing_times


def inference(args):
    if args.cpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

    np.random.seed(1234)  # seed the randomizer to extract same audio embeddings

    # set hparams from args
    sif.hparams.set_hparam('eval_ckpt', args.model_checkpoint)
    sif.hparams.set_hparam('img_height', args.image_height)
    sif.hparams.set_hparam('img_width', args.image_width)
    sif.hparams.set_hparam('num_channels', args.num_channels)

    videos_directory = Path(args.videos_directory)
    parent_output_directory = videos_directory.joinpath(args.output_directory)
    parent_output_directory.mkdir(exist_ok=True)

    # save the command used to run this inference script
    with parent_output_directory.joinpath('command.txt').open('w') as f:
        for arg in sys.argv:
            f.write(f'{arg} \\\n')

    video_durations, preprocessing_times, synthesising_times = [], [], []

    # preprocess videos
    output_directories, to_preprocess = [], []
    print('Preprocessing videos...')
    video_paths_to_process = list(videos_directory.glob(f'*.{args.video_format}'))
    if args.num_samples:
        video_paths_to_process = np.random.choice(video_paths_to_process, args.num_samples, replace=False)
    total_num_samples = len(video_paths_to_process)
    for video_path in video_paths_to_process:
        video_name = video_path.name.replace(f'.{args.video_format}', '')
        if args.included_paths and video_name not in args.included_paths:
            continue
        output_directory = parent_output_directory.joinpath(video_name)
        if output_directory.exists():
            if args.redo:
                shutil.rmtree(str(output_directory))
            else:
                output_directories.append(output_directory)
                continue
        output_directory.mkdir()

        to_preprocess.append([output_directory, video_path])

    if len(to_preprocess) > 0:
        num_processes = 5
        preprocess_tasks = []
        for i in range(num_processes):
            preprocess_tasks.append([i, args, to_preprocess[i::num_processes]])

        # run video pre-processing in parallel to speed things up
        with multiprocessing.Pool(processes=num_processes) as pool:
            results = pool.starmap(preprocess_wrapper, preprocess_tasks)
            for _output_directories, _video_durations, _preprocessing_times in results:
                output_directories.extend(_output_directories)
                video_durations.extend(_video_durations)
                preprocessing_times.extend(_preprocessing_times)

    num_preprocessed_samples = len(output_directories)
    num_failed_preprocessing_samples = total_num_samples - num_preprocessed_samples

    if num_failed_preprocessing_samples == total_num_samples:
        print('All videos failed preprocessing...')
        return

    # run synthesis
    synthesizer = sif.Synthesizer(verbose=False)
    print('Running synthesizer...')
    for output_directory in tqdm(output_directories):
        if output_directory.joinpath('generated_audio.wav').exists():
            continue
        start_time = time.time()
        speech_synthesis(
            synthesizer=synthesizer,
            video_directory=output_directory,
            normalise_frames=args.normalise_frames,
            combine_audio_and_video=args.combine_audio_and_video,
            clean_up=args.clean_up
        )
        synthesising_times.append(time.time() - start_time)

    if len(video_durations) > 0:
        print('Av. Video Duration (secs):', round(np.mean(video_durations), 1))
        print('Av. Preprocessing Time (secs):', round(np.mean(preprocessing_times), 1))
        print('Av. Synthesising Time (secs):', round(np.mean(synthesising_times), 1))

    stats = [
        f'Total No. Samples: {total_num_samples}',
        f'No. Preprocessed Samples: {num_preprocessed_samples}',
        f'No. Failed Preprocessed Samples: {num_failed_preprocessing_samples}',
    ]

    # don't generate STOI stats if using a different audio embedding from the video
    if not args.neutral_audio_path and not args.closest_audio_embedding and not args.neutral_audio_from_video_path:
        # generate STOI stats for all the videos
        print('Generating STOI stats...')
        av_stoi, av_estoi = 0, 0
        for output_directory in tqdm(output_directories):
            # calculated on samples that exist
            stoi, estoi = get_stois(output_directory)
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
    groundtruth_path = videos_directory.joinpath(args.groundtruth_name)
    run_asr = args.gcloud_credentials_path or args.deep_speech
    if groundtruth_path.exists() and run_asr:

        pred_wav_names = []
        if args.asr_test_groundtruth:
            pred_wav_names += ['audio.wav']
            if args.preprocess_audio:
                pred_wav_names += ['audio_preprocessed.wav']

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
                asr_included_paths=args.asr_included_paths,
                asr_excluded_paths=args.asr_excluded_paths
            )
            stats.extend(asr_results)

    for stat_line in stats:
        print(stat_line)

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


def analysis(args):
    # TODO: analyse inference output e.g.
    #  - which words performed the best/worst?
    #  - which phrases performed the best/worst? (WER per phrase)
    data = []
    with open(args.phrases_path, 'r') as f:
        for line in f.read().splitlines():
            phrase, key_words = re.match(r'(.+),(\[.+\])', line).groups()
            phrase = phrase.lower().replace('?', '').replace('!', '')
            key_words = ast.literal_eval(key_words)
            data.append([phrase, key_words])
    phrases_df = pd.DataFrame(data=data, columns=['Phrase', 'Key Words'])
    phrases = list(phrases_df['Phrase'].values)
    phrases_d = {phrase.replace(' ', '').replace('\'', ''): phrase for phrase in phrases}

    results = []
    for d in Path(args.test_directory).glob('*'):
        if not d.is_dir():
            continue
        gt = phrases_d[d.name.split('_')[0].lower()]
        asr_results_path = d.joinpath('Google_asr_results_generated_audio.txt')
        if not asr_results_path.exists():
            continue
        with asr_results_path.open('r') as f:
            predictions = f.read().splitlines()
        prediction = predictions[0]
        # results.append([expand_contractions(gt), expand_contractions(prediction)])
        results.append([gt, prediction])

    # which words performed the best/worst
    words = set()
    for phrase in phrases:
        for word in phrase.split(' '):
            words.add(word)
    words = list(words)
    word_counts = {word: {'correct': 0, 'count': 0} for word in words}
    for gt, prediction in results:
        gt_split, prediction_split = gt.split(' '), prediction.split(' ')
        if len(gt_split) != len(prediction_split):
            continue
        for i in range(len(gt_split)):
            word = gt_split[i]
            word_counts[word]['count'] += 1
            if gt_split[i] == prediction_split[i]:
                word_counts[word]['correct'] += 1
    word_counts = {word: d['correct'] / d['count'] for word, d in word_counts.items() if d['count'] > 0}
    word_counts = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)
    print('Top 10 Words by acc:', word_counts[:10])
    print('Bottom 10 Words:', word_counts[-10:], '\n')

    # which phrases performed best/worst
    phrase_wers = {phrase: {'wer': 0, 'count': 0} for phrase in phrases}
    for gt, prediction in results:
        phrase_wers[gt]['count'] += 1
        phrase_wers[gt]['wer'] += calculate_wer(gt, prediction)
    phrase_wers = {phrase: d['wer'] / d['count'] for phrase, d in phrase_wers.items() if d['count'] > 0}
    phrase_wers = sorted(phrase_wers.items(), key=lambda x: x[1])  # WER = lower is better
    print('Top 10 Phrases by WER:', phrase_wers[:10])
    print('Bottom 10 Phrases:', phrase_wers[-10:])


def main(args):
    f = {
        'inference': inference,
        'analysis': analysis,
    }
    f[args.run_type](args)


if __name__ == '__main__':
    def file_path_contents(s):
        all_contents = []
        for path in s.split(','):
            with open(path, 'r') as f:
                all_contents.extend(f.read().splitlines())

        return all_contents

    parser = argparse.ArgumentParser()
    sub_parsers = parser.add_subparsers(dest='run_type')
    
    parser_1 = sub_parsers.add_parser('inference')
    parser_1.add_argument('videos_directory')
    parser_1.add_argument('output_directory')
    parser_1.add_argument('model_checkpoint')
    parser_1.add_argument('--batch_size', type=int, default=32)
    parser_1.add_argument('--fps', type=int, default=25)
    parser_1.add_argument('--redo', action='store_true')
    parser_1.add_argument('--video_format', default='mp4')
    parser_1.add_argument('--normalise_frames', action='store_true')
    parser_1.add_argument('--blur_frames', action='store_true')
    parser_1.add_argument('--combine_audio_and_video', action='store_true')
    parser_1.add_argument('--neutral_audio_path')  # for extracting a neutral audio embedding
    parser_1.add_argument('--neutral_audio_from_video_path')  # for extracting a neutral audio embedding from video
    parser_1.add_argument('--clean_up', action='store_true')
    parser_1.add_argument('--audio_preprocessing', action='store_true')
    parser_1.add_argument('--closest_audio_embedding', action='store_true')
    parser_1.add_argument('--by_rate_of_speech', action='store_true')
    parser_1.add_argument('--by_content', action='store_true')
    parser_1.add_argument('--stats_only', action='store_true')
    parser_1.add_argument('--gcloud_credentials_path')
    parser_1.add_argument('--deep_speech', action='store_true')
    parser_1.add_argument('--asr_test_groundtruth', action='store_true')
    parser_1.add_argument('--phrases_path', help='Contains phrases to key word mapping to tally by keywords')
    parser_1.add_argument('--image_height', type=int, default=128, help='Used for mouth ROI')
    parser_1.add_argument('--image_width', type=int, default=128, help='Used for mouth ROI')
    parser_1.add_argument('--num_channels', type=int, default=3)
    parser_1.add_argument('--num_samples', type=int, help='Randomly select this number of samples to generate')
    parser_1.add_argument('--tally_by_keywords', action='store_true', help='Tally the rank accuracies by keywords')
    parser_1.add_argument('--lrw', action='store_true')
    parser_1.add_argument('--asr_language_code', default='en-GB')
    parser_1.add_argument('--asr_model', default='command_and_search')
    parser_1.add_argument('--asr_sample_rate', type=int, default=16000)
    parser_1.add_argument('--calculate_hit_rate', action='store_true')
    parser_1.add_argument('--included_paths', type=file_path_contents)
    parser_1.add_argument('--asr_included_paths', type=file_path_contents)
    parser_1.add_argument('--asr_excluded_paths', type=file_path_contents)
    parser_1.add_argument('--groundtruth_name', default='groundtruth.csv')
    parser_1.add_argument('--use_old_ffmpeg', action='store_true')
    parser_1.add_argument('--use_old_mouth_extractor', action='store_true')
    parser_1.add_argument('--use_perspective_warp', action='store_true')
    parser_1.add_argument('--use_cfe_cropper', action='store_true')
    parser_1.add_argument('--greyscale', action='store_true')
    parser_1.add_argument('--smooth_frames', action='store_true')
    parser_1.add_argument('--cfe_host')
    parser_1.add_argument('--cpu', action='store_true')
    parser_1.add_argument('--debug', action='store_true')

    parser_2 = sub_parsers.add_parser('analysis')
    parser_2.add_argument('test_directory')
    parser_2.add_argument('phrases_path')

    main(parser.parse_args())
