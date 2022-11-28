"""the SRAVI training dataset is imbalanced in terms of word counts"""
import argparse
import collections
import json
import os
import pickle
import random
import re
import shutil
import sys
import tempfile
import uuid
from pathlib import Path

import matplotlib.pyplot as plt
import redis
import requests
from tqdm import tqdm

from audio_utils import forced_alignment, get_audio_duration, pad_audio
from video_utils import extract_audio, crop, convert_fps, get_fps


def save_file(response, output_path):
    with open(output_path, 'wb') as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)


def voice_clone(text, audio_path, host, port, output_path):
    with open(audio_path, 'rb') as f:
        response = requests.post(f'http://{host}:{port}/tts',
                                 files={'audio': f},
                                 data={'text': text})
        response.raise_for_status()

    save_file(response, output_path)


def wav2lip(audio_path, video_path, host, port, output_path):
    with open(audio_path, 'rb') as f1, open(video_path, 'rb') as f2:
        response = requests.post(f'http://{host}:{port}/generate',
                                 files={'audio_file': f1, 'video_file': f2})
        response.raise_for_status()

    save_file(response, output_path)


def get_word_cropping_region(audio_path, transcript, word, host, port, delay=None, debug=False):
    # run forced alignment - word should appear in middle
    alignment = forced_alignment(audio_path=audio_path, transcript=transcript, host=host, port=port)
    if alignment is None:
        return
    if debug:
        print(alignment)

    try:
        word_index = [x[0].lower() for x in alignment].index(word)
    except ValueError:
        return
    start_time, end_time = alignment[word_index][1:3]

    if debug:
        print('FA Before', start_time, end_time)

    if delay:
        start_time += delay
        end_time += delay

    mid_time = start_time + ((end_time - start_time) / 2)
    start_time, end_time = mid_time - 0.5, mid_time + 0.5

    if debug:
        print('FA After', start_time, end_time)

    return start_time, end_time


def generate_dataset(args):
    """use real and synthetic data from Wav2Lip to fill in gaps"""
    redis_server = redis.Redis(host=args.redis_host, port=args.redis_port)

    # get video paths from redis server
    video_paths = [redis_server.lindex(args.pull_list_name, i).decode('utf-8')
                   for i in range(redis_server.llen(args.pull_list_name))]
    if not video_paths:
        print('No video paths to process')
        exit()

    # get all phrases
    with open(args.phrases_path, 'r') as f:
        phrases = f.read().splitlines()
    print('Num phrases:', len(phrases))
    phrases = [p.lower() for p in phrases]
    phrases_d = {phrase.replace('\'', '').replace(' ', ''): phrase for phrase in phrases}

    # get unique words
    word_to_phrase_d = {}
    for phrase in phrases:
        for word in phrase.split(' '):
            phrases_containing_word = word_to_phrase_d.get(word, [])
            phrases_containing_word.append(phrase)
            word_to_phrase_d[word] = phrases_containing_word
    unique_words = list(word_to_phrase_d.keys())
    random.shuffle(unique_words)
    print('Num unique words:', len(unique_words))

    # build dict of current speaker word samples
    # something like:
    # {domhnall.boyle@liopa.co.uk: {'bathroom': [['i need the bathroom', <video_path>]]}}
    all_speaker_samples = {}
    for video_path in video_paths:
        video_path = Path(video_path)
        speaker_id, video_name = video_path.parents[0].name, video_path.name
        phrase = video_name.split('_')[0].lower()
        phrase = phrases_d[phrase]

        speaker_word_samples = all_speaker_samples.get(speaker_id, {})
        for word in phrase.split(' '):
            word_samples = speaker_word_samples.get(word, [])
            word_samples.append([phrase, str(video_path)])  # need the phrase for forced alignment later
            speaker_word_samples[word] = word_samples
        all_speaker_samples[speaker_id] = speaker_word_samples

    speaker_ids = list(all_speaker_samples.keys())

    print('Num training speakers:', len(all_speaker_samples))
    print('Approx. Total samples:', len(unique_words) * args.samples_per_word * len(all_speaker_samples))

    output_directory = Path(args.output_directory)
    if args.redo and output_directory.exists():
        shutil.rmtree(str(output_directory))
    output_directory.mkdir(exist_ok=True)

    cloned_audio_path = '/tmp/cloned_audio.wav'
    converted_fps_path = '/tmp/convert_fps.mp4'

    for speaker_id in speaker_ids:
        speaker_words = list(all_speaker_samples[speaker_id].keys())
        print(speaker_id)
        for word in tqdm(unique_words):
            if args.debug:
                print(word)

            is_completed_word = len(list(output_directory.glob(f'{speaker_id}_{word}_*.mp4'))) > 0
            if is_completed_word:
                continue

            word_samples = all_speaker_samples[speaker_id].get(word, [])
            random.shuffle(word_samples)
            num_to_synthesise = max(args.samples_per_word - len(word_samples), 0)  # clip to 0
            num_to_copy = args.samples_per_word - num_to_synthesise

            if args.debug:
                print(f'Current {len(word_samples)}, Num To Synthesise {num_to_synthesise}')

            i = 0
            num_synthesise_attempts = 3

            # first create synthetic data if required
            while num_to_synthesise > 0:
                num_synthesise_attempts -= 1
                if num_synthesise_attempts == 0:
                    break

                phrase_to_synthesise = random.choice(word_to_phrase_d[word])  # random phrase that contains word
                assert word in phrase_to_synthesise

                # TODO: Rate of speech impacts the voice cloning
                # get random reference video - shouldn't be similar to the phrase to synthesise
                while True:
                    random_word = random.choice(list(set(speaker_words) - {word}))
                    if random_word in phrase_to_synthesise:
                        continue
                    break
                ref_video_path = random.choice(all_speaker_samples[speaker_id][random_word])[1]

                # convert fps if applicable
                fps = get_fps(ref_video_path)
                if fps != args.fps:
                    ref_video_path = convert_fps(video_path=ref_video_path, new_video_path=converted_fps_path,
                                                 fps=args.fps)

                # extract the audio
                ref_audio_file = extract_audio(ref_video_path, use_old_ffmpeg=args.use_old_ffmpeg)

                # run voice cloning using the phrase and reference audio
                voice_clone(
                    text=phrase_to_synthesise.upper(),
                    audio_path=ref_audio_file.name,
                    host=args.tts_host,
                    port=args.tts_port,
                    output_path=cloned_audio_path
                )
                if args.debug:
                    shutil.copyfile(cloned_audio_path, str(output_directory.joinpath('debug_cloned_audio.wav')))
                ref_audio_file.close()

                # run forced alignment - better on tightly cropped synthetic audio
                region = get_word_cropping_region(
                    audio_path=cloned_audio_path,
                    transcript=phrase_to_synthesise,
                    word=word,
                    host=args.fa_host,
                    port=args.fa_port,
                    delay=args.delay,
                    debug=args.debug
                )
                if region is None:
                    continue
                start_time, end_time = region

                # pad the synthetic audio with silence at the start and end
                # this is used for later when we need to crop out words at the end of the video
                # no silences means the word is not cropped out correctly
                # need to ensure 1 second of 25 FPS
                padded_audio_file = pad_audio(cloned_audio_path, delay=args.delay, end=True)
                if args.debug:
                    shutil.copyfile(padded_audio_file.name, str(output_directory.joinpath('debug_padded_audio.wav')))

                # generate the lip movements from the padded cloned audio
                synthetic_video_path = f'/tmp/{uuid.uuid4()}.mp4'
                try:
                    wav2lip(
                        audio_path=padded_audio_file.name,
                        video_path=ref_video_path,
                        host=args.wav2lip_host,
                        port=args.wav2lip_port,
                        output_path=synthetic_video_path
                    )
                except Exception as e:
                    print(e)
                    padded_audio_file.close()
                    continue
                if args.debug:
                    shutil.copyfile(synthetic_video_path, str(output_directory.joinpath('debug_synthetic.mp4')))
                padded_audio_file.close()

                # crop out the word of interest
                cropped_video_path = crop(video_path=synthetic_video_path, start=start_time, end=end_time)
                output_video_path = output_directory.joinpath(f'{speaker_id}_{word}_{i+1}_{True}.mp4')
                try:
                    # if the cropping fails
                    shutil.copyfile(cropped_video_path, output_video_path)
                except FileNotFoundError:
                    continue

                for p in [synthetic_video_path, cropped_video_path, cloned_audio_path, converted_fps_path]:
                    if os.path.exists(p):
                        os.remove(p)

                num_to_synthesise -= 1
                i += 1

            if num_synthesise_attempts == 0:
                continue

            # second crop out real samples
            for phrase, video_path in word_samples[:num_to_copy]:

                # convert fps if applicable
                fps = get_fps(video_path)
                if fps != args.fps:
                    video_path = convert_fps(video_path=video_path, new_video_path=converted_fps_path, fps=args.fps)
                if args.debug:
                    print('Converted FPS')

                # extract the audio
                audio_file = extract_audio(video_path, use_old_ffmpeg=args.use_old_ffmpeg)
                if args.debug:
                    print('Extracted audio')

                # run forced alignment
                region = get_word_cropping_region(
                    audio_path=audio_file.name,
                    transcript=phrase,
                    word=word,
                    host=args.fa_host,
                    port=args.fa_port,
                    debug=args.debug
                )
                audio_file.close()
                if region is None:
                    continue
                start_time, end_time = region

                # crop out the word of interest
                cropped_video_path = crop(video_path=video_path, start=start_time, end=end_time)
                output_video_path = output_directory.joinpath(f'{speaker_id}_{word}_{i+1}_{False}.mp4')
                try:
                    # if the cropping fails
                    shutil.copyfile(cropped_video_path, output_video_path)
                except FileNotFoundError:
                    continue

                # clean up
                for p in [converted_fps_path, cropped_video_path]:
                    if os.path.exists(p):
                        os.remove(p)

                i += 1


def analyse_dataset(args):
    dataset_paths = list(Path(args.output_directory).glob('*.mp4'))
    print('Num Video Paths:', len(dataset_paths))

    total_num_samples = args.samples_per_word * args.num_words * args.num_speakers
    print(f'Progress: {round((len(dataset_paths) / total_num_samples) * 100, 1)}%')

    speaker_d = {}  # {speaker: {word: [True, False...]}}
    word_count_d = {}  # {word: 0,...}
    real_word_count_d = {}
    num_real, num_synthetic = 0, 0
    for video_path in dataset_paths:
        name = video_path.name.replace('.mp4', '')
        speaker, word, _, is_synthetic = re.match(r'(.+)_(.+)_(.+)_(.+)', name).groups()
        is_synthetic = True if is_synthetic.strip() == 'True' else False

        if is_synthetic:
            num_synthetic += 1
        else:
            num_real += 1

        speaker_samples = speaker_d.get(speaker, {})
        word_samples = speaker_samples.get(word, [])
        word_samples.append(is_synthetic)
        speaker_samples[word] = word_samples
        speaker_d[speaker] = speaker_samples

        word_count_d[word] = word_count_d.get(word, 0) + 1

        if not is_synthetic:
            real_word_count_d[word] = real_word_count_d.get(word, 0) + 1

    # show % real/synthetic data
    total = num_real + num_synthetic
    sizes = [(num_real / total) * 100, (num_synthetic / total) * 100]
    labels = ['Real', 'Synthetic']
    for label, _size in zip(labels, sizes):
        number = (_size / 100) * total
        print(f'{label}: {round(number)} ({round(_size, 1)}%)')
    print()

    # show completed words per user
    # completed words = 100% of samples
    for speaker, word_samples in speaker_d.items():
        num_completed = 0
        num_real, num_synthetic = 0, 0
        for word, samples in word_samples.items():
            if len(samples) == args.samples_per_word:
                num_completed += 1
            num_synthetic += sum(samples)  # True = is synthetic
            num_real += (len(samples) - sum(samples))

        percent_completed = round((num_completed / args.num_words) * 100, 1)
        real_synthetic_ratio = round(num_real / num_synthetic, 1)  # > 1 means more real than synthetic data

        if percent_completed > 90 and (real_synthetic_ratio >= 0.5):
            print(f'{speaker} % words completed: {percent_completed}%, ratio: {real_synthetic_ratio}')

    print('\nWord Counts:', word_count_d)
    print('Min num samples per word:', min(word_count_d.items(), key=lambda x: x[1]))
    print('Min num real samples per word:', min(real_word_count_d.items(), key=lambda x: x[1]))


def ngram(args):
    """find the most popular bigrams (contexts) in the librispeech text based on the SRAVI words
    use the most uniquely phonetic contexts of these words
    """
    with open('sravi_training_counts.pkl', 'rb') as f:
        sravi_words, _ = zip(*pickle.load(f))

    def ngrams(text, n=2):
        return zip(*[text[i:] for i in range(n)])

    # find most common n-grams (contexts) with SRAVI words in the middle of them
    librispeech_ngram_path = Path('librispeech_ngram.pkl')
    if librispeech_ngram_path.exists():
        with librispeech_ngram_path.open('rb') as f:
            d = pickle.load(f)
    else:
        d = {}  # {"what": {("know", "what", "is"): 850, ...}}
        for text_path in tqdm(list(Path(args.corpus_directory).glob(args.glob_path))):
            with text_path.open('r', encoding='latin-1') as f:
                text = f.read().lower().replace('\n', '')
            text_split = text.split(' ')
            # bigrams = ngrams(text_split, 2)
            trigrams = ngrams(text_split, 3)
            trigram_counts = collections.Counter(trigrams)  # group all possible trigrams by counts
            # ngram_counts.update(trigrams)
            for k, v in trigram_counts.items():
                for word in sravi_words:
                    # # if sravi word in middle of trigram or either end of bigram
                    # if (len(k) == 2 and word in k) or (len(k) == 3 and word == k[1]):
                    #     common = d.get(word, {})
                    #     common[k] = common.get(k, 0) + v  # update the count
                    #     d[word] = common
                    if word == k[1]:
                        for h in [k, k[:2], k[1:]]:
                            # update the count for the trigram and 2 bigrams either side
                            common = d.get(word, {})
                            common[h] = common.get(h, 0) + v
                            d[word] = common

        with librispeech_ngram_path.open('wb') as f:
            pickle.dump(d, f)

    # # show top 10 ngrams per SRAVI word
    # trigram_text = ''
    # for word in sravi_words:
    #     try:
    #         top_10 = sorted(d[word].items(), key=lambda x: x[1], reverse=True)[:10]
    #     except KeyError:  # word not in librispeech text e.g. SRAVI
    #         continue
    #     s = f'{word}: '
    #     for result in top_10:
    #         s += '"' + ' '.join(list(result[0])) + '"'
    #         s += f' ({result[1]}), '
    #     trigram_text += f'{s}\n'
    # with open('librispeech_ngram.txt', 'w') as f:
    #     f.write(trigram_text)

    viseme_to_phoneme = {  # lee and york 2002
        'p': ['p', 'b', 'm', 'em'],
        'f': ['f', 'v'],
        't': ['t', 'd', 's', 'z', 'th', 'dh', 'dx'],
        'w': ['w', 'wh', 'r'],
        'ch': ['ch', 'jh', 'sh', 'zh'],
        'ey': ['eh', 'ey', 'ae', 'aw'],
        'k': ['k', 'g', 'n', 'l', 'nx', 'hh', 'y', 'el', 'en', 'ng'],
        'iy': ['iy', 'ih'],
        'aa': ['aa'],
        'ah': ['ah', 'ax', 'ay'],
        'er': ['er'],
        'ao': ['ao', 'oy', 'ix', 'ow'],
        'uh': ['uh', 'uw'],
        'sp': ['sil', 'sp']
    }
    phoneme_to_viseme = {
        phoneme: viseme
        for viseme, phonemes in viseme_to_phoneme.items()
        for phoneme in phonemes
    }
    words_to_visemes = {}
    with open('/shared/Repos/visual-dtw/cmudict-en-us.dict', 'r') as f:
        for line in f.readlines():
            line = line.split(' ')
            word, phonemes = line[0], line[1:]
            phonemes = list(map(lambda phone: phone.lower().strip(), phonemes))
            visemes = list(map(lambda phone: phoneme_to_viseme[phone], phonemes))
            words_to_visemes[word] = visemes
    visemes = list(set(viseme_to_phoneme.keys()))
    print('Visemes:', visemes)

    def get_phonemes(word):
        # this is actually visemes rather than phonemes
        return words_to_visemes[word]

    def get_unique_phonetic_contexts(contexts, direction, n=8):
        _contexts = []
        unique_phonemes = []
        for bigram, count in contexts:
            # extract phoneme of interest
            # if left context, phoneme of interest is last in left word
            # if right context, phoneme of interest is first in right word
            try:
                if direction == 'left':
                    phoneme = get_phonemes(bigram[0])[-1]
                else:
                    phoneme = get_phonemes(bigram[1])[0]
            except KeyError:
                continue

            if phoneme in unique_phonemes:  # already recorded this phoneme
                continue

            if len(unique_phonemes) == n:
                break

            _contexts.append((bigram, count))
            unique_phonemes.append(phoneme)  # update recorded phonemes

        # print(_contexts, unique_phonemes, '\n')

        return _contexts

    # show top 10 bigrams per SRAVI word
    text = ''
    for word in tqdm(sravi_words):
        if word not in d:
            continue
        bigrams = sorted([(k, v) for k, v in d[word].items() if len(k) == 2], key=lambda x: x[1], reverse=True)
        left_contexts, right_contexts = [], []
        for bigram, count in bigrams:
            bigram = ' '.join(list(bigram)).strip()  # remove any whitespace
            bigram = re.sub('[^a-z\' ]', '', bigram)  # sub all non-alphabetic chars with blanks
            bigram = bigram.split(' ')
            if len(bigram) != 2:  # ensure bigrams
                continue
            if bigram[0] == word:  # split into left and right contexts
                right_contexts.append((bigram, count))
            elif bigram[1] == word:
                left_contexts.append((bigram, count))

        # get top most phonetically unique contexts
        left_contexts = get_unique_phonetic_contexts(left_contexts, 'left')
        right_contexts = get_unique_phonetic_contexts(right_contexts, 'right')

        text += f'{word}: '
        text += ' '.join([f"\"{' '.join(x[0])}\" ({x[1]})," for x in left_contexts])
        text += ' '.join([f"\"{' '.join(x[0])}\" ({x[1]})," for x in right_contexts])[:-1] + '\n'

    with open('librispeech_ngram.txt', 'w') as f:
        f.write(text)


def generate_dataset_2(args):
    """just crop out every SRAVI word and save to disk for the preprocessor"""
    redis_server = redis.Redis(host=args.redis_host, port=args.redis_port)

    # get video paths from redis server
    video_paths = [redis_server.lindex(args.pull_list_name, i).decode('utf-8')
                   for i in range(redis_server.llen(args.pull_list_name))]
    if not video_paths:
        print('No video paths to process')
        exit()

    # get all phrases
    with open(args.phrases_path, 'r') as f:
        phrases = f.read().splitlines()
    print('Num phrases:', len(phrases))
    phrases = [p.lower() for p in phrases]
    phrases_d = {phrase.replace('\'', '').replace(' ', ''): phrase for phrase in phrases}

    # get all video samples and phrases - needs for forced alignment later
    all_samples = []
    for video_path in video_paths:
        video_path = Path(video_path)
        speaker_id, video_name = video_path.parents[0].name, video_path.name
        phrase = video_name.split('_')[0].lower()
        phrase = phrases_d[phrase]
        all_samples.append([video_path, phrase])

    output_directory = Path(args.output_directory)
    if args.redo and output_directory.exists():
        shutil.rmtree(str(output_directory))
    output_directory.mkdir()

    # run forced alignment and crop out words
    for video_path, phrase in tqdm(all_samples):
        # don't preprocess the audio in case FA not used to input
        audio_file = extract_audio(video_path=str(video_path))
        audio_length = get_audio_duration(audio_path=audio_file.name)

        # run forced alignment
        alignment = forced_alignment(audio_path=audio_file.name, transcript=phrase, host=args.fa_host,
                                     port=args.fa_port)
        if alignment is None or len(alignment) == 0:
            continue
        if args.debug:
            print(alignment, audio_length)

        crop_criteria = []
        for word in phrase.split(' '):
            try:
                word_index = [x[0].lower() for x in alignment].index(word)
            except ValueError:
                print('Word not in alignment:', word, alignment)
                continue
            start_time, end_time = alignment[word_index][1:3]

            if args.debug:
                print('FA Before', start_time, end_time)

            mid_time = start_time + ((end_time - start_time) / 2)
            start_time, end_time = mid_time - 0.5, mid_time + 0.5

            if args.debug:
                print('FA After', start_time, end_time)

            crop_criteria.append([word, start_time, end_time])

        # we want to crop silences (sil) at the start and end too
        crop_criteria.extend([['sil', 0.0001, 1.0001],  # first 1 second of sil
                              ['sil', audio_length - 1, audio_length]])  # last 1 second of sil
        speaking_start_time = alignment[0][1]
        speaking_end_time = alignment[-1][2]
        if speaking_start_time >= 1:
            crop_criteria.append(['sil', speaking_start_time - 1, speaking_start_time])  # sil before first spoken word
        if audio_length - speaking_end_time >= 1:
            crop_criteria.append(['sil', speaking_end_time, speaking_end_time + 1])  # sil from last spoken word

        if args.debug:
            print(crop_criteria)

        speaker_id = video_path.parents[0].name
        phrase = video_path.stem.split('_')[0]
        for word, start_time, end_time in crop_criteria:
            # crop using the start and end times
            cropped_video_path = crop(
                video_path=str(video_path),
                start=start_time,
                end=end_time
            )

            parent_directory = output_directory.joinpath(speaker_id)
            parent_directory.mkdir(exist_ok=True)

            output_video_path = parent_directory.joinpath(f'{phrase}_{word}_{str(uuid.uuid4())}.mp4')
            try:
                # if the cropping fails
                shutil.copyfile(cropped_video_path, output_video_path)
            except FileNotFoundError:
                continue

            if os.path.exists(cropped_video_path):
                os.remove(cropped_video_path)

        audio_file.close()


def generate_data_capture_scripts(args):
    words_of_interest = """
        uncomfortable
        comfortable
        depressed
        yesterday
        treatment
        important
        happening
        headlines
        bathroom
        disagree
        relative
        happened
        anxious
        thirsty
        suction
        feeling
        blanket
        painful
        explain
        worried
        quietly
        please
        bright
        toilet
        hungry
        doctor
        family
        scared
        turned
        throat
        lonely
        better
        drink
        noisy
        cough
        thank
        sleep
        tired
        hurts
        happy
        speak
        moved
        cold
        time
        pain
        help
        turn
        good
        sore
        lips
        okay
        no
        ok
    """.split('\n')
    words_of_interest = [w.strip() for w in words_of_interest if w.strip() != '']

    words_d = {}
    with open(args.ngram_path, 'r') as f:
        for line in f.read().splitlines():
            word, contexts = line.split(':')
            word = word.strip()
            if '#' in word or word not in words_of_interest:
                continue
            contexts = contexts.strip()
            contexts = [re.match(r'"(.+)" \(\d+\)', context.strip()).groups()[0] 
                        for context in contexts.split(',')]
            if len(contexts) != 16:
                print(word, len(contexts))
                exit()
            words_d[word] = contexts
   
    # {"items": ["one zero", "one one", "one two", "one three", "one four", "one five", "one six", "one seven",
    # "one eight", "one nine", "two zero"], "cameraId": "FRONT"}
    scripts = [{'items': [], 'cameraId': 'FRONT'} for _ in range(2)]
    for word, contexts in words_d.items():
        left_contexts, right_contexts = contexts[:8], contexts[8:16]
        # print(word, len(left_contexts), len(right_contexts))

        # phrase_1 = f"{', '.join(left_contexts[:4])}, {', '.join(right_contexts[4:])}"
        # phrase_2 = f"{', '.join(left_contexts[4:])}, {', '.join(right_contexts[:4])}"
        # # print(word, phrase_1, phrase_2)

        sentences_1 = [' '.join(left_contexts[:4]), ' '.join(right_contexts[4:])]
        sentences_2 = [' '.join(left_contexts[4:]), ' '.join(right_contexts[:4])]

        scripts[0]['items'].extend(sentences_1)
        scripts[1]['items'].extend(sentences_2)

    # 8 instances per word = 1 phrase
    # training scripts
    for script in scripts:
        print(len(script['items']))
        print(json.dumps(script))

    # # we need a training script
    # # ~100 sentences, 2 per word
    # # 4 word utterances per sentence = 8 per word
    # left_context_sentences, right_context_sentences = [], []
    # for word, contexts in words_d.items():
    #     left_contexts, right_contexts = contexts[:8], contexts[8:16]
    #
    #     left_context_sentence = ' '.join(left_contexts[:4])
    #     right_context_sentence = ' '.join(right_contexts[:4])
    #
    #     left_context_sentences.append(left_context_sentence)
    #     right_context_sentences.append(right_context_sentence)
    # script = {'items': left_context_sentences + right_context_sentences, 'cameraId': 'FRONT'}
    # print(script)

    # # we also need a test script
    # # 2 sentences per word for testing from librispeech
    # # NOT in the SRAVI context
    # word_sentences_d = {word: [] for word in words_of_interest}
    # for text_path in tqdm(list(Path(args.corpus_directory).glob(args.glob_path))):
    #     with text_path.open('r', encoding='ISO-8859-1') as f:
    #         text = f.read().lower().replace('\n', '')
    #     for sentence in text.split('.'):
    #         # sentence = sentence.strip().replace('.', '')
    #         # sentence = ' '.join([w.strip() for w in sentence.split(' ')])
    #         sentence_words = re.findall(r'(\w+)', sentence)
    #         sentence = ' '.join(sentence_words)
    #         if len(sentence_words) > 10:
    #             continue
    #         for word in words_of_interest:
    #             if word in sentence_words and len(word_sentences_d[word]) < 2:
    #                 word_sentences_d[word].append(sentence)
    #                 break
    #     if all([len(sentences) == 2 for sentences in word_sentences_d.values()]):
    #         break
    # script = {'items': [], 'cameraId': 'FRONT'}
    # for sentences in word_sentences_d.values():
    #     script['items'].extend(sentences)
    # print(json.dumps(script))


def main(args):
    f = {
        'generate_dataset': generate_dataset,
        'analyse_dataset': analyse_dataset,
        'ngram': ngram,
        'generate_dataset_2': generate_dataset_2,
        'generate_data_capture_scripts': generate_data_capture_scripts
    }
    f[args.run_type](args)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    sub_parsers = parser.add_subparsers(dest='run_type')

    parser_1 = sub_parsers.add_parser('generate_dataset')
    parser_1.add_argument('output_directory')
    parser_1.add_argument('pull_list_name')
    parser_1.add_argument('phrases_path')
    parser_1.add_argument('samples_per_word', type=int)
    parser_1.add_argument('--redis_host', default='redis')
    parser_1.add_argument('--redis_port', type=int, default=6379)
    parser_1.add_argument('--tts_host', default='0.0.0.0')
    parser_1.add_argument('--tts_port', type=int, default=8000)
    parser_1.add_argument('--wav2lip_host', default='wav2lip')
    parser_1.add_argument('--wav2lip_port', type=int, default=8000)
    parser_1.add_argument('--fa_host', default='forced-alignment')
    parser_1.add_argument('--fa_port', type=int, default=8082)
    parser_1.add_argument('--use_old_ffmpeg', action='store_true')
    parser_1.add_argument('--fps', type=int, default=25)
    parser_1.add_argument('--delay', type=int, default=2)
    parser_1.add_argument('--debug', action='store_true')
    parser_1.add_argument('--redo', action='store_true')

    parser_2 = sub_parsers.add_parser('analyse_dataset')
    parser_2.add_argument('output_directory')
    parser_2.add_argument('samples_per_word', type=int)
    parser_2.add_argument('num_words', type=int)
    parser_2.add_argument('num_speakers', type=int)

    parser_3 = sub_parsers.add_parser('ngram')
    parser_3.add_argument('corpus_directory')
    parser_3.add_argument('glob_path')

    parser_4 = sub_parsers.add_parser('generate_dataset_2')
    parser_4.add_argument('output_directory')
    parser_4.add_argument('pull_list_name')
    parser_4.add_argument('phrases_path')
    parser_4.add_argument('--redis_host', default='redis')
    parser_4.add_argument('--redis_port', type=int, default=6379)
    parser_4.add_argument('--fa_host', default='forced-alignment')
    parser_4.add_argument('--fa_port', type=int, default=8082)
    parser_4.add_argument('--debug', action='store_true')
    parser_4.add_argument('--redo', action='store_true')

    parser_5 = sub_parsers.add_parser('generate_data_capture_scripts')
    parser_5.add_argument('ngram_path')   
    parser_5.add_argument('corpus_directory')
    parser_5.add_argument('glob_path') 

    main(parser.parse_args())
