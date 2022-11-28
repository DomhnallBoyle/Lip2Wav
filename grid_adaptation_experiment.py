"""how much data is required to adapt to a new vocab?

Use the GRID dataset, leave 2 speakers for testing
Each time, increase the number samples per word in the training data
E.g. iteration 1, 54 unique GRID words * 10 samples per word = 540 training samples
"""
import argparse
import collections
import itertools
import math
import random
import re
import shutil
import tempfile
import time
import uuid
from http import HTTPStatus
from pathlib import Path
from six import iteritems, itervalues, text_type

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
from jiwer import wer
from tqdm import tqdm

from sample_pool import SamplePool
from video_utils import convert_fps, extract_audio, crop, get_fps, get_num_frames, get_video_duration

NUM_SPEAKERS = 33
NUM_WORDS = 51
DIGITS = ['zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine']
LETTERS = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't',
           'u', 'v', 'w', 'x', 'y', 'z']
COLOURS = ['blue', 'green', 'red', 'white']
ACTIONS = ['bin', 'lay', 'place', 'set']
PREPS = ['at', 'by', 'in', 'with']
ADVERBS = ['again', 'now', 'please', 'soon']


def generate_dataset(args):
    """Use forced alignment to crop out the words and create training samples"""

    dataset_path = Path(args.dataset_path)
    output_directory_path = Path(args.output_directory)
    if not args.stats_only:
        if output_directory_path.exists():
            shutil.rmtree(output_directory_path)
        output_directory_path.mkdir()

    num_speakers = args.num_speakers if args.num_speakers else NUM_SPEAKERS
    num_words = NUM_WORDS

    # get all GRID samples
    dataset_paths = []
    for video_path in dataset_path.glob(args.video_glob):
        speaker_id = video_path.parents[0].name
        video_name = video_path.stem.split('_')[0]

        align_path = dataset_path.joinpath('../align').joinpath(f'{video_name}.align')
        if align_path.exists():
            dataset_paths.append([speaker_id, str(video_path), str(align_path)])

    print('Num Video Paths:', len(dataset_paths))
    if not args.skip_assert:
        total_video_paths = num_speakers * 1000
        assert len(dataset_paths) == total_video_paths, f'{len(dataset_paths)} != {total_video_paths}'

    # create dict of {speaker_id: {word: [[video_path, transcript], [video_path, transcript]]} ... }
    # required for forced alignment
    dataset = {}
    for speaker_id, video_path, align_path in dataset_paths:
        with open(align_path, 'r') as f:
            words = []
            for line in f.read().splitlines():
                word = line.split(' ')[-1]
                if word in ['sil', 'sp']:
                    continue
                words.append(word)
        transcript = ' '.join(words).lower()
        speaker_dataset = dataset.get(speaker_id, {})
        for word in words:
            word_samples = speaker_dataset.get(word, [])
            word_samples.append([video_path, transcript])
            speaker_dataset[word] = word_samples
        dataset[speaker_id] = speaker_dataset
    max_samples, min_samples = 0, np.inf
    for speaker_id, speaker_dataset in dataset.items():
        print(f'\nSpeaker {speaker_id} word counts:', {k: len(v) for k, v in speaker_dataset.items()})
        print(f'Speaker {speaker_id} total word count:', sum([len(v) for v in speaker_dataset.values()]))
        assert len(speaker_dataset) == 51  # num unique words
        for samples in speaker_dataset.values():
            num_samples = len(samples)
            if num_samples > max_samples:
                max_samples = num_samples
            if num_samples < min_samples:
                min_samples = num_samples
    del dataset_paths

    print('Num Speakers:', num_speakers)
    print('Num Words:', num_words)
    print('Selected # Samples Per Word Per Speaker:', args.num_samples_per_word_per_person)
    print('Max Samples Per Word Per Speaker:', max_samples)
    print('Min Samples Per Word Per Speaker:', min_samples)

    if args.num_samples_per_word_per_person > max_samples:
        print(f'Only allowed <= {max_samples} samples')
        exit()

    if args.stats_only:
        exit()

    # e.g. num_samples_per_word_per_person = 20
    # num_people * num_words * 20 = 31 * 51 * 20 = 31610 word instances / training samples

    # grab the samples per word per person
    samples_this_training_round = []
    for speaker_id, speaker_dataset in dataset.items():
        for word, samples in speaker_dataset.items():
            random.shuffle(samples)
            random_word_samples = samples[:args.num_samples_per_word_per_person]
            # if there are only 40 samples, running samples[:100] will still select the 40 samples
            assert len(random_word_samples) <= args.num_samples_per_word_per_person
            random_word_samples = [[word, *sample] for sample in random_word_samples]
            samples_this_training_round.extend(random_word_samples)
    del dataset

    upper_bound = num_speakers * num_words * args.num_samples_per_word_per_person
    assert len(samples_this_training_round) <= upper_bound
    print('Num Samples This Training Round:', len(samples_this_training_round), upper_bound)

    # GRID is 25 FPS
    # each sample for Lip2Wav requires 25 frames
    # which means we need to extract 1 second clips after forced alignment
    # use padding either side of the word to do this
    for word, video_path, transcript in tqdm(samples_this_training_round):
        audio_file = extract_audio(video_path=video_path)
        text_file = tempfile.NamedTemporaryFile(suffix='.txt')
        with open(text_file.name, 'w') as f:
            f.write(transcript)
        text_file.seek(0)
        with open(audio_file.name, 'rb') as f1, open(text_file.name, 'rb') as f2:
            response = requests.post(f'http://{args.forced_alignment_host}:8082/align',
                                     files={'audio': f1.read(), 'transcript': f2.read()})
        if response.status_code != HTTPStatus.OK:
            print(response.message, response.status_code)
            continue
        alignment = response.json()['alignment']

        audio_file.close()
        text_file.close()

        # get word alignment
        word_index = [x[0].lower() for x in alignment].index(word)
        start_time, end_time = alignment[word_index][1:3]
        
        # tightly crop = ignoring start and end silences
        if args.tightly_crop and word_index == 0:
            end_time = start_time + 1
        elif args.tightly_crop and word_index == len(alignment) - 1:
            start_time = end_time - 1
        else:
            mid_time = start_time + ((end_time - start_time) / 2)
            start_time, end_time = mid_time - 0.5, mid_time + 0.5
        print(video_path, word, start_time, end_time, end_time - start_time)

        cropped_video_path = crop(video_path=video_path, start=start_time, end=end_time)
        output_video_path = output_directory_path.joinpath(f'{word}_{uuid.uuid4()}.mpg')

        # output file to directory to be added by the feeder
        shutil.copyfile(cropped_video_path, output_video_path)


def analyse_google_asr(args):
    test_dir = Path(args.test_results_path)
    gt_df = pd.read_csv(args.groundtruth_path, names=['Video Name', 'Phrase'])

    word_confusion_d = {}
    for index, row in gt_df.iterrows():
        video_name, gt = row['Video Name'], row['Phrase'].lower()
        d = test_dir.joinpath(video_name)
        if not d.exists():
            continue
        predictions_path = d.joinpath('Google_asr_results_generated_audio.txt')
        if not predictions_path.exists():
            continue
        with predictions_path.open('r') as f:
            predictions = f.read().splitlines()
        if not predictions:
            continue
        prediction = predictions[0]
        gt_split, pred_split = gt.split(' '), prediction.split(' ')
        if len(pred_split) != 6:
            continue  # just use the predictions with 6 words for now (gt is 6 words)
        for i in range(6):
            gt_word, pred_word = gt_split[i], pred_split[i]
            if gt_word == 'sp':
                continue
            confusion_l = word_confusion_d.get(gt_word, [])
            confusion_l.append(pred_word)
            word_confusion_d[gt_word] = confusion_l

    def get_stats(phrase):
        confusion_count, av_wer = 0, 0
        phrase_words = phrase.split(' ')
        for word in phrase_words:
            confusion_l = word_confusion_d[word]
            unique_confusion_s = set(confusion_l) - {word}
            confusion_count += len(unique_confusion_s)
            av_wer += wer([word] * len(confusion_l), confusion_l)

        return confusion_count, av_wer / len(phrase_words)

    if args.phrase:
        print(args.phrase, get_stats(args.phrase))
        return

    data = []
    for word in word_confusion_d.keys():
        data.append([word, *get_stats(word)])

    # plot unique word confusion counts
    data = sorted(data, key=lambda x: x[1], reverse=True)
    words, confusion_counts, _ = zip(*data)
    df = pd.DataFrame({'confusion_count': confusion_counts}, index=words)
    df.plot.barh(width=0.25, figsize=(8, 8))
    plt.xlabel('Unique Confusion Count')
    plt.ylabel('Word')
    plt.title(test_dir.name)
    plt.tight_layout()
    plt.show()

    # plot unique word WERs
    data = sorted(data, key=lambda x: x[2], reverse=True)
    words, _, wers = zip(*data)
    df = pd.DataFrame({'wer': wers}, index=words)
    df.plot.barh(width=0.25, figsize=(8, 8))
    plt.xlabel('WER')
    plt.ylabel('Word')
    plt.title(test_dir.name)
    plt.tight_layout()
    plt.show()

    print('Av. Confusion Count:', np.mean(confusion_counts))
    print('Av. WER:', np.mean(wers))


def analyse_kaldi_asr(args):
    decode_path = Path(args.decode_path)
    gt_df = pd.read_csv(args.groundtruth_path, names=['Video Name', 'GT'])

    r1_tally = 0
    total_tally = 0
    word_matches_d = {}
    for decode_log in decode_path.glob('decode.*.log'):
        with decode_log.open('r') as f:
            decode_lines = f.read()
        matches = re.findall(r'([a-zA-Z0-9_-]+) (.+)\nLOG .+ Log-like', decode_lines)
        total_tally += len(matches)
        for video_name, prediction in matches:
            gt = gt_df[gt_df['Video Name'] == video_name]['GT'].values[0]
            prediction = prediction.lower().strip()

            # rank 1 accuracy tally
            if prediction == gt:
                r1_tally += 1

            # get word matches (6 words per utterance)
            gt_split, prediction_split = gt.split(' '), prediction.split(' ')
            for i in range(6):
                gt_word, prediction_word = gt_split[i], prediction_split[i]
                if gt_word == 'sp': continue
                word_matches = word_matches_d.get(gt_word, [])
                word_matches.append(prediction_word)
                word_matches_d[gt_word] = word_matches

    data_d = {}
    for word, word_matches in word_matches_d.items():
        _wer = round(wer([word] * len(word_matches), word_matches), 1)
        data_d[word] = _wer
    data = sorted(data_d.items(), key=lambda x: x[1], reverse=True)

    # plot unique word WERs
    words, wers = zip(*data)
    # df = pd.DataFrame({'wer': wers}, index=words)
    # df.plot.barh(width=0.25, figsize=(8, 8))
    # plt.xlabel('WER')
    # plt.ylabel('Word')
    # plt.title('WERs or unique GRID words')
    # plt.tight_layout()
    # plt.xlim((0, 1))
    # plt.show()

    # makeup of words, letters above 50%
    num_words = len(data_d.keys())
    words_above_half_wer = {word: _wer for word, _wer in data_d.items() if _wer > 0.5}
    num_words_above_half_wer = len(words_above_half_wer)

    letters_above_half_wer = {word: _wer for word, _wer in words_above_half_wer.items() if word in LETTERS}
    num_letters_above_half_wer = len(letters_above_half_wer)

    # num_letters = sum([1 for letter in LETTERS if letter in data_d])
    # num_letters_above_half_wer = sum([1 for letter in LETTERS if data_d.get(letter, 0) > 0.5])
    # worst_performing_letters = sorted({letter: data_d.get(letter, 0) for letter in LETTERS}.items(),
    #                                   key=lambda x: x[1])[-3:]

    print('Rank 1 Accuracy:', r1_tally / total_tally)
    print('Av. WER:', np.mean(wers))

    print('Num words:', num_words)
    print('% words above 50% WER:', num_words_above_half_wer / num_words)
    print('% words above 50% WER that are letters:', num_letters_above_half_wer / num_words_above_half_wer)

    # print('Num letters:', num_letters)
    # print('% letters above 50% WER:', num_letters_above_half_wer / num_letters)
    # print('Worst performing letters:', worst_performing_letters)


class WFST(object):
    """
    WFST class.
    Notes:
        * Weight (arc & state) is stored as raw probability, then normalized and converted to negative log likelihood/probability before export.
    """

    zero = float('inf')  # Weight of non-final states; a state is final if and only if its weight is not equal to self.zero
    one = 0.0
    eps = u'<eps>'
    eps_disambig = u'#0'
    silent_labels = frozenset((eps, eps_disambig, u'!SIL'))
    native = property(lambda self: False)

    def __init__(self):
        self.clear()

    def clear(self):
        self._arc_table_dict = collections.defaultdict(list)  # { src_state: [[src_state, dst_state, label, olabel, weight], ...] }  # list of its outgoing arcs
        self._state_table = dict()  # { id: weight }
        self._next_state_id = 0
        self.start_state = self.add_state()
        self.filename = None

    num_arcs = property(lambda self: sum(len(arc_list) for arc_list in itervalues(self._arc_table_dict)))
    num_states = property(lambda self: len(self._state_table))

    def iter_arcs(self):
        return itertools.chain.from_iterable(itervalues(self._arc_table_dict))

    def is_state_final(self, state):
        return (self._state_table[state] != 0)

    def add_state(self, weight=None, initial=False, final=False):
        """ Default weight is 1. """
        self.filename = None
        id = int(self._next_state_id)
        self._next_state_id += 1
        if weight is None:
            weight = 1 if final else 0
        else:
            assert final
        self._state_table[id] = float(weight)
        if initial:
            self.add_arc(self.start_state, id, None)
        return id

    def add_arc(self, src_state, dst_state, label, olabel=None, weight=None):
        """ Default weight is 1. None label is replaced by eps. Default olabel of None is replaced by label. """
        self.filename = None
        if label is None: label = self.eps
        if olabel is None: olabel = label
        if weight is None: weight = 1
        self._arc_table_dict[src_state].append(
            [int(src_state), int(dst_state), text_type(label), text_type(olabel), float(weight)])

    def get_fst_text(self, eps2disambig=False):
        eps_replacement = self.eps_disambig if eps2disambig else self.eps
        arcs_text = u''.join("%d %d %s %s %f\n" % (
            src_state,
            dst_state,
            ilabel if ilabel != self.eps else eps_replacement,
            olabel,
            -math.log(weight) if weight != 0 else 0,
            # -math.log(weight) if weight != 0 else self.zero,
        ) for (src_state, dst_state, ilabel, olabel, weight) in self.iter_arcs())

        states_text = u''.join("%d %f\n" % (
            id,
            -math.log(weight) if weight != 0 else self.zero,
        ) for (id, weight) in iteritems(self._state_table) if weight != 0)

        text = arcs_text + states_text

        return text


def create_grid_fst_strict_grammar(args):
    fst = WFST()

    # TODO: should action state be initial state?
    current_state = fst.add_state(initial=True)
    action_state = fst.add_state()
    colour_state = fst.add_state()
    preps_state = fst.add_state()
    letter_state = fst.add_state()
    digit_state = fst.add_state()
    adverb_state = fst.add_state(final=True)

    for action in ACTIONS:
        fst.add_arc(current_state, action_state, action.upper(), weight=args.weight)
    current_state = action_state
    for colour in COLOURS:
        fst.add_arc(current_state, colour_state, colour.upper(), weight=args.weight)
    current_state = colour_state
    for prep in PREPS:
        fst.add_arc(current_state, preps_state, prep.upper(), weight=args.weight)
    current_state = preps_state
    for letter in LETTERS:
        fst.add_arc(current_state, letter_state, letter.upper(), weight=args.weight)
    current_state = letter_state
    for digit in DIGITS:
        fst.add_arc(current_state, digit_state, digit.upper(), weight=args.weight)
    current_state = digit_state
    for adverb in ADVERBS:
        fst.add_arc(current_state, adverb_state, adverb.upper(), weight=args.weight)

    print(fst.get_fst_text())


def create_grid_fst_free_grammar(args):
    fst = WFST()

    initial_state = fst.add_state(initial=True)
    final_state = fst.add_state(final=True)

    words = [*ACTIONS, *COLOURS, *PREPS, *LETTERS, *DIGITS, *ADVERBS]
    words = [w.upper() for w in words]

    states_d = {}

    def get_state(word):
        if word not in states_d:
            word_state = fst.add_state()
            fst.add_arc(initial_state, word_state, word, weight=args.weight)  # add arc from initial to word state
            fst.add_arc(word_state, final_state, None)  # add arc from word state to final state
            states_d[word] = word_state

        return states_d[word]

    for word in words:
        word_state = get_state(word)
        other_words = list(set(words) - {word})
        for other_word in other_words:
            other_word_state = get_state(other_word)
            fst.add_arc(word_state, other_word_state, other_word, weight=args.weight)

    with open('grammar.txt', 'w') as f:
        f.write(fst.get_fst_text())


def create_shuffled_grid_phrases(args):
    indices = list(range(6))
    words = [ACTIONS, COLOURS, PREPS, LETTERS, DIGITS, ADVERBS]

    phrase_counter = 0
    while phrase_counter != args.num_phrases:
        random.shuffle(indices)
        phrase = ''
        for i in indices:
            word = random.choice(words[i])
            phrase += f'{word} '
        print(phrase_counter+1, phrase)
        phrase_counter += 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    sub_parsers = parser.add_subparsers(dest='run_type')

    parser_1 = sub_parsers.add_parser('generate_dataset')
    parser_1.add_argument('dataset_path', help='/shared/HDD/datasets/gridcorpus')
    parser_1.add_argument('num_samples_per_word_per_person', type=int)
    parser_1.add_argument('output_directory')
    parser_1.add_argument('--forced_alignment_host', default='0.0.0.0')
    parser_1.add_argument('--stats_only', action='store_true')
    parser_1.add_argument('--num_speakers', type=int)
    parser_1.add_argument('--video_glob', default='*.mp4')
    parser_1.add_argument('--skip_assert', action='store_true')
    parser_1.add_argument('--tightly_crop', action='store_true')

    parser_2 = sub_parsers.add_parser('analyse_google_asr')
    parser_2.add_argument('test_results_path')
    parser_2.add_argument('groundtruth_path')
    parser_2.add_argument('--phrase')

    parser_3 = sub_parsers.add_parser('analyse_kaldi_asr')
    parser_3.add_argument('decode_path')
    parser_3.add_argument('groundtruth_path')

    parser_4 = sub_parsers.add_parser('create_grid_fst_strict_grammar')
    parser_4.add_argument('--weight', type=int)

    parser_5 = sub_parsers.add_parser('create_grid_fst_free_grammar')
    parser_5.add_argument('--weight', type=int)

    parser_6 = sub_parsers.add_parser('create_shuffled_grid_phrases')
    parser_6.add_argument('--num_phrases', type=int, default=20)

    f = {
        'generate_dataset': generate_dataset,
        'analyse_google_asr': analyse_google_asr,
        'analyse_kaldi_asr': analyse_kaldi_asr,
        'create_grid_fst_strict_grammar': create_grid_fst_strict_grammar,
        'create_grid_fst_free_grammar': create_grid_fst_free_grammar,
        'create_shuffled_grid_phrases': create_shuffled_grid_phrases
    }
    args = parser.parse_args()
    f.get(args.run_type, exit)(args)
