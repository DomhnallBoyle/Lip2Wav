import argparse
import re
import shutil
import subprocess
from pathlib import Path

import pandas as pd


def inference(args):
    # create wav.scp, spk2utt, utt2spk and text files
    test_directory = Path(args.test_directory)
    kaldi_directory = Path(args.kaldi_directory)

    # load groundtruth data
    data = []
    with open(args.groundtruth_path, 'r') as f:
        for line in f.read().splitlines():
            video_name, phrase, alternatives = re.match(r'([a-zA-Z0-9-_]+),([a-z0-9 \']+),?(\[.+\])?', line).groups()
            data.append([video_name, phrase, alternatives])
    df = pd.DataFrame(data=data, columns=['Video Name', 'Phrase', 'Alternatives'])

    kaldi_data_directory = test_directory.joinpath('kaldi_data').resolve()
    if kaldi_data_directory.exists():
        shutil.rmtree(str(kaldi_data_directory))
    kaldi_data_directory.mkdir()

    synthesised_data = []
    for generated_audio_path in test_directory.glob(args.audio_glob_path):
        name = generated_audio_path.parents[0].name
        gt = df[df['Video Name'] == name]['Phrase'].values[0].upper()
        synthesised_data.append([str(generated_audio_path.resolve()), name, gt])
    synthesised_data = sorted(synthesised_data, key=lambda x: x[1])  # sort by name alphabetically

    with kaldi_data_directory.joinpath('wav.scp').open('w') as f1, \
            kaldi_data_directory.joinpath('text').open('w') as f2, \
            kaldi_data_directory.joinpath('spk2utt').open('w') as f3:
        for generated_audio_path, name, gt in synthesised_data:
            f1.write(f'{name} {generated_audio_path}\n')
            f2.write(f'{name} {gt}\n')
            f3.write(f'{name} {name}\n')
    shutil.copyfile(kaldi_data_directory.joinpath('spk2utt'), kaldi_data_directory.joinpath('utt2spk'))

    def run_kaldi_command(cmd):
        subprocess.run(f'./path_2.sh; {cmd}', cwd=str(kaldi_directory), shell=True)

    # feature extraction
    data_directory = kaldi_data_directory.joinpath('synthesised')
    run_kaldi_command(f'./utils/copy_data_dir.sh {str(kaldi_data_directory)} {str(data_directory)}')
    run_kaldi_command(f'./steps/make_mfcc.sh --nj 20 --mfcc-config conf/mfcc_hires.conf --cmd "run.pl" {str(data_directory)}')
    run_kaldi_command(f'./steps/compute_cmvn_stats.sh {str(data_directory)}')
    run_kaldi_command(f'./utils/fix_data_dir.sh {str(data_directory)}')

    # decoding
    num_samples = int(subprocess.check_output(f'wc -l <{str(data_directory)}/spk2utt', shell=True).decode('utf-8'))
    decode_directory = kaldi_directory.joinpath('exp/chain_cleaned/tdnn_1d_sp/decode_test')
    if decode_directory.exists():
        shutil.rmtree(str(decode_directory))
    run_kaldi_command(f'./steps/online/nnet2/extract_ivectors_online.sh --cmd "run.pl" --nj "{num_samples}" {str(data_directory)} exp/nnet3_cleaned/extractor exp/nnet3_cleaned/ivectors_test')
    run_kaldi_command(f'./steps/nnet3/decode.sh --acwt 1.0 --post-decode-acwt 10.0 --nj 8 --cmd "run.pl" --online-ivector-dir exp/nnet3_cleaned/ivectors_test {args.graph_directory} {str(data_directory)} {str(decode_directory)}')

    # run analysis
    args.results_directory = decode_directory
    analysis(args)


def analysis(args):
    print('\nAnalysis:\n')
    results_directory = Path(args.results_directory)

    def get_tally(_data):
        r1_tally = 0
        for gt, pred in _data:
            if gt.split('_')[0] == pred.replace('\'', '').replace(' ', ''):
                r1_tally += 1

        return round(r1_tally / len(data), 4) * 100

    data = []
    for decode_log in results_directory.joinpath('log').glob('decode.*.log'):
        with decode_log.open('r') as f:
            lines = f.read()
        matches = re.findall(r'([a-zA-Z0-9_-]+) (.+)\nLOG .+ Log-like', lines)
        for name, prediction in matches:
            data.append([name.lower(), prediction.lower()])

    with results_directory.joinpath('scoring_kaldi').joinpath('best_wer').open('r') as f:
        lines = f.read().splitlines()
        wer = float(re.match(r'.+ (\d+\.\d+) .+', lines[0]).groups()[0])

    print('R1 Acc %:', get_tally(data))
    print('WER:', wer)


def main(args):
    {
        'inference': inference,
        'analysis': analysis
    }[args.run_type](args)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    sub_parsers = parser.add_subparsers(dest='run_type')

    parser_1 = sub_parsers.add_parser('inference')
    parser_1.add_argument('test_directory')
    parser_1.add_argument('groundtruth_path')
    parser_1.add_argument('kaldi_directory')
    parser_1.add_argument('graph_directory')
    parser_1.add_argument('--audio_glob_path', default='*/generated_audio.wav')

    parser_2 = sub_parsers.add_parser('analysis')
    parser_2.add_argument('results_directory')

    main(parser.parse_args())
