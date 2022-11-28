"""
Re-run librispeech evaluations
- also play mel-spec with highest activation (in sliding window), does this help in R1 accuracy?
"""
import argparse
import random
import shutil
import tempfile
from pathlib import Path

import numpy as np

from audio_utils import play_audio
from video_inference import preprocess_video, sif, speech_synthesis

AUDIO_PATH = 'SRAVI_embedding_experiments/darryl_good_sample/audio_google.wav'
OUTPUT_DIRECTORY = Path('/shared/synthesised_video')


def main(args):
    test_directory = Path(args.test_directory)

    sif.hparams.set_hparam('eval_ckpt', args.model_checkpoint)
    sif.hparams.set_hparam('img_height', args.image_height)
    sif.hparams.set_hparam('img_width', args.image_width)

    synthesizer = sif.Synthesizer(verbose=False)

    gts, init_preds, preds = [], [], []
    video_paths = list(test_directory.glob('*.mp4'))
    random.shuffle(video_paths)
    video_paths = video_paths[:args.num_samples]

    # get groundtruth from file
    gts_d = {}
    with test_directory.joinpath('groundtruth.csv').open('r') as f:
        for line in f.read().splitlines():
            video_name, gt = line.split(',')
            gts_d[video_name] = gt

    for video_path in video_paths:

        if OUTPUT_DIRECTORY.exists():
            shutil.rmtree(OUTPUT_DIRECTORY)
        OUTPUT_DIRECTORY.mkdir()

        audio_file = tempfile.NamedTemporaryFile(suffix='.wav')
        shutil.copyfile(AUDIO_PATH, audio_file.name)

        # preprocess video
        result = preprocess_video(
            process_index=0,
            video_path=str(video_path),
            fps=25,
            output_directory=OUTPUT_DIRECTORY,
            audio_preprocessing=True,
            speaker_embedding_audio_file=audio_file
        )
        audio_file.close()
        if not result:
            continue

        speech_synthesis(
            synthesizer=synthesizer,
            video_directory=OUTPUT_DIRECTORY,
            combine_audio_and_video=True,
            # save_mels=True,
            save_alignments=True,
            save_wavs=True
        )

        # get initial prediction
        while True:
            play_audio(OUTPUT_DIRECTORY.joinpath('generated_audio.wav'))
            initial_prediction = input('Prediction: ')
            if initial_prediction:
                init_preds.append(initial_prediction)
                break

        if not args.skip_focus:
            # find largest activation
            max_v = 0
            best_wav_path = None
            for alignment_path in OUTPUT_DIRECTORY.glob('alignments_*.npy'):
                alignment_index = alignment_path.name.split('.')[0][-1]

                alignment = np.load(alignment_path)
                # alignment = sif.audio.load_wav(alignment_path, 16000)

                # alignment = np.absolute(alignment)
                percentile = np.percentile(alignment, 50)  # N% is this value or below
                alignment = alignment[alignment > percentile]
                # print(percentile, alignment.shape)

                if alignment.sum() > max_v:
                    max_v = alignment.sum()
                    best_wav_path = OUTPUT_DIRECTORY.joinpath(f'wav_{alignment_index}.wav')

            # get focused prediction
            while True:
                play_audio(best_wav_path)
                prediction = input('Prediction: ')
                if prediction:
                    preds.append(prediction)
                    break

        gts.append(gts_d[video_path.stem])

    def get_accuracy(_gts, _preds):
        tally = 0
        for g, p in zip(_gts, _preds):
            if p in g.split(' '):
                tally += 1

        print(tally / len(_gts), '\n')

    for gt, init_pred, pred in zip(gts, init_preds, preds):
        print(gt, '1:', init_pred, '2:', pred)

    get_accuracy(gts, init_preds)
    get_accuracy(gts, preds)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('test_directory')
    parser.add_argument('model_checkpoint')
    parser.add_argument('--image_height', type=int, default=50)
    parser.add_argument('--image_width', type=int, default=100)
    parser.add_argument('--num_samples', type=int, default=10)
    parser.add_argument('--skip_focus', action='store_true')

    main(parser.parse_args())
