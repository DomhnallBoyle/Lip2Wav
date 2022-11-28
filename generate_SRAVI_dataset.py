"""take the SRAVI 100 dataset and separate it out to same format as LRW for testing with"""
import argparse
import shutil
from pathlib import Path


def main(args):
    with open(args.phrases_path, 'r') as f:
        phrases = f.read().splitlines()

    input_dataset_path = Path(args.input_dataset_path)
    user_directories = list(input_dataset_path.glob('*'))
    user_ids = [p.name for p in user_directories]

    output_dataset_path = Path(args.output_dataset_path)
    if output_dataset_path.exists():
        shutil.rmtree(str(output_dataset_path))
    output_dataset_path.mkdir()

    for i, phrase in enumerate(phrases):
        phrase = phrase.upper().replace(' ', '_')
        output_phrase_directory = output_dataset_path.joinpath(phrase).joinpath('test')
        output_phrase_directory.mkdir(parents=True)

        counter = 1
        for user_directory in user_directories:
            user_id = user_directory.name.upper()
            video_paths = list(user_directory.glob(f'SRAVIExtended-{user_id}-P{i+1}-S*.mp4'))
            for video_path in video_paths:
                from_path = str(video_path)
                to_path = str(output_phrase_directory.joinpath(f'{counter}.mp4'))
                shutil.copyfile(from_path, to_path)
                counter += 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('input_dataset_path')
    parser.add_argument('output_dataset_path')
    parser.add_argument('phrases_path')

    main(parser.parse_args())
