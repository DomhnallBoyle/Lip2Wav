import argparse
import re
import subprocess

import cv2
import numpy as np
from vidaug import augmentors as va


def get_video_rotation(video_path):
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


def apply_augmentation(video):
    sometimes = lambda aug: va.Sometimes(0.5, aug)
    random_int = lambda max: np.random.randint(-max, max)

    seq = va.Sequential([
        sometimes(va.HorizontalFlip()),  # flip video horizontally
        sometimes(va.Add(random_int(20))),  # add random value to pixels between (-max, max)
        sometimes(va.RandomRotate(degrees=10)),  # random rotate of angle between (-degrees, degrees)
        sometimes(va.RandomTranslate(x=10, y=10))  # translates randomly in x from (-x, x), randomly in y from (-y, y)
    ])

    return seq(video)


def main(args):
    video_reader = cv2.VideoCapture(args.video_path)
    fps = int(video_reader.get(cv2.CAP_PROP_FPS))
    rotation = get_video_rotation(args.video_path)
    frames = []
    while True:
        success, frame = video_reader.read()
        if not success:
            break
        frame = fix_frame_rotation(frame, rotation)
        frames.append(frame)

    aug_frames = apply_augmentation(frames)

    for frame, aug_frame in zip(frames, aug_frames):
        cv2.imshow('Original', frame)
        cv2.imshow('Augmented', aug_frame)
        cv2.waitKey(fps)

    cv2.destroyAllWindows()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('video_path')

    main(parser.parse_args())
