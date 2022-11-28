import json
import random
import re
import subprocess
import tempfile
from datetime import timedelta
from http import HTTPStatus

import cv2
import numpy as np
import requests
from vidaug import augmentors as va

# this is a static build from https://www.johnvansickle.com/ffmpeg/old-releases/ffmpeg-4.4.1-i686-static.tar.xz
# requires new ffmpeg version for:
# - duration of extracted audio == video
# - contains x264 codec in build required for clean video frames
FFMPEG_PATH = '/opt/lip2wav/ffmpeg-4.4.1-i686-static/ffmpeg'
FFPROBE_PATH = '/opt/lip2wav/ffmpeg-4.4.1-i686-static/ffprobe'
OLD_FFMPEG_PATH = 'ffmpeg-2.8.15'

FFMPEG_OPTIONS = '-hide_banner -loglevel error'

VIDEO_CROP_COMMAND = f'{FFMPEG_PATH} {FFMPEG_OPTIONS} -y -i {{input_video_path}} -ss {{start_time}} -to {{end_time}} -async 1 {{output_video_path}}'
VIDEO_INFO_COMMAND = f'{FFMPEG_PATH} -i {{input_video_path}}'
VIDEO_DURATION_COMMAND = f'{FFPROBE_PATH} {FFMPEG_OPTIONS} -show_entries format=duration -of default=noprint_wrappers=1:nokey=1 {{video_path}}'
VIDEO_TO_AUDIO_COMMAND = f'{{ffmpeg_path}} {FFMPEG_OPTIONS} -threads 1 -y -i {{input_video_path}} -async 1 -ac 1 -vn -acodec pcm_s16le -ar {{sr}} {{output_audio_path}}'
VIDEO_CONVERT_FPS_COMMAND = f'{FFMPEG_PATH} {FFMPEG_OPTIONS} -y -i {{input_video_path}} -strict -2 -filter:v fps=fps={{fps}} {{output_video_path}}'  # copies original codecs and metadata (rotation)
VIDEO_SPEED_ALTER_COMMAND = f'{FFMPEG_PATH} {FFMPEG_OPTIONS} -y -i {{input_video_path}} -filter_complex "[0:v]setpts={{video_speed}}*PTS[v];[0:a]atempo={{audio_speed}}[a]" -map "[v]" -map "[a]" {{output_video_path}}'
VIDEO_REMOVE_AUDIO_COMMAND = f'{FFMPEG_PATH} {FFMPEG_OPTIONS} -y -i {{input_video_path}} -c copy -an {{output_video_path}}'
VIDEO_ADD_AUDIO_COMMAND = f'{FFMPEG_PATH} {FFMPEG_OPTIONS} -y -i {{input_video_path}} -i {{input_audio_path}} -strict -2 -c:v copy -c:a aac {{output_video_path}}'


def get_num_frames(video_path):
    video_capture = cv2.VideoCapture(video_path)
    num_frames = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
    video_capture.release()

    return num_frames


def get_video_frame(video_path, index, rotation=None):
    if rotation is None:
        rotation = get_video_rotation(video_path)

    video_capture = cv2.VideoCapture(video_path)
    i = 0
    selected_frame = None
    while True:
        success, frame = video_capture.read()
        if not success:
            break
        if i == index:
            frame = fix_frame_rotation(frame, rotation)
            selected_frame = frame
            break
        i += 1

    video_capture.release()

    return selected_frame


def get_video_duration(video_path):
    result = subprocess.check_output(VIDEO_DURATION_COMMAND.format(video_path=video_path).split(' '),
                                     stderr=subprocess.STDOUT).decode()

    return float(result)


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


def get_video_frames(video_path, rotation=None, greyscale=False):
    if rotation is None:
        rotation = get_video_rotation(video_path)

    video_reader = cv2.VideoCapture(video_path)
    frames = []
    while True:
        success, frame = video_reader.read()
        if not success:
            break
        if greyscale:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # returns single channel frame
        frame = fix_frame_rotation(frame, rotation)
        frames.append(frame)

    video_reader.release()

    return frames


def save_video_frames(video_frames, video_path, fps, colour): 
    height, width = video_frames[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')

    video_writer = cv2.VideoWriter(video_path, fourcc, fps, (width, height), colour)
    for frame in video_frames: 
        video_writer.write(frame.astype(np.uint8))
    video_writer.release()


def show_frames(video_frames, delay, title):
    for frame in video_frames:
        cv2.imshow(title, frame)
        cv2.waitKey(delay)


def run_video_augmentation(video_path, new_video_path, random_prob=0.5):
    if random.random() < random_prob:
        # https://trac.ffmpeg.org/wiki/How%20to%20speed%20up%20/%20slow%20down%20a%20video
        # speed required between 0 and 2
        # < 1 = slow down
        # > 1 = speed up
        speed = round(random.uniform(0.5, 1.5), 2)

        subprocess.call(VIDEO_SPEED_ALTER_COMMAND.format(
            input_video_path=video_path,
            output_video_path=new_video_path,
            video_speed=round(1. / speed, 2),
            audio_speed=float(speed)
        ), shell=True)

        return new_video_path

    return video_path


class RandomRotate:

    def __init__(self, degrees):
        self.degrees = degrees

    def __call__(self, clip):
        image_center = tuple(np.array(clip[0].shape[1::-1]) / 2)
        rot_mat = cv2.getRotationMatrix2D(image_center, self.degrees, 1.0)

        return [cv2.warpAffine(frame, rot_mat, frame.shape[1::-1], flags=cv2.INTER_LINEAR)
                for frame in clip]


def run_frame_augmentation(frames, method, intensity_aug=False, random_prob=0.5, rotation_range=10, intensity_range=30):
    sometimes = lambda aug: va.Sometimes(random_prob, aug)
    random_int = lambda max: np.random.randint(-max, max)  # inclusive

    # TODO: Zoom in/out

    if method == 'full':
        seq = va.Sequential([
            RandomRotate(degrees=random_int(rotation_range)),  # random rotate of angle between (-degrees, degrees)
        ])
    elif method == 'mouth':
        augs = [
            sometimes(va.HorizontalFlip()),  # flip video horizontally
        ]
        if intensity_aug:
            augs += [sometimes(va.Add(random_int(intensity_range)))]  # add random value to pixels between (-max, max)
        seq = va.Sequential(augs)
    else:
        print(f'{method} does not exist')
        return

    return seq(frames)


def extract_audio(video_path, sr=16000, use_old_ffmpeg=False):
    audio_file = tempfile.NamedTemporaryFile(suffix='.wav')

    if use_old_ffmpeg:
        ffmpeg_path = OLD_FFMPEG_PATH
    else:
        ffmpeg_path = FFMPEG_PATH

    subprocess.call(VIDEO_TO_AUDIO_COMMAND.format(
        ffmpeg_path=ffmpeg_path,
        input_video_path=video_path,
        sr=sr,
        output_audio_path=audio_file.name
    ), shell=True)

    return audio_file


def convert_fps(video_path, new_video_path, fps):
    subprocess.call(VIDEO_CONVERT_FPS_COMMAND.format(
        input_video_path=video_path,
        output_video_path=new_video_path,
        fps=fps
    ), shell=True)

    return new_video_path


def replace_audio(video_path, audio_path, output_video_path):
    with tempfile.NamedTemporaryFile(suffix='.mp4') as f:
        subprocess.call(VIDEO_REMOVE_AUDIO_COMMAND.format(
            input_video_path=video_path,
            output_video_path=f.name
        ), shell=True)

        subprocess.call(VIDEO_ADD_AUDIO_COMMAND.format(
            input_video_path=f.name,
            input_audio_path=audio_path,
            output_video_path=output_video_path
        ), shell=True)


def get_lip_embeddings(video_path):
    with open(video_path, 'rb') as f:
        response = requests.post('http://127.0.0.1:6002/lip_embeddings', files={'video': f.read()})
        if response.status_code != 200:
            print(video_path, 'failed to extract ARK', response.status_code)
            return

        return json.loads(response.content)


def crop(video_path, start, end):
    suffix = video_path.split('/')[-1].split('.')[1]
    output_video_path = f'/tmp/cropped_video.{suffix}'

    subprocess.call(VIDEO_CROP_COMMAND.format(
        input_video_path=video_path,
        start_time='0' + str(timedelta(seconds=start))[:-3],
        end_time='0' + str(timedelta(seconds=end))[:-3],
        output_video_path=output_video_path
    ), shell=True)

    return output_video_path


def run_cfe_cropper(video_path, host):
    with open(video_path, 'rb') as f:
        response = requests.post(f'{host}/api/v1/extract/', files={'video': f.read()}, verify=False)
    if response.status_code != HTTPStatus.OK:
        print(response.__dict__)
        return

    # cfe returns video in .avi format
    with tempfile.NamedTemporaryFile(suffix='.avi') as f:
        for chunk in response.iter_content(chunk_size=1024):
            f.write(chunk)
        f.seek(0)

        return get_video_frames(f.name, greyscale=True)
