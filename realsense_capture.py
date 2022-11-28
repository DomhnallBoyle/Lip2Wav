import argparse
import ast
import shutil
import time
import uuid
from pathlib import Path

import cv2
import numpy as np
import pyaudio
import pyrealsense2 as rs
import wave

from video_utils import replace_audio

CHUNK = 1024
SAMPLE_FORMAT = pyaudio.paInt16
CHANNELS = 1
SAMPLE_RATE = 16000

WARM_UP_TIME = 1

INFRARED_TMP = '/tmp/infrared.mp4'
COLOUR_TMP = '/tmp/colour.mp4'
AUDIO_PATH = '/tmp/audio.wav'


def main(args):
    print(args.phrases)

    output_directory = Path(args.output_directory)
    if args.redo and output_directory.exists():
        shutil.rmtree(output_directory)
    output_directory.mkdir(exist_ok=True)

    config = rs.config()
    config.enable_stream(rs.stream.infrared)
    config.enable_stream(rs.stream.color)

    for i, phrase in enumerate(args.phrases):
        pipeline = rs.pipeline()
        pipeline_profile = pipeline.start(config)

        # disable dot laser pattern
        device = pipeline_profile.get_device()
        depth_sensor = device.query_sensors()[0]
        depth_sensor.set_option(rs.option.emitter_enabled, 0)

        print(f'Recording {i+1}:', phrase)
        phrase = phrase.replace('\'', '').replace(' ', '').lower()

        infrared_video = cv2.VideoWriter(INFRARED_TMP, cv2.VideoWriter_fourcc(*'mp4v'), 15, (848, 480))
        colour_video = cv2.VideoWriter(COLOUR_TMP, cv2.VideoWriter_fourcc(*'mp4v'), 15, (640, 480))

        p = pyaudio.PyAudio()
        audio_stream = p.open(format=SAMPLE_FORMAT,
                              channels=CHANNELS,
                              rate=SAMPLE_RATE,
                              frames_per_buffer=CHUNK,
                              input=True)
        audio_frames = []

        warmed_up = False
        warm_up_start = None

        try:
            while True:
                audio_data = audio_stream.read(CHUNK)

                # capture frames
                frames = pipeline.wait_for_frames()
                infrared_frame = cv2.cvtColor(np.asanyarray(frames.get_infrared_frame().get_data()), cv2.COLOR_GRAY2BGR)
                colour_frame = cv2.cvtColor(np.asanyarray(frames.get_color_frame().get_data()), cv2.COLOR_RGB2BGR)

                if not warmed_up:
                    if warm_up_start is None:
                        warm_up_start = time.time()

                    current_time = time.time()
                    if current_time - warm_up_start > WARM_UP_TIME:
                        print('Warmed up...')
                        warmed_up = True

                    continue

                # save audio
                audio_frames.append(audio_data)

                # save frames
                infrared_video.write(infrared_frame)
                colour_video.write(colour_frame)

                if args.debug:
                    cv2.imshow('Infrared', infrared_frame)
                    cv2.imshow('Colour', colour_frame)
                    if cv2.waitKey(33) == ord('q'):
                        raise KeyboardInterrupt()

        except KeyboardInterrupt:
            audio_stream.stop_stream()
            audio_stream.close()
            p.terminate()

            pipeline.stop()
            cv2.destroyAllWindows()

            infrared_video.release()
            colour_video.release()

        # save audio to file
        wf = wave.open(AUDIO_PATH, 'wb')
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(p.get_sample_size(SAMPLE_FORMAT))
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(b''.join(audio_frames))
        wf.close()

        uuid_str = str(uuid.uuid4())
        replace_audio(
            video_path=INFRARED_TMP,
            audio_path=AUDIO_PATH,
            output_video_path=output_directory.joinpath(f'{phrase}_infrared_{uuid_str}.mp4')
        )
        replace_audio(
            video_path=COLOUR_TMP,
            audio_path=AUDIO_PATH,
            output_video_path=output_directory.joinpath(f'{phrase}_colour_{uuid_str}.mp4')
        )

        if args.pause_every and (i + 1) % args.pause_every == 0:
            print('Pausing...')
            try:
                while True:
                    time.sleep(10)
            except KeyboardInterrupt:
                print('Continuing...')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('phrases', type=ast.literal_eval)
    parser.add_argument('output_directory')
    parser.add_argument('--pause_every', type=int)
    parser.add_argument('--redo', action='store_true')
    parser.add_argument('--debug', action='store_true')

    main(parser.parse_args())
