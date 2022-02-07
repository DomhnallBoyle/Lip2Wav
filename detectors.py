import argparse

import cv2
import dlib
import numpy as np

import face_alignment

MOUTH_WIDTH = 100
MOUTH_HEIGHT = 50
HORIZONTAL_PAD = 0.19

s3fd_fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, flip_input=False, device='cpu')
dlib_face_detector = dlib.get_frontal_face_detector()
dlib_shape_predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')


def get_mouth_points(frame, method='dlib'):
    mouth_points = []
    if method == 'dlib':
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        rects = dlib_face_detector(frame, 0)

        shape = None
        for rect in rects:
            shape = dlib_shape_predictor(frame, rect)

        if shape is None:  # no face predictions, return empty
            return []

        for i in range(48, 68):
            part = shape.part(i)
            mouth_points.append((part.x, part.y))
    elif method == 's3fd':
        landmarks = s3fd_fa.get_landmarks_from_image(np.asarray(frame))
        if landmarks is None:
            return []

        for i in range(len(landmarks)):
            if i < 48:  # only extract mouth region
                continue
            mouth_points.append((landmarks[i][0], landmarks[i][1]))

    return mouth_points


def get_mouth_frames(frames, method='dlib'):
    normalize_ratio = None
    mouth_frames = []
    for frame in frames:
        mouth_points = get_mouth_points(frame, method=method)
        if not mouth_points:  # need mouth points for every frame
            return []

        np_mouth_points = np.array(mouth_points)
        mouth_centroid = np.mean(np_mouth_points[:, -2:], axis=0)

        if normalize_ratio is None:
            mouth_left = np.min(np_mouth_points[:, :-1]) * (1.0 - HORIZONTAL_PAD)
            mouth_right = np.max(np_mouth_points[:, :-1]) * (1.0 + HORIZONTAL_PAD)

            # mouth_left = np.min(np_mouth_points[:, :-1])
            # mouth_right = np.max(np_mouth_points[:, :-1])
            #
            # width_pad = (mouth_right - mouth_left) * 0.19
            #
            # mouth_left -= width_pad
            # mouth_right += width_pad

            normalize_ratio = MOUTH_WIDTH / float(mouth_right - mouth_left)

        new_img_shape = (int(frame.shape[0] * normalize_ratio), int(frame.shape[1] * normalize_ratio))
        resized_img = cv2.resize(frame, (new_img_shape[1], new_img_shape[0]))

        mouth_centroid_norm = mouth_centroid * normalize_ratio

        mouth_l = int(mouth_centroid_norm[0] - MOUTH_WIDTH / 2)
        mouth_r = int(mouth_centroid_norm[0] + MOUTH_WIDTH / 2)
        mouth_t = int(mouth_centroid_norm[1] - MOUTH_HEIGHT / 2)
        mouth_b = int(mouth_centroid_norm[1] + MOUTH_HEIGHT / 2)

        mouth_crop_image = resized_img[mouth_t:mouth_b, mouth_l:mouth_r]
        if mouth_crop_image.shape != (MOUTH_HEIGHT, MOUTH_WIDTH, 3):
            mouth_crop_image = cv2.resize(mouth_crop_image, (MOUTH_WIDTH, MOUTH_HEIGHT))

        mouth_frames.append(mouth_crop_image)

    return mouth_frames


def main(args):
    from video_utils import get_fps, get_video_frames, show_frames

    frames = get_video_frames(args.video_path)
    fps = get_fps(args.video_path)
    mouth_frames = get_mouth_frames(frames, method=args.method)
    show_frames(mouth_frames, fps, 'Mouth Frames')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('video_path')
    parser.add_argument('method', choices=['dlib', 's3fd'])

    main(parser.parse_args())
