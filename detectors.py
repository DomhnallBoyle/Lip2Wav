import argparse
import time

import cv2
import dlib
import numpy as np

MOUTH_WIDTH = 100
MOUTH_HEIGHT = 50
HORIZONTAL_PAD = 0.19
ROI = {
    'eye1': slice(36, 42),
    'eye2': slice(42, 48),
    'mouth': slice(48, 68)
}

dlib_face_detector = dlib.get_frontal_face_detector()
dlib_shape_predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')


def get_face_landmarks(frame, face_coords=None):
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    if face_coords is None:
        face_rects = dlib_face_detector(frame_gray, 0)
        if not face_rects:
            return
        face_coords = face_rects[0]

    landmarks = dlib_shape_predictor(frame_gray, face_coords)

    return face_coords, np.array([[p.x, p.y] for p in landmarks.parts()])


def get_roi_mid_point(roi):
    x, y, w, h = cv2.boundingRect(roi)
    mid_x = x + w // 2
    mid_y = y + h // 2

    return mid_x, mid_y


def align_frame(frame, landmarks):
    left_eye_x, left_eye_y = get_roi_mid_point(landmarks[ROI['eye1']])
    right_eye_x, right_eye_y = get_roi_mid_point(landmarks[ROI['eye2']])

    delta_x = right_eye_x - left_eye_x
    delta_y = right_eye_y - left_eye_y

    try:
        angle = np.arctan(delta_y / delta_x)
        angle = (angle * 180) / np.pi
    except ZeroDivisionError:
        angle = 0

    height, width = frame.shape[:2]
    center = (width // 2, height // 2)
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(frame, rotation_matrix, (width, height))

    return rotated


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


def get_mouth_frames_old(frames, method='dlib'):
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


def get_mouth_frames(frames):
    # https://github.com/Chris10M/Lip2Speech/blob/81a57364054c7fbbede7c6de15861e49d53419d0/datasets/face_utils.py
    normalize_ratio = None
    mouth_frames = []

    for frame in frames:
        face_stats = get_face_landmarks(frame)
        if face_stats is None:
            return []
        face_coords, landmarks = face_stats

        try:
            frame = align_frame(frame, landmarks)
        except cv2.error:
            return []

        _, landmarks = get_face_landmarks(frame, face_coords)
        mouth_points = landmarks[ROI['mouth']]
        mouth_centroid = np.mean(mouth_points[:, -2:], axis=0)

        if normalize_ratio is None:
            mouth_left = np.min(mouth_points[:, :-1]) * (1.0 - HORIZONTAL_PAD)
            mouth_right = np.max(mouth_points[:, :-1]) * (1.0 + HORIZONTAL_PAD)

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


def get_mouth_frames_wrapper(frames, use_old_method=False):
    if use_old_method:
        return get_mouth_frames_old(frames, method='dlib')

    return get_mouth_frames(frames)


def main(args):
    from video_utils import get_fps, get_video_frames, get_video_rotation, show_frames

    rotation = get_video_rotation(args.video_path)
    frames = get_video_frames(args.video_path, rotation=rotation)
    fps = get_fps(args.video_path)

    show_frames(frames, fps, 'Frames')

    start_time = time.time()
    mouth_frames = get_mouth_frames_old(frames, method='dlib')
    print('Took', time.time() - start_time)
    show_frames(mouth_frames, fps, 'Mouth Frames Old')
    cv2.imwrite('mouth_frames_old.jpg', mouth_frames[15])

    start_time = time.time()
    mouth_frames = get_mouth_frames(frames)
    print('Took', time.time() - start_time)
    show_frames(mouth_frames, fps, 'Mouth Frames New')
    cv2.imwrite('mouth_frames_new.jpg', mouth_frames[15])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('video_path')

    main(parser.parse_args())
