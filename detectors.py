import argparse
import os
import time

import cv2
import dlib
import numpy as np
from ibug.face_alignment import FANPredictor
from ibug.face_detection import RetinaFacePredictor

MOUTH_WIDTH = 100
MOUTH_HEIGHT = 50
HORIZONTAL_PAD = 0.19
ROI = {
    'eye1': slice(36, 42),
    'eye2': slice(42, 48),
    'mouth': slice(48, 68)
}

# perspective warp crop
SOURCE_KEY_POINTS = [
    [36, 37, 38, 39, 40, 41],  # right eye
    [42, 43, 44, 45, 46, 47],  # left eye
    [8],  # chin
    [33],  # nose
    [5],  # right jaw
    [11],  # left jaw
    [60, 61, 62, 63, 64, 65, 66, 67],  # mouth inner
    [48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59]  # mouth outer
]
DEST_POINTS = [
    [0.26, 0],
    [0.74, 0],
    [0.5, 0.9],
    [0.5, 0.45],
    [0.05, 0.68],
    [0.95, 0.68],
    [0.5, 0.65],
    [0.5, 0.65]
]
ROI_TOP, ROI_BOTTOM, ROI_LEFT, ROI_RIGHT = 0.28, 0.92, 0.18, 0.82

WINDOW_LENGTH = 12  # to smooth the landmark jitter

dlib_face_detector = dlib.get_frontal_face_detector()
dlib_shape_predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

fan_face_detector = None
fan_landmark_predictor = None


def get_face_landmarks(frame, face_coords=None):
    if len(frame.shape) == 3:
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    else:
        frame_gray = frame

    if face_coords is None:
        face_rects = dlib_face_detector(frame_gray, 0)
        if not face_rects:
            return
        face_coords = face_rects[0]

    landmarks = dlib_shape_predictor(frame_gray, face_coords)

    return face_coords, np.array([[p.x, p.y] for p in landmarks.parts()])


def get_face_landmarks_fan(frame): 
    global fan_face_detector, fan_landmark_predictor
    if fan_face_detector is None or fan_landmark_predictor is None:
        fan_face_detector = RetinaFacePredictor(
            device=os.environ.get('CUDA_VISIBLE_DEVICES', 'cuda'),
            threshold=0.8, 
            model=RetinaFacePredictor.get_model('resnet50')
        )
        fan_landmark_predictor = FANPredictor(
            device=os.environ.get('CUDA_VISIBLE_DEVICES', 'cuda'),
            model=FANPredictor.get_model(os.environ.get('FAN_LANDMARKS_MODEL', '2dfan2'))  # default = 2dfan2, vsrml uses this
        )  

    detected_faces = fan_face_detector(frame, rgb=False)  # i.e. BGR format
    landmarks, _ = fan_landmark_predictor(frame, detected_faces, rgb=False)  # i.e. BGR format

    return detected_faces[0], landmarks[0]


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


def get_mouth_frames(frames, face_stats=None):
    # https://github.com/Chris10M/Lip2Speech/blob/81a57364054c7fbbede7c6de15861e49d53419d0/datasets/face_utils.py
    normalize_ratio = None
    mouth_frames = []

    for i, frame in enumerate(frames):
        if face_stats:
            face_coords, landmarks = face_stats[i]['c'], face_stats[i]['l']
        else:
            _face_stats = get_face_landmarks(frame)
            if _face_stats is None:
                return []
            face_coords, landmarks = _face_stats

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


def affine_transform(
    frame,
    landmarks,
    reference,
    grayscale=False,
    target_size=(256, 256),
    reference_size=(256, 256),
    stable_points=(28, 33, 36, 39, 42, 45, 48, 54),
    interpolation=cv2.INTER_LINEAR,
    border_mode=cv2.BORDER_CONSTANT,
    border_value=0
):
    # Prepare everything
    if grayscale and frame.ndim == 3:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    stable_reference = np.vstack([reference[x] for x in stable_points])
    stable_reference[:, 0] -= (reference_size[0] - target_size[0]) / 2.0
    stable_reference[:, 1] -= (reference_size[1] - target_size[1]) / 2.0

    # Warp the face patch and the landmarks
    transform = cv2.estimateAffinePartial2D(np.vstack([landmarks[x] for x in stable_points]),
                                            stable_reference, method=cv2.LMEDS)[0]
    transformed_frame = cv2.warpAffine(
        frame,
        transform,
        dsize=(target_size[0], target_size[1]),
        flags=interpolation,
        borderMode=border_mode,
        borderValue=border_value,
    )
    transformed_landmarks = np.matmul(landmarks, transform[:, :2].transpose()) + transform[:, 2].transpose()

    return transformed_frame, transformed_landmarks


def cut_patch(img, landmarks, height, width, threshold=5):
    center_x, center_y = np.mean(landmarks, axis=0)

    if center_y - height < 0:                                                
        center_y = height                                                    
    if center_y - height < 0 - threshold:                                    
        raise Exception('too much bias in height')                           
    if center_x - width < 0:                                                 
        center_x = width                                                     
    if center_x - width < 0 - threshold:                                     
        raise Exception('too much bias in width')                            
                                                                             
    if center_y + height > img.shape[0]:                                     
        center_y = img.shape[0] - height                                     
    if center_y + height > img.shape[0] + threshold:                         
        raise Exception('too much bias in height')                           
    if center_x + width > img.shape[1]:                                      
        center_x = img.shape[1] - width                                      
    if center_x + width > img.shape[1] + threshold:                          
        raise Exception('too much bias in width')                            
                                                                             
    cutted_img = np.copy(img[ int(round(center_y) - round(height)): int(round(center_y) + round(height)),
                         int(round(center_x) - round(width)): int(round(center_x) + round(width))])
    
    return cutted_img


def get_mouth_frames_perspective_warp_2(frames, landmarks, greyscale):
    frame_idx = 0
    reference = np.load('20words_mean_face.npy')
    while True:
        try:
            frame = frames[frame_idx]
        except IndexError:
            break
        if frame_idx == 0:
            sequence = []
            sequence_frame = []
            sequence_landmarks = []

        window_margin = min(WINDOW_LENGTH // 2, frame_idx, len(landmarks) - 1 - frame_idx)
        smoothed_landmarks = np.mean([landmarks[x] for x in range(frame_idx - window_margin, frame_idx + window_margin + 1)], axis=0)
        smoothed_landmarks += landmarks[frame_idx].mean(axis=0) - smoothed_landmarks.mean(axis=0)
        transformed_frame, transformed_landmarks = affine_transform(
            frame,
            smoothed_landmarks,
            reference,
            grayscale=greyscale,
        )
        sequence.append( cut_patch( transformed_frame,
                                    transformed_landmarks[48:68],
                                    96 // 2,
                                    96 // 2,))

        sequence_frame.append(transformed_frame)
        sequence_landmarks.append(transformed_landmarks)
        frame_idx += 1

    return np.array(sequence), np.array(sequence_frame), np.array(sequence_landmarks)


def get_mouth_frames_perspective_warp(frames, face_stats=None, reshape_height=88, reshape_width=88):
    mouth_frames = []

    for i, frame in enumerate(frames):
        if face_stats:
            face_coords, landmarks = face_stats[i]['c'], face_stats[i]['l']
        else:
            face_coords, landmarks = get_face_landmarks(frame)

        # get the average point for mouth, nose etc
        source_points = []
        for key_points in SOURCE_KEY_POINTS:
            count = 0
            avg_point = [0, 0]
            num_avgd_points = len(key_points)

            while count < num_avgd_points:
                current_point_num = key_points[count]
                current_key_point = landmarks[current_point_num]
                avg_point[0] += current_key_point[0]
                avg_point[1] += current_key_point[1]
                count += 1

            avg_point[0] /= count
            avg_point[1] /= count
            source_points.append(avg_point)

        real_world_dest_points = []
        height, width = frame.shape[:2]
        frame_size = min(height, width)
        for dp in DEST_POINTS:
            real_world_dest_points.append([dp[0] * frame_size, dp[1] * frame_size])

        # if points don't exist on the frame, it will return a blank image
        transform, _ = cv2.findHomography(np.asarray(source_points), np.asarray(real_world_dest_points))
        warped_face = cv2.warpPerspective(frame, transform, (height, width))

        left, top = ROI_LEFT * frame_size, ROI_TOP * frame_size
        right = left + ((ROI_RIGHT - ROI_LEFT) * frame_size)
        bottom = top + ((ROI_BOTTOM - ROI_TOP) * frame_size)

        cropped_mouth = warped_face[int(top):int(bottom), int(left):int(right)]
        if cropped_mouth.shape[:2] != (reshape_height, reshape_width):
            cropped_mouth = cv2.resize(cropped_mouth, (reshape_width, reshape_height))

        mouth_frames.append(cropped_mouth)

    return mouth_frames


def smooth_landmarks(detections, window_length=WINDOW_LENGTH):
    # averaging the location of each keypoint with the same keypoint in n frames either side of it
    # n = half_window_length
    detections_new = {}
    half_window_length = window_length // 2

    for i in range(len(detections)):
        detections_new[i] = {'c': detections[i]['c']}

        start = i - half_window_length
        end = i + half_window_length + 1

        if start < 0:
            start = 0
        if end > len(detections) - 1:
            end = len(detections) - 1

        avg_landmarks = []
        for k in range(68):
            points = [detections[j]['l'][k] for j in range(start, end)]
            av_point = [sum(x) / len(x) for x in zip(*points)]
            avg_landmarks.append(av_point)
        detections_new[i]['l'] = np.asarray(avg_landmarks)

    assert len(detections_new) == len(detections)

    return detections_new


def get_mouth_frames_wrapper(frames, use_old_method=False, use_perspective_warp=False, face_stats=None, **kwargs):
    if use_old_method:
        return get_mouth_frames_old(frames, method='dlib')
    elif use_perspective_warp:
        return get_mouth_frames_perspective_warp(frames, face_stats=face_stats, **kwargs)
    else:
        return get_mouth_frames(frames, face_stats=face_stats)


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
