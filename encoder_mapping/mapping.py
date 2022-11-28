"""Learn a mapping between a face and audio encoder"""
import argparse
import collections
import datetime
import hashlib
import logging
import os
import random
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torchaudio
import torchaudio.transforms as AT
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from facenet_pytorch import InceptionResnetV1
from torch import optim as Optimzer
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from detectors import dlib_face_detector, dlib_shape_predictor
from video_utils import extract_audio, get_num_frames, get_video_frame

CURRENT_DIR = os.path.dirname(__file__)
ROI = {
    'nose': slice(27, 31),
    'nose_point': slice(30, 31),
    'nostril': slice(31, 36),
    'eye1': slice(36, 42),
    'eye2': slice(42, 48)
}


def get_roi_mid_point(landmarks, roi):
    x, y, w, h = cv2.boundingRect(landmarks[roi])
    mid_x = x + w // 2
    mid_y = y + h // 2

    return mid_x, mid_y


def align_face(frame, face_coords, landmarks):
    landmarks = np.array(landmarks)

    left_eye = get_roi_mid_point(landmarks, ROI['eye1'])
    right_eye = get_roi_mid_point(landmarks, ROI['eye2'])

    left_eye_x, left_eye_y = left_eye
    right_eye_x, right_eye_y = right_eye

    delta_x = right_eye_x - left_eye_x
    delta_y = right_eye_y - left_eye_y

    try:
        angle = np.arctan(delta_y/delta_x)
        angle = (angle * 180) / np.pi
    except ZeroDivisionError:
        angle = 0

    x1, y1, x2, y2 = face_coords
    center_pred = x1 + ((x2 - x1) // 2), y1 + ((y2 - y1) // 2)
    nose_point = landmarks[ROI['nose_point']][0]

    if abs(nose_point[0] - center_pred[0]) > 20:
        return None

    img = frame[y1: y2, x1: x2].numpy()

    h, w = img.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, (angle), 1.0)
    rotated = cv2.warpAffine(img, M, (w, h))

    return rotated


def align_and_crop_face(frame, face_coords, landmarks):
    face = align_face(frame, face_coords, landmarks)
    if face is None:
        return face

    return torch.from_numpy(face).permute(2, 0, 1)


def save_figure_to_numpy(fig):
    # save it to a numpy array.
    data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
    data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))

    return data


def plot_alignment_to_numpy(alignment, info=None):
    fig, ax = plt.subplots(figsize=(6, 4))
    im = ax.imshow(alignment, aspect='auto', origin='lower', interpolation='none')
    fig.colorbar(im, ax=ax)
    xlabel = 'Decoder timestep'
    if info is not None:
        xlabel += '\n\n' + info
    plt.xlabel(xlabel)
    plt.ylabel('Encoder timestep')
    plt.tight_layout()

    fig.canvas.draw()
    data = save_figure_to_numpy(fig)
    plt.close()

    return data


def av_audio_face_collate_fn(batch):
    audios, faces = zip(*batch)
    min_samples_in_batch = min([s.shape[1] for s in audios])
    trimmed_audios = torch.zeros(len(audios), min_samples_in_batch)

    for idx, audio in enumerate(audios):
        S = min_samples_in_batch
        trimmed_audios[idx, :S] = audio[:, :S]

    faces_tensor = torch.cat([f.unsqueeze(0) for f in faces], dim=0)

    return trimmed_audios, faces_tensor


class TFBoard(SummaryWriter):

    def __init__(self, log_dir):
        super(TFBoard, self).__init__(log_dir)

    def log_training(self, reduced_loss, grad_norm, learning_rate, duration, iteration):
        self.add_scalar('training.loss', reduced_loss, iteration)
        self.add_scalar('grad.norm', grad_norm, iteration)
        self.add_scalar('learning.rate', learning_rate, iteration)
        self.add_scalar('duration', duration, iteration)

    def log_similarity_matrix(self, alignments, iteration):
        index = random.randint(0, alignments.size(0) - 1)
        self.add_image('sim_matrix', plot_alignment_to_numpy(alignments[index].data.cpu().numpy().T),
                       iteration, dataformats='HWC')


class Logger:

    def __init__(self, model_info):
        self.logger = logging.getLogger()
        self.model_save_path = os.path.join(CURRENT_DIR, 'saved_models', hashlib.md5(model_info.encode()).hexdigest())
        self.tensorboard_save_path = os.path.join(self.model_save_path, 'tf-logs')

        if not os.path.exists(self.model_save_path):
            os.makedirs(self.model_save_path)
            os.makedirs(self.tensorboard_save_path)

        self.setup_logger()
        self.logger.info(model_info)

        self.tensorboard = TFBoard(self.tensorboard_save_path)

    def setup_logger(self):
        log_file = os.path.join(self.model_save_path, f'model-log-{time.strftime("%d-%m-%Y")}.log')
        log_level = logging.INFO
        log_format = '%(levelname)s %(filename)s(%(lineno)d): %(message)s'
        logging.basicConfig(level=log_level, format=log_format, filename=log_file)
        logging.root.addHandler(logging.StreamHandler())


class FaceRecogniser(nn.Module):

    def __init__(self, device):
        super().__init__()
        self.resnet = InceptionResnetV1(pretrained='casia-webface').to(device)
        for p in self.resnet.parameters():
            p.requires_grad = False  # freeze params
        self.resnet.last_linear.requires_grad_(True)  # last 2 blocks are left unfrozen i.e. trainable
        self.resnet.last_bn.requires_grad_(True)

        # attach another FC layer in top
        self.projection_layer = nn.Sequential(
            nn.Linear(512, 512),
            nn.GELU(),
            nn.Linear(512, 256)
        ).to(device)

    def forward(self, x):
        # input is a single face
        x = self.resnet.conv2d_1a(x)
        x = self.resnet.conv2d_2a(x)
        x = self.resnet.conv2d_2b(x)
        x = self.resnet.maxpool_3a(x)
        x = self.resnet.conv2d_3b(x)
        x = self.resnet.conv2d_4a(x)
        x = self.resnet.conv2d_4b(x)
        x = self.resnet.repeat_1(x)
        x = self.resnet.mixed_6a(x)
        x = self.resnet.repeat_2(x)
        x = self.resnet.mixed_7a(x)
        x = self.resnet.repeat_3(x)
        x = self.resnet.block8(x)
        x = self.resnet.avgpool_1a(x)
        x = self.resnet.dropout(x)
        x = self.resnet.last_linear(x.view(x.shape[0], -1))
        embeddings_raw = self.resnet.last_bn(x)
        projection = (self.projection_layer(embeddings_raw))

        return projection


class AudioEncoder(nn.Module):

    def __init__(self, device, weights_path):
        super().__init__()

        self.lstm = nn.LSTM(input_size=40, hidden_size=256, num_layers=3, batch_first=True).to(device)
        self.linear = nn.Linear(in_features=256, out_features=256).to(device)

        state_dict = torch.load(weights_path, map_location=device)['model_state']
        self.load_state_dict(state_dict, strict=False)

        for name, p in self.named_parameters():
            p.requires_grad = False  # all params frozen i.e. not trained

        self.mel_spec = AT.MelSpectrogram(sample_rate=16000, n_fft=400, hop_length=160, n_mels=40).to(device)

    def forward(self, utterances, hidden_init=None):
        utterances = self.mel_spec(utterances).permute(0, 2, 1)  # convert to mel-spec and swap dimensions
        out, (hidden, cell) = self.lstm(utterances, hidden_init)
        embeds_raw = self.linear(hidden[-1])

        return embeds_raw


class MiniBatchContrastiveLoss(nn.Module):

    def __init__(self, device, t=1.0):
        super().__init__()
        self.t = nn.Parameter(torch.tensor(t, requires_grad=True, device=device))
        self.BCE = nn.BCEWithLogitsLoss()
        self.MSE = nn.MSELoss()
        self.log_softmax = nn.LogSoftmax(dim=-1)
        self.count = 0
        self.device = device

    def forward(self, features, losses=None):
        self.t.data = self.t.data.clamp(max=100)
        if losses is None:
            losses = dict()
        audio_embeddings, face_embeddings = features
        num_samples = face_embeddings.shape[0]

        # MSE between face and audio embeddings
        losses['l2_loss'] = self.MSE(F.normalize(F.relu(face_embeddings), dim=1),
                                     F.normalize(F.relu(audio_embeddings), dim=1))

        # verification loss between (face + audio embeddings) and targets (ids)
        # i.e. matching embeddings to actual people
        logits = face_embeddings @ audio_embeddings.T * self.t
        targets = torch.arange(0, num_samples).to(self.device)  # targets are just ids
        weight = (torch.ones(num_samples).float() * (num_samples - 1)).to(self.device)
        c_loss = (F.cross_entropy(logits, targets, weight=weight) +
                  F.cross_entropy(logits.T, targets.T, weight=weight)) / 2
        losses['c_loss'] = c_loss

        return losses


class VoxCeleb2(Dataset):

    def __init__(self, root_path):
        super().__init__()
        self.root_path = root_path

        self.fps = 25
        self.duration = 1
        self.sample_rate = 16000

        self.face_recogniser_resize = transforms.Compose([
            transforms.Resize((160, 160)),
            transforms.Lambda(lambda im: (im.float() - 127.5) / 128.0)  # normalise
        ])

        self.face_augmentation = transforms.RandomHorizontalFlip()
        self.face_detector = dlib_face_detector

        self.items = {}
        self.video_paths_list = os.path.join(CURRENT_DIR, 'video_paths.txt')
        print('Getting video paths...')
        if os.path.exists(self.video_paths_list):
            with open(self.video_paths_list, 'r') as f:
                video_paths = f.read().splitlines()
        else:
            video_paths = list(Path(self.root_path).glob('**/*.mp4'))
            with open(self.video_paths_list, 'w') as f:
                for video_path in video_paths:
                    f.write(f'{video_path}\n')
        for i, video_path in enumerate(video_paths):
            self.items[i] = str(video_path)
        random.shuffle(self.items)

        self.length = len(self.items)
        self.random_indices = np.random.choice(len(self), 2 * len(self)).tolist()
        self.index = -1

    def __len__(self):
        return self.length

    def reset_item(self):
        if self.index < 0:
            random.shuffle(self.random_indices)
            self.index = len(self.random_indices) - 1

        index = self.random_indices[self.index]
        self.index -= 1

        return self[index]

    def __getitem__(self, index):
        video_path = self.items[index]

        # select random frame
        total_frames = get_num_frames(video_path)
        end_time = total_frames / self.fps
        duration = self.duration
        if int(end_time) == 0:
            return self.reset_item()
        start_time = random.choice(np.arange(0, end_time, 0.25))
        if (start_time + duration) > end_time:
            start_time -= duration

        frame_time = start_time + np.random.uniform(0, 0.25, 1)
        absolute_frame_index = int(frame_time * self.fps)

        try:
            frame = get_video_frame(video_path, absolute_frame_index)
            frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        except cv2.error:
            return self.reset_item()

        bbs = dlib_face_detector(frame_gray, 0)
        if len(bbs) != 1:
            return self.reset_item()
        bb = bbs.pop()
        landmarks = [[p.x, p.y] for p in dlib_shape_predictor(frame_gray, bb).parts()]
        face_coords = [bb.left(), bb.top(), bb.right(), bb.bottom()]

        try:
            face = align_and_crop_face(torch.from_numpy(frame), face_coords, landmarks)
        except cv2.error:
            return self.reset_item()

        if face is None:
            return self.reset_item()

        face = self.face_augmentation(self.face_recogniser_resize(face))

        # load audio
        # TODO: this might be too slow - should extract these onto disk beforehand
        audio_file = extract_audio(video_path)
        try:
            audio, sampling_rate = torchaudio.load(audio_file.name, frame_offset=int(self.sample_rate * start_time),
                                                   num_frames=int(self.sample_rate * duration), normalize=True,
                                                   format='wav')  # 1 second of audio
            audio_file.close()
        except Exception:
            audio_file.close()
            return self.reset_item()

        assert sampling_rate == self.sample_rate

        return audio, face


def train(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    dataset = VoxCeleb2(args.dataset_root_path)

    face_encoder = FaceRecogniser(device=device)
    audio_encoder = AudioEncoder(device=device, weights_path=args.audio_encoder_weights_path)

    logger = Logger(str(face_encoder))

    contrastive_loss = MiniBatchContrastiveLoss(device=device)

    optimiser = Optimzer.SGD(face_encoder.parameters(), weight_decay=1e-5, lr=1e-3, momentum=0.9)
    t_optimiser = Optimzer.Adam([contrastive_loss.t])  # optimising loss param

    start_iteration = 0
    if args.model_save_path:
        loaded_model = torch.load(args.model_save_path, map_location=device)
        state_dict = loaded_model['state_dict']
        face_encoder.load_state_dict(state_dict, strict=False)
        start_iteration = loaded_model['start_iteration'] + 2
        optimiser.load_state_dict(loaded_model['optimize_state'])
        t_optimiser.load_state_dict(loaded_model['t']['optim'])
        contrastive_loss.t = loaded_model['t']['value']
        print(f'Model Loaded: {args.model_save_path} @ start_it: {start_iteration} t: {contrastive_loss.t}')

    for group in optimiser.param_groups:
        group['initial_lr'] = 1e-3
    lr_scheduler = Optimzer.lr_scheduler.CosineAnnealingLR(
        optimiser, (args.max_iterations * args.batch_size) // len(dataset),
        last_epoch=(start_iteration * args.batch_size) // len(dataset),
        verbose=True
    )

    data_loader = DataLoader(dataset,
                             batch_size=args.batch_size,
                             shuffle=True,
                             num_workers=args.num_workers,
                             pin_memory=True,
                             drop_last=False,
                             collate_fn=av_audio_face_collate_fn,
                             persistent_workers=True,
                             prefetch_factor=4)
    ds_iterator = iter(data_loader)

    num_epochs = 0
    loss_logs = collections.defaultdict(float)
    start_time = global_start_time = time.time()

    for iteration in range(start_iteration, args.max_iterations):
        try:
            batch = next(ds_iterator)
        except StopIteration:
            ds_iterator = iter(data_loader)
            num_epochs += 1
            lr_scheduler.step()
            batch = next(ds_iterator)

        audios, faces = batch
        audios, faces = audios.to(device), faces.to(device)

        with torch.no_grad():
            audio_embeddings = audio_encoder(audios)
        face_embeddings = face_encoder(faces)

        losses = contrastive_loss([audio_embeddings, face_embeddings])
        for k, v in losses.items():
            loss_logs[k] += v.item()
        loss = sum(losses.values())

        optimiser.zero_grad()
        t_optimiser.zero_grad()

        # backprop
        loss.backward()
        grad_norm = clip_grad_norm_(face_encoder.parameters(), 10)

        # update weights
        optimiser.step()
        t_optimiser.step()

        loss_logs['loss'] += loss.item()

        if (iteration + 1) % args.log_every == 0:
            for k, v in loss_logs.items():
                loss_logs[k] = round(v / args.log_every, 5)  # average the losses

            end_time = time.time()
            iteration_time, global_time = end_time - start_time, end_time - global_start_time
            eta = int((args.max_iterations - iteration) * (global_time / iteration))  # (no. iterations left) x (av. iteration time so far)
            eta = str(datetime.timedelta(seconds=eta))
            logs_message = ', '.join([f'{k}: {v}' for k, v in loss_logs.items()])
            message = f'Epoch: {num_epochs}, ' \
                      f'Iteration: {iteration + 1}, {(iteration + 1) / args.max_iterations}% Done, ' \
                      f'Losses: {logs_message}, ' \
                      f'ETA: {eta}, ' \
                      f'Iteration Time: {iteration_time:.2f}'

            logger.logger.info(message)
            logger.tensorboard.log_training(loss_logs['loss'], grad_norm, 1, iteration_time, iteration + 1)

            loss_logs = collections.defaultdict(float)
            start_time = end_time

        if (iteration + 1) % args.save_every == 0:
            save_name = f'{iteration + 1}_{int(time.time())}.pth'
            save_path = os.path.join(logger.model_save_path, save_name)
            torch.save({
                'start_iteration': iteration,
                'state_dict': face_encoder.state_dict(),
                'optimize_state': optimiser.state_dict(),
                # 'min_eval_loss': min_eval_loss,
                't': {'value': contrastive_loss.t, 'optim': t_optimiser.state_dict()},
            }, save_path)
            print(f'Model saved at iteration {(iteration + 1)}')

    save_path = os.path.join(logger.model_save_path, 'model_final.pth')
    face_encoder.cpu()
    torch.save({'state_dict': face_encoder.state_dict()}, save_path)
    logger.logger.info('training done, model saved to: {}'.format(save_path))


def analyse_dataset(args):
    dataset = VoxCeleb2(args.root_path)
    print('Dataset size:', dataset.length)

    av_time = 0
    for i in range(args.num_samples):
        start_time = time.time()
        audio, face = dataset[i]
        av_time += (time.time() - start_time)
        audio = audio.cpu().detach().numpy()
        face = face.permute(1, 2, 0).cpu().detach().numpy()

        if args.debug:
            print(audio.shape, face.shape)
            cv2.imshow('Face', face)
            cv2.waitKey(0)
    av_time /= args.num_samples
    print(f'Average time per sample:', av_time)


def main(args):
    f = {
        'train': train,
        'analyse_dataset': analyse_dataset
    }
    f[args.run_type](args)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    sub_parsers = parser.add_subparsers(dest='run_type')

    parser_1 = sub_parsers.add_parser('train')
    parser_1.add_argument('audio_encoder_weights_path')
    parser_1.add_argument('dataset_root_path')
    parser_1.add_argument('--max_iterations', type=int, default=1000000)
    parser_1.add_argument('--batch_size', type=int, default=32)
    parser_1.add_argument('--num_workers', type=int, default=6)
    parser_1.add_argument('--log_every', type=int, default=10)
    parser_1.add_argument('--save_every', type=int, default=1000)
    parser_1.add_argument('--model_save_path')

    parser_2 = sub_parsers.add_parser('analyse_dataset')
    parser_2.add_argument('root_path')
    parser_2.add_argument('--num_samples', type=int, default=10)
    parser_2.add_argument('--debug', action='store_true')

    main(parser.parse_args())
