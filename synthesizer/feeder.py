import random

from sklearn.model_selection import train_test_split
from synthesizer.utils.text import text_to_sequence
from synthesizer.infolog import log
import tensorflow as tf
import numpy as np
import threading
import time
import os
from os.path import dirname, join, basename, isfile
import cv2
import glob
import time

from sample_pool import SamplePool

_batches_per_group = 4
test_feeding_status, load_next_test_sample = False, False
num_queued_batches = 0
selected_window_counts = {}
speaker_ids = {}


class Feeder:
    """
    Feeds batches of data into queue on a background thread.
    """

    def __init__(self, coordinator, hparams, num_test_batches, apply_augmentation=False,
                 training_sample_pool_location='/tmp/training_sample_pool',
                 val_sample_pool_location='/tmp/val_sample_pool', 
                 use_selection_weights=False):
        super(Feeder, self).__init__()
        self._coord = coordinator
        self._hparams = hparams
        self._train_offset = 0
        self._test_offset = 0
        self.num_test_batches = num_test_batches

        # self.filelist = {'train': self._hparams.all_images,
        #                  'test': self._hparams.all_test_images}

        self.training_sample_pool_location = training_sample_pool_location
        self.val_sample_pool_location = val_sample_pool_location
        self.use_selection_weights = use_selection_weights

        self.apply_augmentation = apply_augmentation

        # pad input sequences with the <pad_token> 0 ( _ )
        self._pad = 0
        # explicitely setting the padding to a value that doesn"t originally exist in the spectogram
        # to avoid any possible conflicts, without affecting the output range of the model too much
        if hparams.symmetric_mels:
            self._target_pad = -hparams.max_abs_value
        else:
            self._target_pad = 0.
        # Mark finished sequences with 1s
        self._token_pad = 1.

        with tf.device("/cpu:0"):
            # Create placeholders for inputs and targets. Don"t specify batch size because we want
            # to be able to feed different batch sizes at eval time.
            self._placeholders = [
                tf.placeholder(tf.float32, shape=(None, hparams.T, hparams.img_height,
                                                  hparams.img_width, hparams.num_channels), name='inputs'),
                tf.placeholder(tf.int32, shape=(None,), name="input_lengths"),
                tf.placeholder(tf.float32, shape=(None, hparams.mel_step_size, hparams.num_mels), name="mel_targets"),
                #tf.placeholder(tf.float32, shape=(None, None), name="token_targets"),
                tf.placeholder(tf.int32, shape=(None,), name="targets_lengths"),
                tf.placeholder(tf.int32, shape=(hparams.tacotron_num_gpus, None), name="split_infos"),
                tf.placeholder(tf.float32, shape=(None, 256), name="speaker_embeddings"),
                tf.placeholder(tf.int32, shape=(None,), name='speaker_targets')
            ]

            # Create queue for buffering data
            # queue = tf.FIFOQueue(8, [tf.float32, tf.int32, tf.float32, tf.float32,
            #						 tf.int32, tf.int32, tf.float32], name="input_queue")
            queue = tf.FIFOQueue(8, [tf.float32, tf.int32, tf.float32, tf.int32, tf.int32, tf.float32, tf.int32],
                                 name="input_queue")
            self._enqueue_op = queue.enqueue(self._placeholders)
            #self.inputs, self.input_lengths, self.mel_targets, self.token_targets, \
            #	self.targets_lengths, self.split_infos, self.speaker_embeddings = queue.dequeue()
            self.inputs, self.input_lengths, self.mel_targets, self.targets_lengths, self.split_infos, \
                self.speaker_embeddings, self.speaker_targets = queue.dequeue()

            self.inputs.set_shape(self._placeholders[0].shape)
            self.input_lengths.set_shape(self._placeholders[1].shape)
            self.mel_targets.set_shape(self._placeholders[2].shape)
            # self.token_targets.set_shape(self._placeholders[3].shape)
            self.targets_lengths.set_shape(self._placeholders[3].shape)
            self.split_infos.set_shape(self._placeholders[4].shape)
            self.speaker_embeddings.set_shape(self._placeholders[5].shape)
            self.speaker_targets.set_shape(self._placeholders[6].shape)

            # Create eval queue for buffering eval data
            # eval_queue = tf.FIFOQueue(1, [tf.float32, tf.int32, tf.float32, tf.float32,
            #							  tf.int32, tf.int32, tf.float32], name="eval_queue")
            eval_queue = tf.FIFOQueue(1, [tf.float32, tf.int32, tf.float32,
                                          tf.int32, tf.int32, tf.float32], name="eval_queue")
            self._eval_enqueue_op = eval_queue.enqueue(self._placeholders)
            #self.eval_inputs, self.eval_input_lengths, self.eval_mel_targets, \
            #	self.eval_token_targets, self.eval_targets_lengths, \
            #	self.eval_split_infos, self.eval_speaker_embeddings = eval_queue.dequeue()

            self.eval_inputs, self.eval_input_lengths, self.eval_mel_targets, \
                self.eval_targets_lengths, \
                self.eval_split_infos, self.eval_speaker_embeddings = eval_queue.dequeue()

            self.eval_inputs.set_shape(self._placeholders[0].shape)
            self.eval_input_lengths.set_shape(self._placeholders[1].shape)
            self.eval_mel_targets.set_shape(self._placeholders[2].shape)
            # self.eval_token_targets.set_shape(self._placeholders[3].shape)
            self.eval_targets_lengths.set_shape(self._placeholders[3].shape)
            self.eval_split_infos.set_shape(self._placeholders[4].shape)
            self.eval_speaker_embeddings.set_shape(self._placeholders[5].shape)

    def start_threads(self, session):
        self._session = session
        thread = threading.Thread(name="background", target=self._enqueue_next_train_group)
        thread.daemon = True  # Thread will close when parent quits
        thread.start()

        test_thread = threading.Thread(name="background_test", target=self._enqueue_next_test_group)
        test_thread.daemon = True  # Thread will close when parent quits
        test_thread.start()

    def get_selection_weights(self, split):
        # get selection weights of images
        image_paths = self.filelist[split]
        speaker_counts = {}
        for image_path in image_paths:
            speaker_id = image_path.split('/')[-3]
            speaker_counts[speaker_id] = speaker_counts.get(speaker_id, 0) + 1

        speaker_weight = 1. / len(speaker_counts)

        selection_weights = []
        for image_path in image_paths:
            speaker_id = image_path.split('/')[-3]
            weight = speaker_weight / speaker_counts[speaker_id]
            selection_weights.append(weight)
        assert len(selection_weights) == len(image_paths)

        log(f'{split} speaker image counts: {speaker_counts}\n'
            f'speaker weight: {speaker_weight}\n')

        return selection_weights

    def get_image_paths(self, split, num_samples, selection_weights=None):
        if self.use_selection_weights and not selection_weights:
            selection_weights = self.get_selection_weights(split=split)

        image_paths = []
        while len(image_paths) != num_samples:
            image_path = self.get_random_image_paths(split=split, k=1,
                                                     selection_weights=selection_weights)[0]
            if not self.get_item(image_path, validate_only=True):
                continue
            image_paths.append(image_path)

        return image_paths

    def make_test_batches(self, batch_size, num_batches):
        start = time.time()

        image_paths = self.get_image_paths(split='test', num_samples=batch_size * num_batches)

        if self.use_selection_weights:
            # check selected sample counts
            speaker_counts = {}
            for image_path in image_paths:
                speaker_id = image_path.split('/')[-3]
                speaker_counts[speaker_id] = speaker_counts.get(speaker_id, 0) + 1
            log(f'Validation speaker counts: {speaker_counts}')

        batches = [image_paths[i: i+batch_size]
                   for i in range(0, len(image_paths), batch_size)]

        log('\nGenerated %d test batches of size %d in %.3f sec' %
            (len(batches), batch_size, time.time() - start))

        return np.asarray(batches)

    def _enqueue_next_train_group(self):
        global num_queued_batches
        num_queued_batches = 0
        num_batches = _batches_per_group
        batch_size = self._hparams.tacotron_batch_size
        r = self._hparams.outputs_per_step
        sample_pool = SamplePool(location=self.training_sample_pool_location)

        while not self._coord.should_stop():
            while num_queued_batches != num_batches:
                start = time.time()
                batch = sample_pool.read(count=batch_size, use_selection_weights=self.use_selection_weights)
                if not batch:
                    time.sleep(0.1)
                    continue
                # bucket samples based on similar output sequence length for efficiency
                batch.sort(key=lambda x: x[-1])

                feed_dict = dict(zip(self._placeholders, self._prepare_batch(batch, r, is_training=True)))
                self._session.run(self._enqueue_op, feed_dict=feed_dict)
                num_queued_batches += 1

                log(f'Generated 1 train batch of size {len(batch)} in {time.time() - start:.3f} sec')
                log(f'Num queued batches: {num_queued_batches}')
            time.sleep(0.1)

    def _enqueue_next_test_group(self):
        global test_feeding_status, load_next_test_sample
        batch_size = self._hparams.tacotron_batch_size
        r = self._hparams.outputs_per_step
        sample_pool = SamplePool(location=self.val_sample_pool_location)

        while not self._coord.should_stop():
            if test_feeding_status:  # begin testing time
                i = 0
                while i < self.num_test_batches:
                    if load_next_test_sample:
                        batch = sample_pool.read(count=batch_size, use_selection_weights=self.use_selection_weights)
                        if not batch:
                            time.sleep(0.1)
                            continue
                        np.random.shuffle(batch)  # shuffle the batch to get different saved output
                        feed_dict = dict(zip(self._placeholders, self._prepare_batch(batch, r)))
                        self._session.run(self._eval_enqueue_op, feed_dict=feed_dict)
                        i += 1
                        load_next_test_sample = False
                    time.sleep(0.1)
                log(f'Queued Test batches: {i}')
                test_feeding_status = False
            time.sleep(1)

    def dequeue_training_sample(self):
        global num_queued_batches
        num_queued_batches -= 1

    def set_test_feeding_status(self, feeding_status):
        global test_feeding_status
        test_feeding_status = feeding_status

    def get_test_feeding_status(self):
        global test_feeding_status
        return test_feeding_status

    def set_load_next_test_sample(self, b):
        global load_next_test_sample
        load_next_test_sample = b

    def get_random_image_paths(self, split, k, selection_weights=None):
        return random.choices(self.filelist[split], k=k, weights=selection_weights)  # faster than np.random.choice

    def get_item(self, img_name, split='train', validate_only=False):
        if not os.path.isfile(os.path.join(os.path.dirname(img_name), 'mels.npz')):
            return False

        if not os.path.isfile(os.path.join(os.path.dirname(img_name), 'ref.npz')):
            return False

        window_fnames, start_frame_id = self.get_window(img_name)
        if window_fnames is None:
            return False

        # tally start frame ids of window for analysis
        counts = selected_window_counts.get(split, {})
        counts[start_frame_id] = counts.get(start_frame_id, 0) + 1
        selected_window_counts[split] = counts

        mel = np.load(os.path.join(os.path.dirname(img_name), 'mels.npz'))['spec'].T
        mel = self.crop_audio_window(mel, start_frame_id)
        if mel.shape[0] != self._hparams.mel_step_size:  # 80
            return False

        if validate_only:
            return True

        apply_augmentation = self.apply_augmentation and \
                             np.random.rand() <= self._hparams.augmentation_prob and \
                             split == 'train'

        window = []
        for fname in window_fnames:
            img = cv2.imread(fname)

            # resize the image
            if img.shape[:2] != (self._hparams.img_height, self._hparams.img_width):
                try:
                    img = cv2.resize(img, (self._hparams.img_height, self._hparams.img_width))
                except:
                    continue

            # apply augmentation if applicable
            if apply_augmentation:
                img = img[..., ::-1, :]  # mirror image along vertical axis (reverse columns)

            window.append(img)
        x = np.asarray(window) / 255.

        # speaker embedding
        refs = np.load(os.path.join(os.path.dirname(img_name), 'ref.npz'))['ref']
        # can have multiple refs per speaker
        ref = refs[np.random.choice(len(refs))]

        return x.astype(np.float32), mel.astype(np.float32), ref.astype(np.float32), len(mel)

    def _prepare_batch(self, batches, outputs_per_step, is_training=False):
        assert 0 == len(batches) % self._hparams.tacotron_num_gpus
        size_per_device = int(len(batches) / self._hparams.tacotron_num_gpus)
        np.random.shuffle(batches)

        inputs = None
        mel_targets = None
        #token_targets = None
        targets_lengths = None
        split_infos = []
        speaker_targets = []

        targets_lengths = np.asarray([x[-1] for x in batches], dtype=np.int32)  # Used to mask loss
        input_lengths = np.asarray([len(x[1]) for x in batches], dtype=np.int32)

        for i in range(self._hparams.tacotron_num_gpus):
            batch = batches[size_per_device*i:size_per_device*(i+1)]
            input_cur_device, input_max_len = self._prepare_inputs([x[1] for x in batch])
            inputs = np.concatenate((inputs, input_cur_device), axis=1) if inputs is not None else input_cur_device
            mel_target_cur_device, mel_target_max_len = self._prepare_targets([x[2] for x in batch], outputs_per_step)
            mel_targets = np.concatenate((mel_targets, mel_target_cur_device), axis=1) \
                if mel_targets is not None else mel_target_cur_device

            # Pad sequences with 1 to infer that the sequence is done
            #token_target_cur_device, token_target_max_len = self._prepare_token_targets([x[2] for x in batch], outputs_per_step)
            #token_targets = np.concatenate((token_targets, token_target_cur_device),axis=1) if token_targets is not None else token_target_cur_device
            split_infos.append([input_max_len, mel_target_max_len])

        split_infos = np.asarray(split_infos, dtype=np.int32)

        embed_targets = np.asarray([x[3] for x in batches])

        # grab the speaker targets
        if is_training and self._hparams.speaker_disentanglement:
            for video_path in [x[0] for x in batches]:
                speaker_id = video_path.split('/')[-2]
                if speaker_id in speaker_ids:
                    int_id = speaker_ids[speaker_id]
                else:
                    try:
                        int_id = max(speaker_ids.values()) + 1  # next int ID
                    except ValueError:
                        int_id = 0
                    speaker_ids[speaker_id] = int_id
                speaker_targets.append(int_id)
        speaker_targets = np.asarray(speaker_targets)

        # if using greyscale images, add a new axis to the inputs
        if self._hparams.num_channels == 1:
            inputs = inputs[..., np.newaxis]

        # return inputs, input_lengths, mel_targets, token_targets, targets_lengths, \
        #	   split_infos, embed_targets
        return inputs, input_lengths, mel_targets, targets_lengths, split_infos, embed_targets, speaker_targets

    def _prepare_inputs(self, inputs):
        max_len = max([len(x) for x in inputs])
        return np.stack([self._pad_input(x, max_len) for x in inputs]), max_len

    def _prepare_targets(self, targets, alignment):
        max_len = max([len(t) for t in targets])
        data_len = self._round_up(max_len, alignment)
        return np.stack([self._pad_target(t, data_len) for t in targets]), data_len

    def _pad_input(self, x, length):
        return np.pad(x, (0, length - x.shape[0]), mode="constant", constant_values=self._pad)

    def _pad_target(self, t, length):
        return np.pad(t, [(0, length - t.shape[0]), (0, 0)], mode="constant", constant_values=self._target_pad)

    def _pad_token_target(self, t, length):
        return np.pad(t, (0, length - t.shape[0]), mode="constant", constant_values=self._token_pad)

    def _round_up(self, x, multiple):
        remainder = x % multiple
        return x if remainder == 0 else x + multiple - remainder

    def _round_down(self, x, multiple):
        remainder = x % multiple
        return x if remainder == 0 else x - remainder

    ####### MY FUNCTIONS##################

    def get_frame_id(self, frame):
        return int(basename(frame).split('.')[0])

    # def get_window(self, center_frame):
    #     # function selects a window of 25 frames
    #     # makes sure that selected window is within the bounds of the video
    #     # has functionality to select windows at the start and the end of the video
    #     # which is important for non-speech silence generation i.e. not lrw
    #
    #     center_id = self.get_frame_id(center_frame)
    #     vidname = dirname(center_frame)
    #
    #     if self.lrw:
    #         if self._hparams.T % 2:
    #             window_ids = range(center_id - self._hparams.T//2, center_id + self._hparams.T//2 + 1)
    #         else:
    #             window_ids = range(center_id - self._hparams.T//2, center_id + self._hparams.T//2)
    #
    #         window_fnames = []
    #         for frame_id in window_ids:
    #             frame = join(vidname, '{}.jpg'.format(frame_id))
    #             if not isfile(frame):
    #                 return None, None
    #             window_fnames.append(frame)
    #
    #         start = center_id - self._hparams.T//2
    #
    #         return window_fnames, start
    #     else:
    #         half_num_timesteps = self._hparams.T // 2
    #         num_video_frames = len(glob.glob(join(vidname, '*.jpg')))
    #
    #         if center_id < half_num_timesteps:
    #             start = 0
    #             end = self._hparams.T
    #         elif center_id > (num_video_frames - half_num_timesteps):
    #             start = (num_video_frames - self._hparams.T) + 1
    #             end = num_video_frames + 1
    #         else:
    #             start = center_id - half_num_timesteps
    #             end = center_id + half_num_timesteps + 1
    #
    #         attempts = 2
    #         while attempts != 0:
    #             window_ids = range(start, end)
    #             assert len(window_ids) == 25
    #             window_fnames = [join(vidname, f'{frame_id}.jpg') for frame_id in window_ids]
    #             if not all([isfile(fname) for fname in window_fnames]):
    #                 start -= 1
    #                 end -= 1
    #                 attempts -= 1
    #                 continue
    #
    #             return window_fnames, start
    #
    #         return None, None

    def crop_audio_window(self, spec, start_frame_id):
        # estimate total number of frames from spec (num_features, T)
        # num_frames = (T x hop_size * fps) / sample_rate

        total_num_frames = int((spec.shape[0] * self._hparams.hop_size * self._hparams.fps) / self._hparams.sample_rate)

        start_idx = int(spec.shape[0] * start_frame_id / float(total_num_frames))
        end_idx = start_idx + self._hparams.mel_step_size

        return spec[start_idx: end_idx, :]
