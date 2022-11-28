import sys
from pathlib import Path

import numpy as np

sys.path.append('/shared/Repos/Real-Time-Voice-Cloning')
sys.path.append('/shared/Repos/visual-dtw/app')

import encoder.inference as video_encoder
from encoder_preprocess_2 import get_cfe_features
video_encoder.load_model(Path('/shared/Repos/Real-Time-Voice-Cloning/encoder/saved_models/voxceleb_all.pt'))


def get_lip_movement_embedding(ark_matrix):
    ark_matrix = np.expand_dims(ark_matrix, axis=0)
    lip_movement_embeddings = video_encoder.embed_frames_batch(ark_matrix)

    return lip_movement_embeddings[0]
