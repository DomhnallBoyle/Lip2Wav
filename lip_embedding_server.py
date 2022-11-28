import argparse
import json
import tempfile
import sys
from pathlib import Path

import numpy as np
from flask import Flask, request

app = Flask(__name__)

sys.path.extend([
    '/shared/Repos/Real-Time-Voice-Cloning',
    '/shared/Repos/visual-dtw/app'
])

import encoder.inference as video_encoder
from encoder_preprocess_2 import get_cfe_features

video_encoder.load_model(Path('/shared/Repos/Real-Time-Voice-Cloning/encoder/saved_models/voxceleb_all.pt'),
                         device='cpu')

args = None


@app.route('/lip_embeddings', methods=['POST'])
def get_lip_embeddings():
    uploaded_file = request.files['video']

    with tempfile.NamedTemporaryFile(suffix='.mp4') as f1:
        with open(f1.name, 'wb') as f2:
            f2.write(uploaded_file.read())
        f1.seek(0)

        # NOTE: this is the DTW CFE which uses an old AE model
        # the lip movement encoder was trained using arks from this
        ark_matrix = get_cfe_features(f1.name, args.cfe_host, False)
        if ark_matrix is None:
            return '', 204

    ark_matrix = np.expand_dims(ark_matrix, axis=0)
    lip_movement_embeddings = video_encoder.embed_frames_batch(ark_matrix).tolist()

    return json.dumps(lip_movement_embeddings)


def main():
    app.run(host='0.0.0.0', port=6002)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('cfe_host')
    parser.add_argument('--cfe_verify')

    args = parser.parse_args()

    main()
