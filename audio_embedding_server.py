import argparse
import json
import tempfile

from flask import Flask, request

from audio_utils import extract_audio_embeddings

app = Flask(__name__)


@app.route('/audio_embeddings', methods=['POST'])
def get_audio_embeddings():
    uploaded_file = request.files['audio']

    with tempfile.NamedTemporaryFile(suffix='.wav') as f1:
        with open(f1.name, 'wb') as f2:
            f2.write(uploaded_file.read())
        f1.seek(0)
        speaker_embeddings = extract_audio_embeddings(f1)

    return json.dumps(speaker_embeddings)


def main(args):
    app.run(host='0.0.0.0', port=args.port)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', type=int, default=6001)

    main(parser.parse_args())

