import json
import random
import shutil
from pathlib import Path

import requests
from tqdm import tqdm


def main():
    # parent_directory = '/media/alex/Storage/Domhnall/datasets/gridcorpus/video/s1'
    parent_directory = '/media/alex/Storage/Domhnall/datasets/SRAVI/local/SRAVI_Extended/richard@liopa.ai'

    # extension = '.mpg'
    extension = '.mp4'

    offsets = []
    num_video_paths = 20

    video_paths = list(Path(parent_directory).glob(f'*{extension}'))
    random.shuffle(video_paths)

    for video_path in tqdm(video_paths[:num_video_paths]):

        # face detection
        with open(video_path, 'rb') as f:
            response = requests.post('http://0.0.0.0:8083/detect/', files={'video': f.read()})
        assert response.status_code == 200, response.content

        track = response.json()
        try:
            track = [{'x1': x['0'][0], 'y1': x['0'][1], 'x2': x['0'][2], 'y2': x['0'][3]} for x in track.values()]
        except KeyError:
            continue

        # syncnet
        with open(video_path, 'rb') as f:
            response = requests.post('http://0.0.0.0:8084/synchronise/', files={'video': f.read()},
                                     data={'track': json.dumps(track)})
        assert response.status_code == 200, response.content

        results = response.json()
        if results['confidence'] >= 2:
            offsets.append(results['offset'])

        # download crop
        response = requests.get('http://0.0.0.0:8084/crop/')
        assert response.status_code == 200, response.content
        with open('syncnet_video_after.avi', 'wb') as f:
            f.write(response.content)

        shutil.copyfile(video_path, 'syncnet_video_before.mp4')

    print('Mean offset:', sum(offsets) / len(offsets))
    print('%:', len(offsets) / num_video_paths)


if __name__ == '__main__':
    main()
