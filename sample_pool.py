import os
import pickle
import random
import shutil
import uuid
from pathlib import Path

import numpy as np


class SamplePool:

    def __init__(self, location='/tmp/sample_pool', redo=False):
        self.location = Path(location)
        if self.location.exists() and redo:
            shutil.rmtree(str(self.location))
        self.location.mkdir(exist_ok=True)

    def write(self, obj, _class=None, max_size=500):
        # write sample to file, delete random sample if overloaded
        # eventually, class with the most samples in the pool will be more likely overwritten

        # check number of samples in the pool
        sample_paths = self.sample_paths()
        num_samples = len(sample_paths)

        if num_samples + 1 > max_size:
            # remove sample at random
            while True:
                random_index = random.randint(0, num_samples - 1)  # inclusive
                sample_to_remove = str(sample_paths[random_index])
                if os.path.exists(sample_to_remove):
                    os.remove(sample_to_remove)
                    break
        
        save_name = f'{str(uuid.uuid4())}.npz'
        if _class: 
            save_name = f'{_class}_' + save_name

        np.savez_compressed(str(self.location.joinpath(save_name)), sample=obj)

    def read(self, count, use_selection_weights=False):
        pool_size = self.size()
        if count > pool_size:
            return []

        while True:
            sample_path_list = self.sample_paths()
            if use_selection_weights:
                class_counts = {}
                classes = []
                for sample_path in sample_path_list:
                    _class = sample_path.name.split('_')[0]
                    class_counts[_class] = class_counts.get(_class, 0) + 1
                    classes.append(_class)
                class_weight = 1. / len(class_counts)
                selection_weights = [class_weight / class_counts[_class]
                                     for _class in classes]
                assert len(selection_weights) == len(classes) == len(sample_path_list)

                random_sample_paths = random.choices(sample_path_list, k=count, weights=selection_weights)

                # counting the number of each selected class
                selected_class_counts_path = self.location.joinpath('selected_class_counts.pkl')
                if selected_class_counts_path.exists():
                    with selected_class_counts_path.open('rb') as f:
                        selected_class_counts = pickle.load(f)
                else:
                    selected_class_counts = {}
                for sample_path in random_sample_paths:
                    _class = sample_path.name.split('_')[0]
                    selected_class_counts[_class] = selected_class_counts.get(_class, 0) + 1
                with selected_class_counts_path.open('wb') as f:
                    pickle.dump(selected_class_counts, f)
            else:
                random_sample_paths = random.sample(sample_path_list, count)

            try:
                samples = []
                for path in random_sample_paths:
                    samples.append(np.load(str(path), allow_pickle=True)['sample'])

                return samples
            except Exception as e:
                print(e)

    def sample_paths(self):
        return [f'{d}/{f}' for d, _, fs in os.walk(self.location) for f in fs if f[-4:] == '.npz']
        # return list(self.location.glob('*.npz'))

    def size(self):
        return len(self.sample_paths())
