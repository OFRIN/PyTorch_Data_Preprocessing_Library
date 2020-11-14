import copy

import numpy as np
import multiprocessing as mp

from .utils import load_pickle

class Loader(mp.Process):
    def __init__(self, **kwargs):
        super().__init__()

        self.queue = kwargs['queue']
        self.file_paths = kwargs['file_paths']
        self.the_number_of_loading_file = min(kwargs['the_number_of_loading_file'], len(self.file_paths))
    
    def dataset_generator(self):
        np.random.shuffle(self.file_paths)

        dataset = copy.deepcopy(self.file_paths)
        while len(dataset) > self.the_number_of_loading_file:
            data_dic = {}
            for path in dataset[:self.the_number_of_loading_file]:
                data_dic.update(load_pickle(path))
            dataset = dataset[self.the_number_of_loading_file:]
            
            examples = [data_dic[key] for key in list(data_dic.keys())]
            np.random.shuffle(examples)

            yield examples
    
    def run(self):
        while True:
            for examples in self.dataset_generator():
                self.queue.put(examples)
    
    def get(self):
        return self.queue.get()

    def close(self):
        self.terminate()
        self.join()