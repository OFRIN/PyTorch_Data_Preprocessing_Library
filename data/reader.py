import copy
import glob

import numpy as np
import multiprocessing as mp

from .loader import Loader
from .decoder import Decoder

from .utils import decode_image
from .utils import load_pickle

class Reader_For_Expert:
    def __init__(self, 
        pattern,
        batch_size,
        transform,
        the_size_of_queue_for_loaders, the_size_of_queue_for_decoders, 
        the_number_of_loader, the_number_of_decoder, 
        decode_fn
        ):

        self.batch_size = batch_size
        self.transform = transform
        self.decode_fn = decode_fn

        self.queue_for_decoders = mp.Queue(maxsize=the_size_of_queue_for_decoders)
        self.queue_for_loaders = mp.Queue(maxsize=the_size_of_queue_for_loaders)

        self.file_paths = glob.glob(pattern)
        self.the_number_of_loading_file = 5
        
        self.loaders = [self.make_loader() for _ in range(the_number_of_loader)]
        self.decoders = [self.make_decoder() for _ in range(the_number_of_decoder)]
    
    def make_loader(self):
        return Loader(
            queue=self.queue_for_loaders,
            file_paths=self.file_paths,
            the_number_of_loading_file=self.the_number_of_loading_file
        )
    
    def make_decoder(self):
        return Decoder(
            queue=self.queue_for_decoders,
            queue_of_loader=self.queue_for_loaders,
            batch_size=self.batch_size,
            transform=self.transform,
            decode_fn=self.decode_fn
        )

    def start(self):
        for obj in (self.loaders + self.decoders):
            obj.start()

    def close(self):
        for obj in (self.loaders + self.decoders):
            obj.close()

    def __iter__(self):
        return self

    def __next__(self):
        return self.queue_for_decoders.get()

class Reader_For_Beginner(mp.Process):
    def __init__(self, 
        pattern, 
        batch_size,
        transform,
        decode_fn
        ):
        super().__init__()

        self.file_path = glob.glob(pattern)
        self.batch_size = batch_size
        self.transform = transform
        self.decode_fn = decode_fn

        self.queue = mp.Queue(maxsize=5)

        self.init()

    def init(self):
        self.batch_dataset = []

    def run(self):
        for path in self.file_path:
            data_dic = load_pickle(path)
            for key in data_dic.keys():
                values = self.decode_fn(data_dic[key], self.transform)
                if len(self.batch_dataset) == 0:
                    self.batch_dataset = [[] for i in range(len(values))]

                for dataset, value in zip(self.batch_dataset, values):
                    dataset.append(value)

                if len(self.batch_dataset[0]) == self.batch_size:
                    self.queue.put(copy.deepcopy(self.batch_dataset))
                    self.init()

        self.queue.put(StopIteration)

    def __iter__(self):
        self.start()
        return self

    def __next__(self):
        data = self.queue.get()
        if data == StopIteration:
            self.close()
            raise StopIteration
        
        return data

    def close(self):
        self.terminate()
        self.join()