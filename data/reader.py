# Copyright (C) 2020 * Ltd. All rights reserved.
# author : Sanghyeon Jo <josanghyeokn@gmail.com>

import ray
import copy
import glob
import torch
import threading

import numpy as np

from queue import Queue
from core.decode import decode_fn

from .utils import load_pickle

class SH_Reader(threading.Thread):
    def __init__(self, 
        pattern, 
        transform,
        training,
        batch_size,
        output_names
        ):
        super().__init__()

        self.paths = glob.glob(pattern)
        self.transform = transform

        self.training = training
        self.batch_size = batch_size
        self.output_names = output_names

        self.queue = Queue(50)
        self.the_number_of_loading_file = 5

        self.progress = True

        self.init()
    
    def init(self):
        self.batch_dataset = {name : [] for name in self.output_names}

    def generator(self):
        if self.training:
            np.random.shuffle(self.paths)

        dataset = copy.deepcopy(self.paths)
        while len(dataset) > self.the_number_of_loading_file:
            yield dataset[:self.the_number_of_loading_file]
            dataset = dataset[self.the_number_of_loading_file:]
        yield dataset

    def merge(self, data_list):
        dataset = []
        for data in data_list:
            dataset += data
        return dataset

    def put(self):
        self.queue.put([self.batch_dataset[name] for name in self.output_names])
        self.init()

    def get_length(self):
        return len(self.batch_dataset[self.output_names[0]])

    def update(self, results):
        for result in results:
            for name in self.output_names:
                self.batch_dataset[name].append(result[name])

            if self.get_length() == self.batch_size:
                self.put()        

    def run(self):
        while self.progress:
            for paths in self.generator():
                examples_list = [load_pickle(path) for path in paths]
                results = ray.get([decode_fn.remote(example, self.transform) for example in self.merge(examples_list)])
                
                if self.training:
                    np.random.shuffle(results)

                self.update(results)

            if self.get_length() > 0 and not self.training:
                self.put()

            if not self.training:
                self.progress = False
            
        self.queue.put(StopIteration)

    def __iter__(self):
        return self

    def __next__(self):
        data = self.queue.get()
        if data == StopIteration:
            raise StopIteration
        return data
        