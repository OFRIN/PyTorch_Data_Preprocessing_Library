# Copyright (C) 2020 * Ltd. All rights reserved.
# author : Sanghyeon Jo <josanghyeokn@gmail.com>

import glob
import torch

import numpy as np

from torch.utils.data import Dataset, DataLoader

from .utils import *

class SH_Dataset(Dataset):
    def __init__(self, data_pattern, transform):
        self.data_paths = glob.glob(data_pattern)
        self.data_dic = {path : open(path, 'rb') for path in self.data_paths}
        
        self.dataset = []
        self.transform = transform

        for path in self.data_paths:
            for string in open(path.replace('.sang', '.index'), 'r').readlines():
                start_point, length_of_example = string.strip().split(',')
                self.dataset.append((path, int(start_point), int(length_of_example)))

    def __len__(self):
        return len(self.dataset)

    def decode(self, example):
        image = decode_image(example['encoded_image'])
        if self.transform is not None:
            image = self.transform(image)

        label = example['label']
        return image, label

    def get_example(self, data):
        path, start_point, length_of_example = data

        self.data_dic[path].seek(start_point - self.data_dic[path].tell(), 1)
        bytes_of_example = self.data_dic[path].read(length_of_example)

        return deserialize(bytes_of_example)
    
    def __getitem__(self, index):
        example = self.get_example(self.dataset[index])
        return self.decode(example)

