# Copyright (C) 2020 * Ltd. All rights reserved.
# author : Sanghyeon Jo <josanghyeokn@gmail.com>

import os
import cv2
import sys
import copy
import time
import glob
import json
import random
import pickle
import argparse

import numpy as np
import multiprocessing as mp

import torch
import torch.nn as nn
import torch.nn.functional as F

import torch.backends.cudnn as cudnn

from torchvision import transforms, datasets
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from PIL import Image

from core.data_utils import *

from sanghyeon.data.writer import *
from sanghyeon.data.utils import *

from sanghyeon.utility.utils import *
from sanghyeon.utility.json_utils import *

parser = argparse.ArgumentParser()

parser.add_argument('--seed', default=0, type=int)
parser.add_argument('--data_dir', default='C:/Classification_DB_original/', type=str)
parser.add_argument('--use_cores', default=4, type=int)
parser.add_argument('--batch_size', default=64, type=int)

args = parser.parse_args()

if __name__ == '__main__':
    set_seed(args.seed)

    log_func = print
    
    # 2. Dataset
    base_augmentations = transforms.Compose([
        transforms.Resize(256, interpolation=Image.CUBIC),
        convert_PIL_to_OpenCV
    ])
    
    # 'Food-101', 'Caltech-256', 'CUB-200', 'DTD', 
    for dataset_name in ['Flowers-102', 'Pet', 'Cars', 'Dogs']:
        train_dataset, validation_dataset, test_dataset, classes = get_dataset_func(dataset_name)(args.data_dir, base_augmentations, base_augmentations)
        
        for domain, dataset in zip(['train', 'validation', 'test'], [train_dataset, validation_dataset, test_dataset]):
            writer = SH_Writer(f'./data/{dataset_name}/{domain}/', '{:05d}.sang', 100)
            for index, (image, label) in enumerate(dataset):
                label = label.item()
                if label == -1:
                    continue
                
                writer(str(index), {
                    'encoded_image' : encode_image(image),
                    'label' : label,
                })

                # print(image.shape)
                # print(encode_image(image).shape)
                # image = decode_image(encode_image(image))
                # cv2.imshow('show', image)
                # cv2.waitKey(0)

            writer.save()
        
        log_func('# Dataset ({})'.format(dataset_name))
        log_func('[i] The size of train dataset = {}'.format(len(train_dataset)))
        log_func('[i] The size of validation dataset = {}'.format(len(validation_dataset)))
        log_func('[i] The size of test dataset = {}'.format(len(test_dataset)))
        log_func()
    

