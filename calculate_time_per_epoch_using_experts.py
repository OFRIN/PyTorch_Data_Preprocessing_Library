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

from core.networks import *
from core.data_utils import *
from core.utils import *

from utility.utils import *
from utility.json_utils import *

from data import reader
from data import utils

parser = argparse.ArgumentParser()

###############################################################################
# Dataset
###############################################################################
parser.add_argument('--seed', default=0, type=int)
parser.add_argument('--data_dir', default='C:/Classification_DB_sang/', type=str)
parser.add_argument('--use_cores', default=1, type=int)

###############################################################################
# Network
###############################################################################
parser.add_argument('--dataset_names', default='Cars', type=str)

args = parser.parse_args()

def customized_decode_fn(example, transform):
    image = utils.decode_image(example['encoded_image'])
    
    if transform is not None:
        image = transform(image)

    return image, example['label']

if __name__ == '__main__':
    # 1. Config
    set_seed(args.seed)

    # 2. Dataset
    base_augmentations = [
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(0.5),
        transforms.ColorJitter(brightness = 0.3, contrast = 0.3, saturation = 0.3, hue = 0.1),
    ]

    base_transforms = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]
    )
    
    train_transform = transforms.Compose(
        [convert_OpenCV_to_PIL] + base_augmentations
    )
    
    dataset_name = args.dataset_names.split(',')[0]
    sang_dir = args.data_dir + f'{dataset_name}/'
    
    sample_reader = reader.Reader_For_Expert(
        pattern=sang_dir + f'train/*.sang',
        batch_size=64, transform=train_transform,
        the_size_of_queue_for_loaders=5, the_size_of_queue_for_decoders=100,
        the_number_of_loader=args.use_cores, the_number_of_decoder=args.use_cores*4,
        decode_fn=customized_decode_fn
    )

    timer = Timer()
    
    for i in range(50):
        images, labels = next(sample_reader)

        images = torch.stack([base_transforms(image) for image in images])
        labels = torch.tensor(labels, dtype=torch.long)

        # print(images.size(), labels.size())

    print('[{}] 1 epoch : {}sec'.format(dataset_name, timer.tok()))

