# Copyright (C) 2020 * Ltd. All rights reserved.
# author : Sanghyeon Jo <josanghyeokn@gmail.com>

import os
import cv2
import sys
import copy
import time
import glob
import json
import math
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

from sanghyeon.utility.utils import *
from sanghyeon.utility.json_utils import *

from sanghyeon.data import reader
from sanghyeon.data import utils

parser = argparse.ArgumentParser()

###############################################################################
# GPU Config
###############################################################################
parser.add_argument('--use_gpu', default='0', type=str)

###############################################################################
# Dataset
###############################################################################
parser.add_argument('--seed', default=0, type=int)
parser.add_argument('--data_dir', default='C:/Classification_DB/', type=str)
parser.add_argument('--use_cores', default=4, type=int)

###############################################################################
# Network
###############################################################################
parser.add_argument('--experiment_name', default='Wide-ResNet-28_seed@0', type=str)
parser.add_argument('--dataset_names', default='Cars', type=str)

parser.add_argument('--model_name', default='models.resnet18', type=str)

###############################################################################
# Training
###############################################################################
parser.add_argument('--optimizer', default='Adam', type=str)
parser.add_argument('--scheduler', default='step', type=str)
parser.add_argument('--learning_rate', default=1e-4, type=float)
parser.add_argument('--weight_decay', default=1e-5, type=float)

parser.add_argument('--image_size', default=224, type=int)
parser.add_argument('--batch_size', default=64, type=int)

parser.add_argument('--pretrained', default=True, type=str2bool)

parser.add_argument('--log_interval', default=100, type=int)
parser.add_argument('--val_interval', default=5000, type=int)
parser.add_argument('--max_iteration', default=100000, type=int)

args = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = args.use_gpu

def decode_fn(example, transform):
    image = utils.decode_image(example['encoded_image'])
    image = transform(image)
    return image, example['label']

if __name__ == '__main__':
    # 1. Config
    set_seed(args.seed)

    # 2. Dataset
    base_augmentations = [
        transforms.Resize(math.ceil(args.image_size * 1.14)), 
        transforms.RandomResizedCrop(args.image_size),
        transforms.RandomHorizontalFlip(0.5),
        transforms.ColorJitter(brightness = 0.3, contrast = 0.3, saturation = 0.3, hue = 0.1),
    ]

    test_augmentations = [
        transforms.Resize(math.ceil(args.image_size * 1.14)), 
        transforms.CenterCrop(args.image_size)
    ]

    base_transforms = [
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ]

    train_transform = transforms.Compose(
        [convert_OpenCV_to_PIL] + base_augmentations + base_transforms
    )
    test_transform = transforms.Compose(
        [convert_OpenCV_to_PIL] + test_augmentations + base_transforms
    )

    dataset_names = args.dataset_names.split(',')
    dataset_dic = {dataset_name : {} for dataset_name in dataset_names}
    
    train_decode_fn = lambda example: decode_fn(example, train_transform)
    test_decode_fn = lambda example: decode_fn(example, test_transform)

    class_dic = {
        'Cars' : 196
    }

    for dataset_name in dataset_names:
        sang_dir = args.data_dir + f'{dataset_name}/'
        print(sang_dir)
        
        train_reader = reader.SH_Reader(sang_dir + f'train/*.sang', True, True, args.batch_size, (args.image_size, args.image_size, 3), 1, 2, train_decode_fn, ['image', 'label'])
        validation_reader = reader.SH_Reader(sang_dir + f'validation/*.sang', False, False, args.batch_size, (args.image_size, args.image_size, 3), 1, 2, test_decode_fn, ['image', 'label'])
        test_reader = reader.SH_Reader(sang_dir + f'test/*.sang', False, False, args.batch_size, (args.image_size, args.image_size, 3), 1, 2, test_decode_fn, ['image', 'label'])
        
        dataset_dic[dataset_name]['train'] = train_reader
        dataset_dic[dataset_name]['validation'] = validation_reader
        dataset_dic[dataset_name]['test'] = test_reader
        dataset_dic[dataset_name]['classes'] = class_dic[dataset_name]

        print('# Dataset ({}), classes={}'.format(dataset_name, class_dic[dataset_name]))
    
    train_reader = dataset_dic['Cars']['train']
    validation_reader = dataset_dic['Cars']['validation']
    test_reader = dataset_dic['Cars']['test']
    
    for domain in ['validation', 'test', 'train']:
        sample_reader = dataset_dic['Cars'][domain]
        print(f'{domain}.start()')

        st_time = time.time()

        sample_reader.start()

        for images, labels in sample_reader:
            images = torch.stack(images)
            labels = torch.tensor(labels)

        sample_reader.close()

        print(f'{domain}.close() = {int(time.time() - st_time)}sec')

    train_reader.start()
    print('train_reader.start()')
    
    for images, labels in train_reader:
        images = torch.stack(images)
        labels = torch.tensor(labels)

        print(images.size())
        print(labels.size())
    
    train_reader.close()
    print('train_reader.close()')