# Copyright (C) 2020 * Ltd. All rights reserved.
# author : Sanghyeon Jo <josanghyeokn@gmail.com>

import os
import cv2
import time
import torch
import random
import pickle
import argparse
import datetime

import numpy as np
import xml.etree.ElementTree as ET

from PIL import Image

def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def convert_OpenCV_to_PIL(image):
    # Gray
    if len(np.shape(image)) == 2:
        return Image.fromarray(image)
    return Image.fromarray(image[..., ::-1])

def convert_PIL_to_OpenCV(image):
    # Gray
    if len(np.shape(image)) == 2:
        return np.asarray(image)
    return np.asarray(image)[..., ::-1]

def read_xml(xml_path):
    tree = ET.parse(xml_path)
    root = tree.getroot()
    
    size = root.find('size')
    image_width = int(size.find('width').text)
    image_height = int(size.find('height').text)
    
    bboxes = []
    classes = []
    
    for obj in root.findall('object'):
        label = obj.find('name').text
        bbox = obj.find('bndbox')
        
        bbox_xmin = max(min(int(bbox.find('xmin').text.split('.')[0]), image_width - 1), 0)
        bbox_ymin = max(min(int(bbox.find('ymin').text.split('.')[0]), image_height - 1), 0)
        bbox_xmax = max(min(int(bbox.find('xmax').text.split('.')[0]), image_width - 1), 0)
        bbox_ymax = max(min(int(bbox.find('ymax').text.split('.')[0]), image_height - 1), 0)

        if (bbox_xmax - bbox_xmin) == 0 or (bbox_ymax - bbox_ymin) == 0:
            continue
        
        bboxes.append([bbox_xmin, bbox_ymin, bbox_xmax, bbox_ymax])
        classes.append(label)
    
    return np.asarray(bboxes, dtype = np.float32), np.asarray(classes)

def get_today():
    now = time.localtime()
    s = "%04d-%02d-%02d-%02dh%02dm%02ds" % (now.tm_year, now.tm_mon, now.tm_mday, now.tm_hour, now.tm_min, now.tm_sec)
    return s

def log_print(string, log_path = './log.txt'):
    print(string)

    if log_path is not None:
        f = open(log_path, 'a+')
        f.write(string + '\n')
        f.close()

def csv_print(data_list, log_path = './log.csv'):
    string = ''
    
    for data in data_list[:-1]:
        if type(data) != type(str):
            data = str(data)
        string += (data + ',')

    data = data_list[-1]
    if type(data) != type(str):
        data = str(data)
    string += data
    
    if log_path is not None:
        f = open(log_path, 'a+')
        f.write(string + '\n')
        f.close()

def create_directory(path):
    if not os.path.isdir(path):
        os.makedirs(path)
    return path

class Timer:
    def __init__(self):
        self.start_time = 0.0
        self.end_time = 0.0

        self.tik()
    
    def tik(self):
        self.start_time = time.time()
    
    def tok(self, ms = False, clear=False):
        self.end_time = time.time()

        if ms:
            duration = int((self.end_time - self.start_time) * 1000)
        else:
            duration = int(self.end_time - self.start_time)

        if clear:
            self.tik()

        return duration

class Average_Meter:
    def __init__(self, keys):
        self.keys = keys
        self.data_dic = {key : [] for key in keys}
    
    def add(self, dic):
        for key, value in dic.items():
            self.data_dic[key].append(value)

    def get(self, keys=None, clear=False):
        if keys is None:
            keys = self.keys

        dataset = [np.mean(self.data_dic[key]) for key in keys]
        if clear:
            self.clear()

        return dataset
    
    def clear(self):
        for key in self.data_dic.keys():
            self.data_dic[key] = []

