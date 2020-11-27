# Copyright (C) 2020 * Ltd. All rights reserved.
# author : Sanghyeon Jo <josanghyeokn@gmail.com>

import os

import pickle
import numpy as np

from io import BytesIO
from PIL import Image

def create_directory(path):
    if not os.path.isdir(path):
        os.makedirs(path)
    return path

def deserialize(data):
    return pickle.loads(data)

def serialize(data):
    return pickle.dumps(data)

def encode_image(image_data):
    buffer = BytesIO()
    image_data.save(buffer, format='JPEG')
    return buffer

def decode_image(image_data):
    return Image.open(image_data)
