
import cv2
import pickle

import numpy as np

def load_pickle(pickle_path):
    return pickle.load(open(pickle_path, 'rb'))

def dump_pickle(pickle_path, dataset):
    return pickle.dump(dataset, open(pickle_path, 'wb'))

def encode_image(image_data):
    _, image_data = cv2.imencode('.jpg', image_data)
    return image_data

def decode_image(image_data):
    image_data = np.fromstring(image_data, dtype = np.uint8)
    image_data = cv2.imdecode(image_data, cv2.IMREAD_COLOR)
    return image_data

