import cv2
import glob

import numpy as np

from sanghyeon.data import reader
from sanghyeon.data import utils

def decode_fn(example):
    image = utils.decode_image(example['encoded_image'])
    image = cv2.resize(image, (224, 224))
    return image, example['label']

if __name__ == '__main__':
    root_dir = '//192.168.100.156/gynetworks/'
    data_dir = root_dir + 'Classification_DB/'
    
    for dataset_name in [
        'Caltech-256',
        'Cars',
        'CUB-200',
        'Dogs',
        'DTD',
        'Flowers-102',
        'Food-101',
        'Pet'
        ]:
        
        sang_dir = data_dir + f'{dataset_name}/'
        
        for domain in ['train', 'validation', 'test']:
            data_pattern = sang_dir + f'{domain}/*.sang'
            
            reader = reader.SH_Reader(data_pattern, True, True, 64, (224, 224, 3), 2, 4, decode_fn, ['image', 'label'])
            reader.start()
            
            for images, labels in reader:
                
                for image, label in zip(images, labels):
                    print(label)
                    cv2.imshow('show', image.astype(np.uint8))
                    cv2.waitKey(0)

            reader.close()
            
