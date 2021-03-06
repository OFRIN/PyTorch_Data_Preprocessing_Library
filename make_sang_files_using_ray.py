# Copyright (C) 2020 * Ltd. All rights reserved.
# author : Sanghyeon Jo <josanghyeokn@gmail.com>

import sys
import ray
import copy
import argparse

from PIL import Image
from torchvision import transforms

from core.data_utils import get_dataset_func

from data.writer import SH_Writer
from data.utils import encode_image

from util.json_utils import write_json
from util.time_utils import Timer
from util.utils import set_seed

parser = argparse.ArgumentParser()

parser.add_argument('--seed', default=0, type=int)
parser.add_argument('--data_dir', default='F:/Classification_DB/', type=str)
parser.add_argument('--save_dir', default='./Example_Joblib/', type=str)
parser.add_argument('--the_number_of_example_per_file', default=250, type=int)
parser.add_argument('--the_size_of_accumulation', default=1000, type=int)

args = parser.parse_args()

ray.init()

@ray.remote
def encode(image, label, transform):
    image = transform(image)

    return {
        'encoded_image' : encode_image(image),
        'label' : label,
    }

if __name__ == '__main__':
    set_seed(args.seed)
    
    transform = transforms.Compose([
        transforms.Resize(256, interpolation=Image.CUBIC),
    ])
    
    timer = Timer()

    data_info = {}

    for dataset_name in ['Food-101', 'CUB-200','Caltech-256', 'DTD', 'Flowers-102', 'Pet', 'Cars', 'Dogs']:
        train_dataset, validation_dataset, test_dataset, classes = get_dataset_func(dataset_name)(args.data_dir, transform, transform)

        data_info[dataset_name] = {
            'train' : len(train_dataset),
            'validation' : len(validation_dataset),
            'test' : len(test_dataset),
            'classes' : classes
        }
        
        print('# Dataset ({})'.format(dataset_name))
        print('[i] The size of train dataset = {}'.format(len(train_dataset)))
        print('[i] The size of validation dataset = {}'.format(len(validation_dataset)))
        print('[i] The size of test dataset = {}'.format(len(test_dataset)))
        print()
        
        for domain, dataset in zip(['train', 'validation', 'test'], [train_dataset, validation_dataset, test_dataset]):
            timer.tik()

            ids = []
            
            writer = SH_Writer(f'{args.save_dir}{dataset_name}/{domain}/', "{:05d}.sang", args.the_number_of_example_per_file)
            length = len(dataset)

            for index, (image, label) in enumerate(dataset):
                label = label.item()
                if label == -1:
                    continue

                sys.stdout.write('\r{}, {}, [{}/{}]'.format(dataset_name, domain, index + 1, length))
                sys.stdout.flush()
                
                ids.append(encode.remote(image, label, transform))
                if len(ids) == args.the_size_of_accumulation:
                    for example in ray.get(ids):
                        writer(example)

                    ids = []

            if len(ids) > 0:
                for example in ray.get(ids):
                    writer(example)
            
            writer.end()
            
            print('[i] {} - {} - {}sec'.format(dataset_name, domain, timer.tok()))

    write_json('class_info.json', data_info)

'''

'''