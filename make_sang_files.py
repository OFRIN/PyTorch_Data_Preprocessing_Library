# Copyright (C) 2020 * Ltd. All rights reserved.
# author : Sanghyeon Jo <josanghyeokn@gmail.com>

import ray
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
parser.add_argument('--data_dir', default='C:/Classification_DB/', type=str)
parser.add_argument('--save_dir', default='./Example/', type=str)
parser.add_argument('--the_number_of_image_per_file', default=100, type=int)
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
    
    for dataset_name in ['Food-101', 'Caltech-256', 'CUB-200', 'DTD', 'Flowers-102', 'Pet', 'Cars', 'Dogs']:
        train_dataset, validation_dataset, test_dataset, classes = get_dataset_func(dataset_name)(args.data_dir, transform, transform)

        data_info[dataset_name] = {
            'train' : len(train_dataset),
            'validation' : len(validation_dataset),
            'test' : len(test_dataset)
        }

        print('# Dataset ({})'.format(dataset_name))
        print('[i] The size of train dataset = {}'.format(len(train_dataset)))
        print('[i] The size of validation dataset = {}'.format(len(validation_dataset)))
        print('[i] The size of test dataset = {}'.format(len(test_dataset)))
        print()
        
        for domain, dataset in zip(['train', 'validation', 'test'], [train_dataset, validation_dataset, test_dataset]):
            timer.tik()

            ids = []
            
            writer = SH_Writer(f'{args.save_dir}{dataset_name}/{domain}/', '{:08d}.sang', args.the_number_of_image_per_file)
            for image, label in dataset:
                label = label.item()
                if label == -1:
                    continue
                
                # print(image.shape, label)
                # cv2.imshow('show', image)
                # cv2.waitKey(0)

                ids.append(encode.remote(image, label, transform))
                if len(ids) == args.the_size_of_accumulation:
                    for example in ray.get(ids):
                        writer(example)

                    ids = []

            if len(ids) > 0:
                for example in ray.get(ids):
                    writer(example)
            
            writer.save()
            
            print('[i] {} - {} - {}sec'.format(dataset_name, domain, timer.tok()))

    write_json('class_info.json', data_info)

'''
10sec ./Example_100/Cars/train/*
load_ms : 169ms
merge_ms : 0ms
decode_ms : 10199ms
queue_ms : 0ms
generator_ms : 26ms

9sec ./Example_250/Cars/train/*
load_ms : 149ms
merge_ms : 0ms
decode_ms : 9374ms
queue_ms : 1ms
generator_ms : 13ms

9sec ./Example_500/Cars/train/*
load_ms : 127ms
merge_ms : 0ms
decode_ms : 9660ms
queue_ms : 4ms
generator_ms : 6ms

9sec ./Example_750/Cars/train/*
load_ms : 134ms
merge_ms : 0ms
decode_ms : 9320ms
queue_ms : 2ms
generator_ms : 4ms

9sec ./Example_1000/Cars/train/*
load_ms : 179ms
merge_ms : 0ms
decode_ms : 9630ms
queue_ms : 3ms
generator_ms : 4ms
'''

