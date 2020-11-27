import torch
import random
import numpy as np

from PIL import Image

from torchvision import datasets
from torchvision import transforms

from util.utils import *
from util.txt_utils import *
from util.json_utils import *

class Custom_Dataset(torch.utils.data.Dataset):
    def __init__(self, dataset, indices=None, transform=None):
        self.dataset = dataset
        self.indices = indices
        self.transform = transform

        if self.indices is None:
            self.indices = np.arange(len(self.dataset))
    
    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        image, label = self.dataset[self.indices[index]]
        return image, label

def split_train_and_validation_datasets(dataset, classes, ratio=0.1):
    labels = dataset.labels
    
    train_indices = []
    validation_indices = []

    for class_index in range(classes):
        indices = np.where(labels == class_index)[0]
        validation_size_per_class = int(len(indices) * ratio)
        
        np.random.shuffle(indices)
        
        train_indices.extend(indices[:-validation_size_per_class])
        validation_indices.extend(indices[-validation_size_per_class:])
    
    return train_indices, validation_indices

def get_class_dic(class_path):
    class_dic = {}
    class_names = read_txt(class_path)

    for class_index, class_name in enumerate(class_names):
        class_dic[class_name] = class_index
    
    return class_dic, class_names

class Dataset(torch.utils.data.Dataset):
    def __init__(self, data_dir, json_path, class_path):
        self.data_dir = data_dir
        self.data_dic = read_json(json_path)

        self.class_dic, self.class_names = get_class_dic(class_path)
        self.classes = len(self.class_names)

        self.image_names = list(self.data_dic.keys())
        self.labels = np.asarray([self.class_dic[str(self.data_dic[image_name])] for image_name in self.image_names], dtype=np.int32)

    def __len__(self):
        return len(self.image_names)
    
    def __getitem__(self, index):
        image_name = self.image_names[index]
        
        # image = cv2.imread(self.data_dir + image_name)
        try:
            image = Image.open(self.data_dir + image_name)
        except:
            print(self.data_dir + image_name)
            return None, torch.tensor(-1, dtype=torch.long)

        if image.mode != 'RGB':
            image = image.convert('RGB')

        label = self.labels[index]
        return image, torch.tensor(label, dtype=torch.long)

def get_dataset_func(name):
    if name == 'Caltech-256':
        return get_Caltech_256_datasets
    elif name == 'CUB-200':
        return get_CUB_200_datasets
    elif name == 'DTD':
        return get_DTD_datasets
    elif name == 'Food-101':
        return get_Food_101_datasets
    elif name == 'Flowers-102':
        return get_Flowers_datasets
    elif name == 'Pet':
        return get_Pet_datasets
    elif name == 'Cars':
        return get_Cars_datasets
    elif name == 'Dogs':
        return get_Dogs_datasets
    else:
        raise ValueError("Not Found Name ({})".format(name))

def get_Caltech_256_datasets(data_dir, train_transform, test_transform):
    root_dir = data_dir + 'Caltech-256/'

    base_dataset = Dataset(data_dir, root_dir + 'train.json', root_dir + 'class_names.txt')
    test_dataset = Dataset(data_dir, root_dir + 'test.json', root_dir + 'class_names.txt')
    
    train_indices, validation_indices = split_train_and_validation_datasets(base_dataset, base_dataset.classes, ratio=0.1)

    train_dataset = Custom_Dataset(base_dataset, train_indices, train_transform)
    validation_dataset = Custom_Dataset(base_dataset, validation_indices, test_transform)
    test_dataset = Custom_Dataset(test_dataset, transform=test_transform)

    return train_dataset, validation_dataset, test_dataset, base_dataset.classes

def get_CUB_200_datasets(data_dir, train_transform, test_transform):
    root_dir = data_dir + 'CUB-200/'

    base_dataset = Dataset(data_dir, root_dir + 'train.json', root_dir + 'class_names.txt')
    test_dataset = Dataset(data_dir, root_dir + 'test.json', root_dir + 'class_names.txt')
    
    train_indices, validation_indices = split_train_and_validation_datasets(base_dataset, base_dataset.classes, ratio=0.1)
    
    train_dataset = Custom_Dataset(base_dataset, train_indices, train_transform)
    validation_dataset = Custom_Dataset(base_dataset, validation_indices, test_transform)
    test_dataset = Custom_Dataset(test_dataset, transform=test_transform)

    return train_dataset, validation_dataset, test_dataset, base_dataset.classes

def get_DTD_datasets(data_dir, train_transform, test_transform):
    root_dir = data_dir + 'DTD/'

    train_dataset = Dataset(data_dir, root_dir + 'train.json', root_dir + 'class_names.txt')
    validation_dataset = Dataset(data_dir, root_dir + 'validation.json', root_dir + 'class_names.txt')
    test_dataset = Dataset(data_dir, root_dir + 'test.json', root_dir + 'class_names.txt')
    classes = train_dataset.classes
    
    train_dataset = Custom_Dataset(train_dataset, transform=train_transform)
    validation_dataset = Custom_Dataset(validation_dataset, transform=test_transform)
    test_dataset = Custom_Dataset(test_dataset, transform=test_transform)

    return train_dataset, validation_dataset, test_dataset, classes

def get_Food_101_datasets(data_dir, train_transform, test_transform):
    root_dir = data_dir + 'Food-101/'

    base_dataset = Dataset(data_dir, root_dir + 'train.json', root_dir + 'class_names.txt')
    test_dataset = Dataset(data_dir, root_dir + 'test.json', root_dir + 'class_names.txt')
    
    train_indices, validation_indices = split_train_and_validation_datasets(base_dataset, base_dataset.classes, ratio=0.1)
    
    train_dataset = Custom_Dataset(base_dataset, train_indices, train_transform)
    validation_dataset = Custom_Dataset(base_dataset, validation_indices, test_transform)
    test_dataset = Custom_Dataset(test_dataset, transform=test_transform)

    return train_dataset, validation_dataset, test_dataset, base_dataset.classes

def get_Flowers_datasets(data_dir, train_transform, test_transform):
    root_dir = data_dir + 'Oxford_Flowers-102/'

    train_dataset = Dataset(data_dir, root_dir + 'train.json', root_dir + 'class_names.txt')
    validation_dataset = Dataset(data_dir, root_dir + 'validation.json', root_dir + 'class_names.txt')
    test_dataset = Dataset(data_dir, root_dir + 'test.json', root_dir + 'class_names.txt')
    
    classes = train_dataset.classes

    train_dataset = Custom_Dataset(train_dataset, transform=train_transform)
    validation_dataset = Custom_Dataset(validation_dataset, transform=test_transform)
    test_dataset = Custom_Dataset(test_dataset, transform=test_transform)
    
    return train_dataset, validation_dataset, test_dataset, classes

def get_Pet_datasets(data_dir, train_transform, test_transform):
    root_dir = data_dir + 'Oxford-IIIT_Pet_Dataset/'
    
    base_dataset = Dataset(data_dir, root_dir + 'train.json', root_dir + 'class_names.txt')
    test_dataset = Dataset(data_dir, root_dir + 'test.json', root_dir + 'class_names.txt')
    
    train_indices, validation_indices = split_train_and_validation_datasets(base_dataset, base_dataset.classes, ratio=0.1)
    
    train_dataset = Custom_Dataset(base_dataset, train_indices, train_transform)
    validation_dataset = Custom_Dataset(base_dataset, validation_indices, test_transform)
    test_dataset = Custom_Dataset(test_dataset, transform=test_transform)

    return train_dataset, validation_dataset, test_dataset, base_dataset.classes

def get_Cars_datasets(data_dir, train_transform, test_transform):
    root_dir = data_dir + 'Stanford_Cars_Dataset/'
    
    base_dataset = Dataset(data_dir, root_dir + 'train.json', root_dir + 'class_names.txt')
    test_dataset = Dataset(data_dir, root_dir + 'test.json', root_dir + 'class_names.txt')
    
    train_indices, validation_indices = split_train_and_validation_datasets(base_dataset, base_dataset.classes, ratio=0.1)
    
    train_dataset = Custom_Dataset(base_dataset, train_indices, train_transform)
    validation_dataset = Custom_Dataset(base_dataset, validation_indices, test_transform)
    test_dataset = Custom_Dataset(test_dataset, transform=test_transform)

    return train_dataset, validation_dataset, test_dataset, base_dataset.classes

def get_Dogs_datasets(data_dir, train_transform, test_transform):
    root_dir = data_dir + 'Stanford_Dogs_Dataset/'
    
    base_dataset = Dataset(data_dir, root_dir + 'train.json', root_dir + 'class_names.txt')
    test_dataset = Dataset(data_dir, root_dir + 'test.json', root_dir + 'class_names.txt')
    
    train_indices, validation_indices = split_train_and_validation_datasets(base_dataset, base_dataset.classes, ratio=0.1)
    
    train_dataset = Custom_Dataset(base_dataset, train_indices, train_transform)
    validation_dataset = Custom_Dataset(base_dataset, validation_indices, test_transform)
    test_dataset = Custom_Dataset(test_dataset, transform=test_transform)

    return train_dataset, validation_dataset, test_dataset, base_dataset.classes

