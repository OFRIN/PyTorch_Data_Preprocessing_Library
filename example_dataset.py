import cv2
import numpy as np

from torchvision import transforms
from data.reader import SH_Dataset

root_dir = './Example/'

batch_size = 64
image_size = 224
dataset_name = 'Cars'

train_transform = transforms.Compose([
    transforms.RandomCrop(image_size),
    transforms.RandomHorizontalFlip(0.5),
    # transforms.ColorJitter(brightness = 0.3, contrast = 0.3, saturation = 0.3, hue = 0.1),
    
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

train_dataset = SH_Dataset(root_dir + f'{dataset_name}/train/*.sang', train_transform)

for image, label in train_dataset:
    image = np.asarray(image)[..., ::-1]
    print(image.shape, label)

    cv2.imshow('show', image)
    cv2.waitKey(0)
