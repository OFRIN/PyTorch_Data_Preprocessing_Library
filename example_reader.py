
import ray

from torchvision import transforms
from data.reader import SH_Reader

ray.init(num_cpus=4)

root_dir = 'C:/Classification_DB_PIL/'

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

test_transform = transforms.Compose([
    transforms.CenterCrop(image_size),

    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

train_reader = SH_Reader(
    root_dir + dataset_name + '/train/*',
    train_transform,
    True, 
    batch_size, 
    ['image', 'label']
)

test_reader_fn = lambda: SH_Reader(
    root_dir + dataset_name + '/test/*',
    test_transform,
    False, 
    batch_size, 
    ['image', 'label']
)

train_reader.start()

for i in range(1, 100 + 1):
    images, labels = next(train_reader)
    print(len(images))

    if i % 10 == 0:
        reader = test_reader_fn()
        reader.start()

        for images, labels in iter(reader):
            pass

        del reader

del train_reader
