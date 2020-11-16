import numpy as np

from torchvision import transforms

from utility.utils import convert_OpenCV_to_PIL
from utility.utils import convert_PIL_to_OpenCV

from data.reader import Reader_For_Expert, Reader_For_Beginner
from data.utils import decode_image

def customized_decode_fn(example, transform):
    image = decode_image(example['encoded_image'])

    if transform is not None:
        image = transform(image)

    return image, example['label']

if __name__ == '__main__':
    try:
        train_transform = transforms.Compose(
            [
                convert_OpenCV_to_PIL,
                transforms.RandomResizedCrop(224, scale=(0.08, 1.0)),
                convert_PIL_to_OpenCV
            ]
        )
        
        for i in range(2):
            reader = Reader_For_Beginner(
                pattern='C:/Classification_DB/Flowers-102/train/*',
                batch_size=64, transform=train_transform,
                decode_fn=customized_decode_fn
            )
            
            print(f'# {i + 1}')
            for images, labels in reader:
                print(np.shape(images), np.shape(labels))
        
    except KeyboardInterrupt:
        reader.close()