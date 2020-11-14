import numpy as np

from torchvision import transforms

from sanghyeon.utility.utils import convert_OpenCV_to_PIL
from sanghyeon.utility.utils import convert_PIL_to_OpenCV

from sanghyeon.data.reader import Reader_For_Training, Reader_For_Testing

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
            reader = Reader_For_Testing(
                pattern='C:/Classification_DB/Flowers-102/train/*',
                batch_size=64, transform=train_transform,
                the_number_of_element=2
            )
            
            print(f'# {i + 1}')
            for images, labels in reader:
                print(np.shape(images), np.shape(labels))
        
    except KeyboardInterrupt:
        reader.close()