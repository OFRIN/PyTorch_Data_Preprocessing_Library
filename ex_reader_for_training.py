import numpy as np

from torchvision import transforms

from sanghyeon.utility.utils import convert_OpenCV_to_PIL
from sanghyeon.utility.utils import convert_PIL_to_OpenCV

from sanghyeon.data.reader import Reader_For_Training

if __name__ == '__main__':
    try:
        train_transform = transforms.Compose(
            [
                convert_OpenCV_to_PIL,
                transforms.RandomResizedCrop(224, scale=(0.08, 1.0)),
                convert_PIL_to_OpenCV
            ]
        )
        
        reader = Reader_For_Training(
            pattern='C:/Classification_DB/Flowers-102/train/*',
            batch_size=64, transform=train_transform,
            the_size_of_queue_for_loaders=5, the_size_of_queue_for_decoders=5,
            the_number_of_loader=1, the_number_of_decoder=2,
            the_number_of_loading_file=5, the_number_of_element=2
        )
        reader.start()
        
        while True:
            images, labels = next(reader)
            print(np.shape(images), np.shape(labels))
        
    except KeyboardInterrupt:
        reader.close()