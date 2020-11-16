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
        
        reader = Reader_For_Expert(
            pattern='C:/Classification_DB/Flowers-102/train/*',
            batch_size=64, transform=train_transform,
            the_size_of_queue_for_loaders=5, the_size_of_queue_for_decoders=5,
            the_number_of_loader=1, the_number_of_decoder=2,
            decode_fn=customized_decode_fn
        )
        
        # while True:
        for i in range(100):
            images, labels = next(reader)
            print(np.shape(images), np.shape(labels))

        reader.close()
        
    except KeyboardInterrupt:
        reader.close()

