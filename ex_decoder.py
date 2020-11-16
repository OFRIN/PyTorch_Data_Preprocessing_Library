import glob

import numpy as np
import multiprocessing as mp

from data.loader import Loader
from data.decoder import Decoder

from data.utils import decode_image

def customized_decode_fn(example, transform):
    image = decode_image(example['encoded_image'])

    if transform is not None:
        image = transform(image)

    return image, example['label']

if __name__ == '__main__':
    try:
        queue = mp.Queue(maxsize=5)
        queue_of_loader = mp.Queue(maxsize=5)

        file_paths = glob.glob('C:/Classification_DB/Flowers-102/train/*.sang')
        the_number_of_loading_file = 5
        
        loader = Loader(queue=queue_of_loader, file_paths=file_paths, the_number_of_loading_file=the_number_of_loading_file)
        loader.start()
        
        decoder = Decoder(queue=queue, queue_of_loader=queue_of_loader, batch_size=64, transform=None, decode_fn=customized_decode_fn)
        decoder.start()
        
        while True:
            images, labels = decoder.get()
            print(np.shape(images), np.shape(labels))
    
    except KeyboardInterrupt:
        loader.close()
        decoder.close()

