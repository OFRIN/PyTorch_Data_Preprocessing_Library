import glob

import numpy as np
import multiprocessing as mp

from sanghyeon.data.loader import Loader
from sanghyeon.data.decoder import Decoder

if __name__ == '__main__':
    try:
        queue = mp.Queue(maxsize=5)
        queue_of_loader = mp.Queue(maxsize=5)

        file_paths = glob.glob('C:/Classification_DB/Flowers-102/train/*.sang')
        the_number_of_loading_file = 5

        loader = Loader(queue=queue_of_loader, file_paths=file_paths, the_number_of_loading_file=the_number_of_loading_file)
        loader.start()

        decoder = Decoder(queue=queue, queue_of_loader=queue_of_loader, batch_size=64, transform=None, the_number_of_element=2)
        decoder.start()

        while True:
            images, labels = decoder.get()
            print(np.shape(images), np.shape(labels))
    
    except KeyboardInterrupt:
        loader.close()
        decoder.close()

