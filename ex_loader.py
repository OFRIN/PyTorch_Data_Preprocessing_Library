import sys
import glob
import multiprocessing as mp

from sanghyeon.data.loader import Loader

if __name__ == '__main__':
    try:
        queue = mp.Queue(maxsize=5)
        file_paths = glob.glob('C:/Classification_DB/Flowers-102/train/*.sang')
        the_number_of_loading_file = 5

        loader = Loader(queue=queue, file_paths=file_paths, the_number_of_loading_file=the_number_of_loading_file)
        loader.start()
        
        while True:
            examples = loader.get()
            print(len(examples))
    
    except KeyboardInterrupt:
        loader.close()

