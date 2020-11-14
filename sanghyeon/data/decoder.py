import copy
import multiprocessing as mp

from .utils import decode_image

class Decoder(mp.Process):
    def __init__(self, **kwargs):
        super().__init__()
        
        self.queue = kwargs['queue']
        self.queue_of_loader = kwargs['queue_of_loader']
        
        self.batch_size = kwargs['batch_size']
        self.transform = kwargs['transform']

        self.decode_fn = kwargs['decode_fn']
        self.the_number_of_element = kwargs['the_number_of_element']

        self.init()
    
    def init(self):
        self.batch_dataset = [[] for i in range(self.the_number_of_element)]

    def run(self):
        while True:
            for example in self.queue_of_loader.get():
                # for dataset, value in zip(self.batch_dataset, self.decode_fn(self, example)):
                for dataset, value in zip(self.batch_dataset, self.decode_fn(example, self.transform)):
                    dataset.append(value)
                
                if len(self.batch_dataset[0]) == self.batch_size:
                    self.queue.put(copy.deepcopy(self.batch_dataset))
                    self.init()
    
    def get(self):
        return self.queue.get()

    def close(self):
        self.terminate()
        self.join()
                    

