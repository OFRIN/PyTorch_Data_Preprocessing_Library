
from .utils import dump_pickle
from ..utility.utils import create_directory

class SH_Writer:
    def __init__(self, dataset_dir, data_pattern, the_number_of_example):
        self.dataset_index = 1
        self.dataset_format = create_directory(dataset_dir) + data_pattern

        self.accumulated_size = 0
        self.the_number_of_example = the_number_of_example
        
        self.dataset = {}
    
    def __call__(self, key, example):    
        self.accumulated_size += 1
        self.dataset[key] = example

        if self.accumulated_size == self.the_number_of_example:
            self.save()

            self.accumulated_size = 0
            self.dataset = {}
    
    def get_path(self):
        path = self.dataset_format.format(self.dataset_index)
        return path

    def save(self):
        if self.accumulated_size > 0:
            print('# Write path : {}'.format(self.get_path()))
            dump_pickle(self.get_path(), self.dataset)
            self.dataset_index += 1