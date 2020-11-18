# Copyright (C) 2020 * Ltd. All rights reserved.
# author : Sanghyeon Jo <josanghyeokn@gmail.com>

from .txt_utils import add_txt

def log_print(string, path):
    print(string)
    add_txt(path, string)

def csv_print(data_list, path):
    string = ''
    
    for data in data_list[:-1]:
        if type(data) != type(str):
            data = str(data)
        string += (data + ',')

    data = data_list[-1]
    if type(data) != type(str):
        data = str(data)
    string += data
    
    add_txt(path, string)
