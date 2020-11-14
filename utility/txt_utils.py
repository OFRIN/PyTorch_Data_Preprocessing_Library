def read_txt(filepath):
    with open(filepath, 'r') as f:
        return [line.strip() for line in f.readlines()]

def write_txt(filepath, data_list):
    with open(filepath, 'w') as f:
        for data in data_list:
            f.write(data + '\n')