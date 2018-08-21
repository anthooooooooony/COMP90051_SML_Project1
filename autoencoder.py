import pandas as pd

TRAIN_FILE = 'train.txt'

axis_dict = {}

with open(TRAIN_FILE, 'r') as f:
    index_num = 0
    for i in range(2):
        line = f.readline()
        line = line.strip()
        line = line.split('\t')
        for index, item in enumerate(line):
            axis_dict[item] = i + index
        index_num += len(line)


#def matrixConstruct():
