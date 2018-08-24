import sys
import numpy as np
from scipy.sparse import coo_matrix, vstack


TRAIN_FILE = 'train.txt'

data = {}

with open(TRAIN_FILE, 'r') as f:
    for line in f.readlines():
        line = line.strip()
        line = line.split('\t')
        data.setdefault(line[0], line[1:])

print 'The length of the dict is {}'.format(len(data))

for i in sorted(data.keys()[:10]):
        print 'The key is {}, the value is {}'.format(i, data[i])


