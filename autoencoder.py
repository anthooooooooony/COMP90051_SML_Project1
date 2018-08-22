import sys
import numpy as np
from scipy.sparse import coo_matrix, vstack



TRAIN_FILE = 'train.txt'

axis_dict = {}

with open(TRAIN_FILE, 'r') as f:
    for line in f.readlines():
        line = line.strip()
        line = line.split('\t')
        # print len(line)
        for uid in line:
            axis_dict.setdefault(uid, len(axis_dict))


"""
with open(TRAIN_FILE, 'r') as f:
    for line in f.readlines():
        line = line.strip()
        line = line.split('\t')
        row = np.ones((1, len(line))) # row index
        l = [axis_dict[tar_node] for tar_node in line]
        col = np.array(l) # col index
        data = np.ones(1, len(line))
        data[1, axis_dict[line[0]]] = 0
        matrix(data, (row, col))

print matrix.toarray()[:2, :167]
"""



