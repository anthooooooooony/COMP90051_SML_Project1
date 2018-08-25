import time
import pickle
from scipy.sparse import coo_matrix, vstack
import networkx as nx



TRAIN_FILE = 'train.txt'

data = {}

with open(TRAIN_FILE, 'r') as f:
    for line in f.readlines()[:20]:
        line = line.strip()
        line = line.split('\t')
        data.setdefault(line[0], line[1:])

print 'The length of the dict is {}'.format(len(data))
print
"""
for i in sorted(data.keys()):
        print 'The key is {}, the value is {}'.format(i, len(data[i]))
        print
"""
start_time = time.time()

graph = nx.DiGraph(data)
print 'Graph constructing time: {}'.format(time.time() - start_time)
print 'The number of nodes: {}'.format(graph.number_of_nodes())
print

start_time = time.time()

adjm = nx.adjacency_matrix(graph)
print 'Adjacency constructing time: {}'.format(time.time() - start_time)

binf = open('adjm.bin', 'wb')
start_time = time.time()
pickle.dump(adjm, binf)
print 'pickle storing time: {}'.format(time.time() - start_time)
binf.close()

#print adjm.todense()
