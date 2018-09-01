import time
import pickle
from random import choice
from scipy.spatial import distance
import numpy as np
import pandas as pd
import lightgbm
import networkx as nx

TRAIN_FILE = 'train.txt'
INSTANCE_LIMIT = 10000

'''
with open('test-public.txt', 'r') as f:
    
    lines = f.readlines()[1:]
    for line in lines:
        line = line.strip()
        line = line.split('\t')
'''


def get_data(input_file, negative_padding=False):
    data = {}
    relations = 0
    nodes = []
    with open(input_file, 'r') as f:
        for line in f.readlines():
            line = line.strip()
            line = line.split('\t')
            data.setdefault(line[0], line[1:])
            relations += len(line[1:])
            nodes.extend([line[0]])
            nodes.extend(line[1:])
    nodes = list(set(nodes))
    print 'The length of the dict is {}'.format(len(data))
    print 'The total number of node is {}'.format(len(nodes))
    print 'The total relation number is {}\n'.format(relations)

    if negative_padding:
        neg_ins = {}
        for _ in range(relations):
            source = choice(nodes)
            target = choice(nodes)
            while target == source:
                target = choice(nodes)
            try:
                while target in data[source]:
                    target = choice(nodes)
            except KeyError:
                pass
            neg_ins.setdefault(source, [])
            neg_ins[source] = neg_ins[source] + [target]
        return data, neg_ins
    return data


def graph_construct(data):
    start_time = time.time()
    graph = nx.Graph(data)
    print 'Graph constructing time: {} hours'.format((time.time() - start_time) / 3600)
    digraph = nx.DiGraph(data)
    start_time = time.time()
    print 'Digraph constructing time: {} hours'.format((time.time() - start_time) / 3600)
    print 'The number of nodes: {}'.format(graph.number_of_nodes())
    return graph, digraph


def to_train_input(data_list, negative_instance = False):
    graph, digraph = graph_construct(data_list[0])
    print 'The number of edge is {}\n'.format(nx.number_of_edges(graph))
    start_time = time.time()
    pagerank = nx.pagerank(digraph)
    print 'Pagerank constructing time: {} hours'.format((time.time() - start_time) / 3600)
    start_time = time.time()
    hub, auth = nx.hits(digraph)
    print 'Hitting time constructing time: {} hours'.format((time.time() - start_time) / 3600)
    '''
    start_time = time.time()
    core = nx.core_number(digraph)
    print 'Core numbers constructing time: {} hours'.format((time.time() - start_time) / 3600)
    '''
    def feature_extraction(data, negative_instance):
        counter = 0
        xs_input, label = [], []
        for source_node, tar_list in data.items():
            for target_node in tar_list:
                xs_input.append(get_feature(source_node, target_node))
                label.append(0) if negative_instance else label.append(1)
                counter += 1
                if counter % 100 == 0:
                    print 'Instance processing over {} out of {}'.format(
                        counter, nx.number_of_edges(graph))
                if counter == INSTANCE_LIMIT:
                    assert len(xs_input) == len(label)
                    return xs_input, label
        assert len(xs_input) == len(label)
        return xs_input, label

    def get_feature(source, target):

        features = {}

        def set_feature(name, val):
            if name not in features:
                features[name] = []
            features[name].append(val)

        def cosine_distance(node_list1, node_list2):
            id2index = dict([(id, i) for i, id in enumerate(set(node_list1 + node_list2))])
            a = np.zeros((len(id2index),))
            b = np.zeros((len(id2index),))
            for key in node_list1:
                a[id2index[key]] = 1
            for key in node_list2:
                b[id2index[key]] = 1
            return distance.cosine(a, b)

        source_succ = set(digraph.successors(source))
        source_pred = set(digraph.predecessors(source))
        target_succ = set(digraph.successors(target))
        target_pred = set(digraph.predecessors(target))
        set_feature('len_source_successors', len(source_succ))
        set_feature('len_target_successors', len(target_succ))
        set_feature('len_source_predecessors', len(source_pred))
        set_feature('len_target_predecessors', len(target_pred))
        common_succ = len(source_succ.intersection(target_succ))
        common_pred = len(source_pred.intersection(source_pred))
        set_feature('common_successor_number', common_succ)
        set_feature('common_predecessor_number', common_pred)
        succ_union = source_succ.union(target_succ)
        pred_union = source_pred.union(target_pred)
        set_feature('jaccard_distance_between_successors',
                    common_succ / len(succ_union) if len(succ_union) != 0 else 0)
        set_feature('jaccard_distance_between_predecessors',
                    common_pred / len(pred_union) if len(pred_union) != 0 else 0)

        #set_feature('successor_cosine', cosine_distance(data[source], data[target]))
        #set_feature('predecessor_cosine', cosine_distance(source_pred, target_pred))

        set_feature('shortest_path', nx.shortest_path_length(digraph, source, target)
                    if digraph.has_edge(source, target) else 0)
        pref_attch = nx.preferential_attachment(graph, [(source, target)])
        for u, v, p in pref_attch:
            set_feature('preference_attachment',p)# if graph.has_edge(source, target) else 0)
        aa_index = nx.adamic_adar_index(graph, [(source, target)])
        for u, v, p in aa_index:
            set_feature('adamic_adar_index',p)# if graph.has_edge(source, target) else 0)
        jcd_coe = nx.jaccard_coefficient(graph, [(source, target)])
        for u, v, p in jcd_coe:
            set_feature('jaccard_coefficient',p)# if graph.has_edge(source, target) else 0)
        reallo_index = nx.resource_allocation_index(graph, [(source, target)])
        for u, v, p in reallo_index:
            set_feature('resource_allocation_index',p)# if graph.has_edge(source, target) else 0)
        set_feature('cluster_source', nx.clustering(graph, source))
        set_feature('cluster_target', nx.clustering(graph, target))

        set_feature('source_pagerank', pagerank[source])
        set_feature('target_pagerank', pagerank[target])
        set_feature('source_authorities', auth[source])
        set_feature('target_authorities', auth[target])
        set_feature('source_hubs', hub[source])
        set_feature('target_hubs', hub[target])
        #set_feature('source_core_num', core[source])
        #set_feature('target_core_num', core[target])
        return features

    if not negative_instance:
        pos_feature, pos_label = feature_extraction(data_list[0], negative_instance)
        return pos_feature, pos_label
    else:
        pos_feature, pos_label = feature_extraction(data_list[0], negative_instance)
        neg_feature, neg_label = feature_extraction(data_list[1], negative_instance)
        return pos_feature, pos_label, neg_feature, neg_label


train_data, neg_train_data = get_data(TRAIN_FILE, negative_padding=True)
#print 'Length of training data is {}'.format(len(train_data.items()[2]))
pos_neg_data = [train_data, neg_train_data]
pos_input, pos_label, neg_input, neg_label = to_train_input(pos_neg_data, True)

input_df = pd.DataFrame(data=pos_input)
csv = input_df.to_csv(index=False, sep='\t')
with open('pos.csv', 'w') as f:
    f.write(csv)
input_df = pd.DataFrame(data=neg_input)
csv = input_df.to_csv(index=False, sep='\t')
with open('neg.csv', 'w') as f:
    f.write(csv)

'''
#start_time = time.time()
#adjm = nx.adjacency_matrix(graph)
#print 'Adjacency constructing time: {} hours'.format((time.time() - start_time)/3600)
binf = open('adjm.bin', 'wb')
start_time = time.time()
pickle.dump(adjm, binf)
print 'pickle storing time: {} hours'.format((time.time() - start_time)/3600)
binf.close()

#print adjm.todense()
'''





#in_deg_cent = nx.in_degree_centrality(graph)
#out_deg_cent = nx.out_degree_centrality(graph)
#print in_deg_cent[node1]
#print out_deg_cent[node2]


gbm = lightgbm.LGBMClassifier(objective='binary', learning_rate=0.01,
                              n_estimators=1000, subsample=0.8)
gbm.fit(train_input, train_label, verbose=200)

