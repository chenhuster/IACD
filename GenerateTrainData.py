# -*- coding: utf-8 -*-
import networkx as nx
import numpy as np
import os
import pandas as pd
np.seterr(divide='ignore',invalid='ignore')
from Utils import *
from tqdm import tqdm

def get_neigbors(g, node, depth=1):
    output = {}
    layers = dict(nx.bfs_successors(g, source=node, depth_limit=depth))
    nodes = [node]
    for i in range(1,depth+1):
        output[i] = []
        for x in nodes:
            output[i].extend(layers.get(x,[]))
        nodes = output[i]
    return output



def Counts_High_Order_Nodes(G, depth = 2):
    NODES_LIST = list(G.nodes)
    output = {}
    output = output.fromkeys(NODES_LIST)
    for node in NODES_LIST:
        count = len(get_neigbors(G, node, depth)[depth])
        output[node] = count

    return output

def Generate_Node_Feature3(G):
    NODES_LIST = list(G.nodes)

    # the number of its one-hop neighbors
    degree_dict = nx.degree_centrality(G)
    degree_list = np.array([degree_dict[i] for i in NODES_LIST])[:, None]
    degree_list = degree_list / np.max(degree_list)


    # the number of its two-hop neighbors
    second_neighbor = Counts_High_Order_Nodes(G, depth=2)
    second_neighbor_list = np.array([second_neighbor[i] for i in NODES_LIST])[:, None]
    second_neighbor_list = second_neighbor_list/np.max(second_neighbor_list)

    # average out-degree of its one-hop neighbors
    neighbor_average_degree = nx.average_neighbor_degree(G)
    neighbor_average_degree_list = np.array([neighbor_average_degree[i] for i in NODES_LIST])[:, None]
    neighbor_average_degree_list = neighbor_average_degree_list / np.max(neighbor_average_degree_list)

    # local clustering coefficient
    local_clustering_dict = nx.clustering(G)
    local_clustering_list = np.array([local_clustering_dict[i] for i in NODES_LIST])[:, None]
    local_clustering_list =  local_clustering_list / np.max(local_clustering_list)

    # constant
    constant_list = np.ones((len(NODES_LIST), 1))

    node_features = np.concatenate((degree_list, second_neighbor_list, neighbor_average_degree_list,
                                    local_clustering_list,  constant_list), axis=1)

    node_features[np.isnan(node_features)] = 0

    return node_features


def GenerateTrainData(id, seed_size, graph, g_adjacent_matrix, g_features):

    print(f'Generating No.{id} training {seed_size} graphs')

    DATASET_PATH = os.path.join(os.getcwd(), 'IC results', '100_4', str(seed_size) +'_' + str(id) + '.csv')
    # os.makedirs(DATASET_PATH, exist_ok=True)
    simulation = pd.read_csv(DATASET_PATH)
    seed_set = np.array(simulation['Seed'], dtype=int)
    seed_set = np.array(seed_set).reshape(-1)
    label_dict = dict(zip(np.array(simulation['Node'], dtype=int), simulation['IC']))


    # Generate Label
    label = []
    node_list = list(graph.nodes)
    for node in node_list:
        label.append(float(label_dict[node]))
    label = (np.array(label) / np.min(label)) - 1.0
    label = np.array(label) / np.max(label)
    label = np.array(label).reshape(-1)

    data_process_path = os.path.join(os.getcwd(), 'data_process', '100_4')
    os.makedirs(data_process_path, exist_ok=True)
    pickle_save(os.path.join(data_process_path, 'train_adj_'+str(seed_size) + '_' + str(id)+'.npy'), g_adjacent_matrix)
    pickle_save(os.path.join(data_process_path, 'train_feature_' + str(seed_size) + '_' + str(id) + '.npy'), g_features)
    pickle_save(os.path.join(data_process_path, 'train_seed_' + str(seed_size) + '_' + str(id) + '.npy'), seed_set)
    pickle_save(os.path.join(data_process_path, 'train_label_' + str(seed_size) + '_' + str(id) + '.npy'),   label)


def PrepareTrainData(seed_sizes, num_per_size):

    Adj = []
    Label =  []
    Feature = []
    Seed = []

    DATA_PATH = os.path.join(os.getcwd(), 'data_process', '100_4')

    for seed_size in seed_sizes:
        for id in range(num_per_size):
            adj_id = pickle_read(os.path.join(DATA_PATH, 'train_adj_'+str(seed_size) + '_' + str(id)+'.npy'))
            feature_id = pickle_read(os.path.join(DATA_PATH, 'train_feature_' + str(seed_size) + '_' + str(id) + '.npy'))
            seed_id = pickle_read(os.path.join(DATA_PATH, 'train_seed_' + str(seed_size) + '_' + str(id) + '.npy'))
            label_id = pickle_read(os.path.join(DATA_PATH, 'train_label_' + str(seed_size) + '_' + str(id) + '.npy'))
            Adj.append(adj_id)
            Feature.append(feature_id)
            Seed.append(seed_id[:, None])
            Label.append(label_id[:, None])

    SAVE_PATH = os.path.join(os.getcwd(), 'data', 'train', '100_4')
    os.makedirs(SAVE_PATH, exist_ok=True)
    pickle_save(os.path.join(SAVE_PATH, 'train_dataset_adj.npy'), Adj)
    pickle_save(os.path.join(SAVE_PATH, 'train_dataset_label.npy'), Label)
    pickle_save(os.path.join(SAVE_PATH, 'train_dataset_feature.npy'), Feature)
    pickle_save(os.path.join(SAVE_PATH, 'train_dataset_seed.npy'),Seed)


def make_dataset(graph, g_features, g_adjacent_matrix):

    seed_sizes_100 = [10, 20, 30, 40, 50]
    num_per_size = 100
    for seed_size in seed_sizes_100:
        for id in range(num_per_size):
            GenerateTrainData(id, seed_size, graph, g_adjacent_matrix, g_features)

    PrepareTrainData(seed_sizes_100, num_per_size)


if __name__ == '__main__':

    BA_100_4 = load_graph('.\\train_sy_network\\Train_100_4.txt')
    seed_sizes = [10, 20, 30, 40, 50]
    num_per_size = 100
    g_adjacent_matrix = np.array(nx.adjacency_matrix(BA_100_4).todense())
    g_features = Generate_Node_Feature3(BA_100_4)
    for seed_size in seed_sizes:
        for id in range(num_per_size):
            GenerateTrainData(id, seed_size, BA_100_4, g_adjacent_matrix, g_features)

    PrepareTrainData(seed_sizes, num_per_size)



