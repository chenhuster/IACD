# -*- coding: utf-8 -*-
import pickle
import logging
import networkx as nx

def load_graph(path):
    """根据边连边读取网络
    Parameters:
        path:网络存放的路径
    return:
        G:读取后的网络
    """
    weight_edge = []
    file = open(path)
    for line in file:
        source, target, weight = line.split(' ')
        weight_edge.append((int(source), int(target))+(float(weight),))
    G_weight = nx.DiGraph()
    G_weight.add_weighted_edges_from(weight_edge)

    return G_weight

def pickle_read(path):
    with open(path, 'rb') as file:
        data = pickle.load(file)
    return data

def pickle_save(path, data):
    with open(path, 'wb') as file:
        pickle.dump(data, file, protocol=pickle.HIGHEST_PROTOCOL)




