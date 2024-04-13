# -*- coding: utf-8 -*-
import argparse
import os
import numpy as np
import networkx as nx
import torch
import shutil
from Utils import load_graph
from GenerateTrainData import make_dataset, Generate_Node_Feature3
from torch_geometric.data import DataLoader
from Train import TrainDataset, train
from Model import GDIM

def seed(graph,seed_size, g_features, g_adjacent_matrix, model, topk):

    result_index = []
    k = seed_size
    seed = np.zeros((graph.number_of_nodes(),), dtype=int)
    model.load_state_dict(torch.load('.\\model_save\\model.pt'))
    print('predicting')
    for i in range(k):
        data =  Generate_Node_Feature3(graph, g_features, g_adjacent_matrix, seed, topk)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = model.to(device)
        data = data.to(device)
        model.eval()
        out = model(data).view(-1).to('cpu')
        node_index = np.argmax(out.detach().numpy())
        seed[node_index] = 1
        result_index.append(node_index)

    result = [list(graph.nodes)[i] for i in result_index]

    return result



if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--train_graph_path', default='./train_sy_network/Train_100_4.txt', type=str, help='path')
    parser.add_argument('--test_graph_path', default='./train_sy_network/Train_100_4.txt', type=str, help='path')
    parser.add_argument('--epochs', default=200, type=int, help='training epochs')
    parser.add_argument('--train_gdc', default=20, type=int)
    parser.add_argument('--test_gdc', default=20, type=int)
    parser.add_argument('--seed_size', default=20, type=int)
    parser.add_argument('--random_seed', default=1234, type=int)

    args = parser.parse_args()
    torch.manual_seed(args.random_seed)
    torch.cuda.manual_seed_all(args.random_seed)
    np.random.seed(args.random_seed)

    #导入训练数据 制作数据集

    train_graph = load_graph(args.train_graph_path)
    ad_matrix = np.array(nx.adjacency_matrix(train_graph).todense())
    feature = Generate_Node_Feature3(train_graph)
    make_dataset(train_graph, feature, ad_matrix)
    shutil.rmtree('./SYN_Dataset')

    #生成dataloader
    print('-----------------new dataset creating---------------')
    tr_gdc = args.train_gdc
    TrainSets = TrainDataset(tr_gdc)
    train_dataset, test_dataset = TrainSets.CreateDataset()
    train_loader = DataLoader(train_dataset, batch_size=20, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    # #训练网络
    print('-----------------Training---------------')

    model = GDIM()

    #
    train(model, args.epochs, train_loader, test_loader)

    #测试网络
    print('-----------------Testing---------------')
    te_gdc = args.test_gdc
    test_graph = load_graph(args.test_graph_path)
    seed_size = args.seed_size
    ad_matrix_test = np.array(nx.adjacency_matrix(test_graph).todense())
    feature_test = Generate_Node_Feature3(test_graph)

    model_test = GDIM()
    seed_set = seed(test_graph, seed_size, feature_test, ad_matrix_test, model_test, topk=te_gdc)
    print('the solution set is:', seed_set)









