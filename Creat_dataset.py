# -*- coding: utf-8 -*-
from Utils import pickle_read
from Utils import  pickle_save
import torch
import numpy as np
from torch_geometric.data import InMemoryDataset
from torch_geometric.data import Data
import torch_geometric.transforms as T


class Creat_dataset(InMemoryDataset):

    def __init__(self, root, seed, Adj, node_feature, node_label, topk=20, transform=None, pre_transform=None):

        self.seed = np.array(pickle_read(seed), dtype=float)
        self.Adj = np.array(pickle_read(Adj), dtype=object)
        self.node_feature = np.array(pickle_read(node_feature), dtype=float)
        self.graph_label_lpa_r = np.array(pickle_read(node_label), dtype=float)
        self.num_graph = len(self.graph_label_lpa_r)
        self.topk = topk
        super(Creat_dataset, self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return []

    @property
    def processed_file_names(self):
        return [r'.\Synthetic.dataset']

    def download(self):
        pass

    def process(self):
        data_list = []  # graph classification need to define data_list for multiple graph
        for i in range(self.num_graph):

            feature_x = self.node_feature[i]
            # feature_x[np.isnan(feature_x)] = 0
            x = torch.tensor(feature_x, dtype=torch.float)

            feature_seed = self.seed[i]
            s = torch.tensor(feature_seed, dtype=torch.float)

            label_y = self.graph_label_lpa_r[i]
            # label_y[np.isnan(label_y)] = 0
            y = torch.tensor(label_y, dtype=torch.float)


            # 非转置
            source_nodes, target_nodes = np.nonzero(self.Adj[i])
            # edge weight
            Weight = []
            for j, k in zip(source_nodes, target_nodes):
                Weight.append(float(self.Adj[i][j][k]))
            edge_weight = torch.tensor(Weight, dtype = torch.float)
            source_nodes = source_nodes.reshape((1, -1))
            target_nodes = target_nodes.reshape((1, -1))
            edge_index = torch.tensor(np.concatenate((source_nodes, target_nodes), axis=0),
                                      dtype=torch.long)  # edge_index should be long type

            data = Data(x=x, s=s, edge_index=edge_index, edge_attr=edge_weight, y=y)
            transform = T.GDC(
                self_loop_weight=0,
                normalization_in='col',
                normalization_out='col',
                diffusion_kwargs=dict(method='ppr', alpha=0.05),
                sparsification_kwargs=dict(method='topk', k=self.topk, dim=0),
                exact=True,
            )
            data = transform(data)
            data_list.append(data)

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])