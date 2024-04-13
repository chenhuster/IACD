# -*- coding: utf-8 -*-

import numpy as np
import torch
from torch.nn import Linear
import torch.nn as nn
from torch_sparse import SparseTensor, matmul, fill_diag, sum, mul

class GDIM(torch.nn.Module):
    def __init__(self):
        super(GDIM, self).__init__()

        self.ap = torch.nn.Sequential(torch.nn.Linear(1, 128),
                                      torch.nn.ReLU(),
                                      # torch.nn.Dropout(0.2),
                                      torch.nn.Linear(128, 1))
        self.ol = torch.nn.Sequential(torch.nn.Linear(1, 128),
                                      torch.nn.ReLU(),
                                      # torch.nn.Dropout(0.2),
                                      torch.nn.Linear(128, 1))
        self.es = torch.nn.Sequential(torch.nn.Linear(5, 128),
                                      torch.nn.ReLU(),
                                      # torch.nn.Dropout(0.2),
                                      torch.nn.Linear(128, 1))

    def forward(self, data):
        x, seed, edge_index, edge_weight = data.x, data.s, data.edge_index, data.edge_attr
        adj_matrix = SparseTensor(row=edge_index[0], col=edge_index[1], value=edge_weight)
        adj = adj_matrix.to_dense()
        seed_idx = torch.LongTensor(np.argwhere(seed.detach().cpu().numpy()[:, 0] == 1)).to('cuda')

        s1 = torch.mm(adj.t(), seed)
        s1 = self.ap(s1)

        onehot = torch.eye(adj.size(0)).to('cuda')
        s_matrix = torch.mm(adj, onehot)
        seed_vector = torch.zeros((1, adj.size(0))).to('cuda')
        for seed_node in seed_idx:
            seed_vector += s_matrix[seed_node]
        o1 = torch.mm(s_matrix, seed_vector.t())
        s2 = self.ol(o1)

        s3 = self.es(x)

        result = s1 + s2 + s3

        result[seed_idx] = 0

        return result






