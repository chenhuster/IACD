# -*- coding: utf-8 -*-
import os
from Model import *
from Utils import *
from Creat_dataset import *
from torch_geometric.data import DataLoader
import pandas as pd
import matplotlib.pyplot as plt

class TrainDataset:
    def __init__(self, topk=20):

        self.TRAIN_DATA_PATH = os.path.join(os.getcwd(), 'data', 'train', '100_4')
        self.topk = topk

    def ReadTrainFile(self):

        seed = os.path.join(self.TRAIN_DATA_PATH, 'train_dataset_seed.npy')
        feature = os.path.join(self.TRAIN_DATA_PATH, 'train_dataset_feature.npy')
        adj = os.path.join(self.TRAIN_DATA_PATH, 'train_dataset_adj.npy')
        label = os.path.join(self.TRAIN_DATA_PATH, 'train_dataset_label.npy')
        num_graph = len(np.array(pickle_read(feature), dtype=object))
        return seed, feature, adj, label, num_graph

    def CreateDataset(self):

        seed, feature, adj, label, num_graph = self.ReadTrainFile()
        syn_dataset = Creat_dataset(root='./' + 'SYN_Dataset', seed=seed, Adj=adj,
                                       node_feature=feature, node_label=label, topk=self.topk)
        train_dataset = syn_dataset[:round(num_graph * 0.9)]
        test_dataset = syn_dataset[round(num_graph * 0.9):]


        return train_dataset, test_dataset


def train(model, epochs, train_loader, test_loader):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, weight_decay=0.0001)
    criterion = torch.nn.MSELoss(reduction='mean')

    epoch_num = epochs  # 242 0.6889

    for epoch in range(epoch_num):
        model.train()
        for data in train_loader:  # Iterate in batches over the training dataset.
            data = data.to(device)
            out = model(data)
            loss = criterion(out, data.y.view(-1, 1))

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        with torch.no_grad():
            loss_train = test(train_loader, model, device, criterion)
            loss_test = test(test_loader, model, device, criterion)
            print('epoch: %d train_loss: %.4f test_loss:  %.4f' % (epoch, loss_train, loss_test))


    torch.save(model.state_dict(), '.\\model_save\\model.pt')


def test(loader, model, device, criterion):
    model.eval()
    loss = 0
    for data in loader:
        data = data.to(device)
        out = model(data)
        loss += criterion(out, data.y.view(-1, 1))

    return loss.item() / len(loader.dataset)



if __name__ == '__main__':

    TrainSets = TrainDataset(20) #导入处理好的训练网络
    train_dataset, test_dataset = TrainSets.CreateDataset() #生成dataloader
    train_loader = DataLoader(train_dataset, batch_size=20, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    torch.manual_seed(1234)
    torch.cuda.manual_seed_all(1234)
    np.random.seed(1234)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = GDIM().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, weight_decay=0.0001)
    criterion = torch.nn.MSELoss(reduction='mean')

    epoch_num = 100 #
    loss_train_list = []
    loss_test_list = []
    influence_list = []

    for epoch in range(epoch_num):
        model.train()

        for data in train_loader:  # Iterate in batches over the training dataset.
            data = data.to(device)
            out = model(data)
            loss = criterion(out, data.y.view(-1, 1))
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        with torch.no_grad():
            loss_train = test(train_loader, model, device, criterion)
            loss_test = test(test_loader, model, device, criterion)
            print('epoch: %d train_loss: %.4f test_loss:  %.4f' % (epoch, loss_train,  loss_test))
            loss_train_list.append(loss_train)
            loss_test_list.append(loss_test)

    torch.save(model.state_dict(), '.\\model_save\\model.pt')






