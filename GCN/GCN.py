import math

import torch

from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
import scipy.sparse as sp

import torch.optim as optim

import redis
import numpy as np


class GraphConvolution(Module):

    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    # 声明初始化 nn.Module 类里面的W,b参数
    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        # 利用矩阵乘法计算输入和权重的乘积
        support = torch.mm(input, self.weight)
        # 利用稀疏矩阵乘法计算邻接矩阵和输入与权重乘积的乘积
        output = torch.spmm(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
            + str(self.in_features) + ' -> ' \
            + str(self.out_features) + ')'


import torch.nn as nn
import torch.nn.functional as F


class GCN(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout):
        super(GCN, self).__init__()

        self.gc1 = GraphConvolution(nfeat, nhid)
        self.gc2 = GraphConvolution(nhid, nclass)
        self.dropout = dropout

    def forward(self, x, y, adj):
        x = F.relu(self.gc1(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc2(x, adj)

        y = F.relu(self.gc1(y, adj))
        y = F.dropout(y, self.dropout, training=self.training)
        y = self.gc2(y, adj)

        z = torch.concat([x, y], 1)
        return F.log_softmax(z, dim=1)


epochs = 100  # Number of epochs to train
lr = 0.005  # Initial learning rate.
weight_decay = 1e-4  # Weight decay (L2 loss on parameters)
hidden = 16  # Number of hidden units.
nb_heads = 8  # Number of head attentions.
dropout = 0.6  # Dropout rate (1 - keep probability).
alpha = 0.2  # Alpha for the leaky_relu.
patience = 100  # Patience
fastmode = False  # Validate during training pass


def main():
    rdb = redis.Redis(host='210.30.96.102', port=31393, db=3, password='dragonfly')

    # 构建特征向量
    data = rdb.lrange("id", 0, -1)
    id = [i.decode('utf-8') for i in data]

    id_map = {j: i for i, j in enumerate(id)}
    print(id_map)

    edges = []
    key = "scheduler:network-topology:*"
    keys = rdb.keys(key)
    node_num = len(id)
    adj = np.zeros((node_num, node_num))
    for key in keys:
        key = key.decode('utf-8')
        c = key.split(':')
        src_id = c[2]
        dest_id = c[3]

        adj[id_map[src_id]][id_map[dest_id]] = 1
        adj[id_map[dest_id]][id_map[src_id]] = 1
        edges.append([id_map[src_id], id_map[dest_id]])

    edges1 = np.array(edges)
    print(edges1)

    print(adj)

    # adj, features, labels, idx_train, idx_val, idx_test = load_cora()
    #
    # features = torch.FloatTensor(np.random.rand(2708, 128))
    #
    # nfeat = node_num
    # nhid = 16
    # nclass = 7
    # dropout = 0.6
    # model = GCN(nfeat=features.shape[1],
    #             nhid=hidden,
    #             nclass=1,
    #             dropout=dropout)
    #
    # optimizer = optim.Adam(model.parameters(),
    #                        lr=lr,
    #                        weight_decay=weight_decay)
    #
    # while True:
    #     output = model(features, adj)

    # # Train model
    # t_total = time.time()
    # loss_values = []
    # bad_counter = 0
    # best = epochs + 1
    # best_epoch = 0
    # for epoch in range(epochs):
    #     t = time.time()
    #     model.train()
    #     optimizer.zero_grad()
    #     output = model(features, adj)
    #     loss_train = F.nll_loss(output[idx_train], labels[idx_train])
    #     acc_train = accuracy(output[idx_train], labels[idx_train])
    #     loss_train.backward()
    #     optimizer.step()
    #
    #     if not fastmode:
    #         # Evaluate validation set performance separately,
    #         # deactivates dropout during validation run.
    #         model.eval()
    #         output = model(features, adj)
    #
    #     loss_val = F.nll_loss(output[idx_val], labels[idx_val])
    #     acc_val = accuracy(output[idx_val], labels[idx_val])
    #     print('Epoch: {:04d}'.format(epoch + 1),
    #           'loss_train: {:.4f}'.format(loss_train.data.item()),
    #           'acc_train: {:.4f}'.format(acc_train.data.item()),
    #           'loss_val: {:.4f}'.format(loss_val.data.item()),
    #           'acc_val: {:.4f}'.format(acc_val.data.item()),
    #           'time: {:.4f}s'.format(time.time() - t))
    #
    #     loss_values.append(loss_val.data.item())
    #
    #     # torch.save(model.state_dict(), '{}.pkl'.format(epoch))
    #     # if loss_values[-1] < best:
    #     #     best = loss_values[-1]
    #     #     best_epoch = epoch
    #     #     bad_counter = 0
    #     # else:
    #     #     bad_counter += 1
    #     #
    #     # if bad_counter == patience:
    #     #     break
    #
    # # files = glob.glob('*.pkl')
    # # for file in files:
    # #     epoch_nb = int(file.split('.')[0])
    # #     if epoch_nb > best_epoch:
    # #         os.remove(file)
    #
    # # testing
    # model.eval()
    # output = model(features, adj)
    # loss_test = F.nll_loss(output[idx_test], labels[idx_test])
    # acc_test = accuracy(output[idx_test], labels[idx_test])
    # print("Test set results:",
    #       "loss= {:.4f}".format(loss_test.data.item()),
    #       "accuracy= {:.4f}".format(acc_test.data.item()))


if __name__ == "__main__":
    main()
