"""
Model-Agnostic Augmentation for Accurate Graph Classification (WWW 2022)

Authors:
- Jaemin Yoo (jaeminyoo@snu.ac.kr), Seoul National University
- Sooyeon Shim (syshim77@snu.ac.kr), Seoul National University
- U Kang (ukang@snu.ac.kr), Seoul National University

This software may be used only for research evaluation purposes.
For other purposes (e.g., commercial), please contact the authors.
"""
import torch
from torch.nn import functional as func
from torch import nn
from torch_geometric import nn as gnn


class MLP(nn.Module):
    def __init__(self, num_features, num_classes, hidden_units=32, num_layers=1):
        super(MLP, self).__init__()
        if num_layers == 1:
            self.layers = nn.Linear(num_features, num_classes)
        elif num_layers > 1:
            layers = [nn.Linear(num_features, hidden_units),
                      nn.BatchNorm1d(hidden_units),
                      nn.ReLU()]
            for _ in range(num_layers - 2):
                layers.extend([nn.Linear(hidden_units, hidden_units),
                               nn.BatchNorm1d(hidden_units),
                               nn.ReLU()])
            layers.append(nn.Linear(hidden_units, num_classes))
            self.layers = nn.Sequential(*layers)
        else:
            raise ValueError()

    def forward(self, x):
        return self.layers(x)


class GIN(nn.Module):
    def __init__(self, num_features, num_classes, hidden_units=32, num_layers=3, dropout=0.15,
                 mlp_layers=2, train_eps=False):
        super(GIN, self).__init__()
        convs, bns = [], []
        linears = [nn.Linear(num_features, num_classes)]
        for i in range(num_layers - 1):
            input_dim = num_features if i == 0 else hidden_units
            convs.append(gnn.GINConv(MLP(input_dim, hidden_units, hidden_units, mlp_layers),
                                     train_eps=train_eps))
            bns.append(nn.BatchNorm1d(hidden_units))
            linears.append(nn.Linear(hidden_units, num_classes))
        self.convs = nn.ModuleList(convs)
        self.bns = nn.ModuleList(bns)
        self.linears = nn.ModuleList(linears)
        self.num_layers = num_layers
        self.dropout = dropout

    def forward(self, x, edge_index, batch):
        h_list = [x]
        for conv, bn in zip(self.convs, self.bns):
            h = conv(h_list[-1], edge_index)
            h_list.append(torch.relu(bn(h)))
        out = 0
        for i in range(self.num_layers):
            h_pooled = gnn.global_add_pool(h_list[i], batch)
            h_pooled = self.linears[i](h_pooled)
            out += func.dropout(h_pooled, self.dropout, self.training)
        return out
