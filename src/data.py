"""
Model-Agnostic Augmentation for Accurate Graph Classification (WWW 2022)

Authors:
- Jaemin Yoo (jaeminyoo@snu.ac.kr), Seoul National University
- Sooyeon Shim (syshim77@snu.ac.kr), Seoul National University
- U Kang (ukang@snu.ac.kr), Seoul National University

This software may be used only for research evaluation purposes.
For other purposes (e.g., commercial), please contact the authors.
"""
import json
import os

import torch
from sklearn.model_selection import StratifiedKFold
from torch_geometric.datasets import TUDataset
import numpy as np
import networkx as nx
from torch_geometric.utils import contains_self_loops, contains_isolated_nodes, \
    is_undirected, to_networkx, degree

ROOT = '../data'
DATASETS = ['DD', 'ENZYMES', 'MUTAG', 'NCI1', 'NCI109', 'PROTEINS', 'PTC_MR',
            'COLLAB', 'Twitter']


def to_degree_features(data):
    d_list = []
    for graph in data:
        d_list.append(degree(graph.edge_index[0], num_nodes=graph.num_nodes))
    x = torch.cat(d_list).long()
    unique_degrees = torch.unique(x)
    mapper = torch.full_like(x, fill_value=1000000000)
    mapper[unique_degrees] = torch.arange(len(unique_degrees))
    x_onehot = torch.zeros(x.size(0), len(unique_degrees))
    x_onehot[torch.arange(x.size(0)), mapper[x]] = 1
    return x_onehot


def load_data(dataset, degree_x=True):
    if dataset == 'Twitter':
        dataset = 'TWITTER-Real-Graph-Partial'
    data = TUDataset(root=os.path.join(ROOT, 'graphs'), name=dataset,
                     use_node_attr=False)
    data.data.edge_attr = None
    if data.num_node_features == 0:
        data.slices['x'] = torch.tensor([0] + data.data.num_nodes).cumsum(0)
        if degree_x:
            data.data.x = to_degree_features(data)
        else:
            num_all_nodes = sum(g.num_nodes for g in data)
            data.data.x = torch.ones((num_all_nodes, 1))
    for i in range(len(data.__data_list__)):
        data.__data_list__[i] = None
    return data


def load_data_fold(dataset, fold, degree_x=True, num_folds=10, seed=0):
    assert 0 <= fold < 10

    data = load_data(dataset, degree_x)
    path = os.path.join(ROOT, 'splits', dataset, f'{fold}.json')
    if not os.path.exists(path):
        skf = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=seed)
        trn_idx, test_idx = list(skf.split(np.zeros(data.len()), data.data.y))[fold]
        trn_idx = [int(e) for e in trn_idx]
        test_idx = [int(e) for e in test_idx]
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'w') as f:
            json.dump(dict(training=trn_idx, test=test_idx), f, indent=4)

    with open(path) as f:
        indices = json.load(f)
    trn_graphs = [data[i] for i in indices['training']]
    test_graphs = [data[i] for i in indices['test']]
    return trn_graphs, test_graphs


def is_connected(graph):
    return nx.is_connected(to_networkx(graph, to_undirected=True))


def print_stats():
    for data in DATASETS:
        out = load_data(data)
        num_graphs = len(out)
        num_nodes = out.data.x.size(0)
        num_edges = out.data.edge_index.size(1) // 2
        num_features = out.num_features
        num_classes = out.num_classes
        print(f'{data}\t{num_graphs}\t{num_nodes}\t{num_edges}\t{num_features}\t'
              f'{num_classes}', end='\t')

        undirected, self_loops, onehot, connected, isolated_nodes = \
            True, False, True, True, False
        for graph in out:
            if not is_undirected(graph.edge_index, num_nodes=graph.num_nodes):
                undirected = False
            if contains_self_loops(graph.edge_index):
                self_loops = True
            if ((graph.x > 0).sum(dim=1) != 1).sum() > 0:
                onehot = False
            if not is_connected(graph):
                connected = False
            if contains_isolated_nodes(graph.edge_index, num_nodes=graph.num_nodes):
                isolated_nodes = True
        print(f'{undirected}\t{self_loops}\t{onehot}\t{connected}\t{isolated_nodes}')


def download():
    for data in DATASETS:
        load_data(data)
        for fold in range(10):
            load_data_fold(data, fold)


if __name__ == '__main__':
    download()
    print_stats()
