"""
Model-Agnostic Augmentation for Accurate Graph Classification (WWW 2022)

Authors:
- Jaemin Yoo (jaeminyoo@snu.ac.kr), Seoul National University
- Sooyeon Shim (syshim77@snu.ac.kr), Seoul National University
- U Kang (ukang@snu.ac.kr), Seoul National University

This software may be used only for research evaluation purposes.
For other purposes (e.g., commercial), please contact the authors.
"""
import numpy as np
import torch
from torch_geometric.data import Data
from torch_geometric.utils import remove_self_loops


def select_nodes(graph):
    target = np.random.randint(graph.num_edges)
    return sorted(graph.edge_index[:, target])


def make_onehot(feature):
    assert (feature >= 0).all()
    selected = np.random.choice(len(feature), p=feature)
    out = torch.zeros_like(feature)
    out[selected] = 1
    return out


def remove_triangles(edge_index, num_nodes, winner, loser):
    row, col = edge_index
    w_neighbors = torch.zeros(num_nodes, dtype=torch.bool)
    w_neighbors[row[col == winner]] = 1
    dup1 = (row == loser) & w_neighbors[col]
    dup2 = (col == loser) & w_neighbors[row]
    return edge_index[:, ~(dup1 | dup2)]


def make_merging_map(num_nodes, winner, loser):
    assert winner < loser
    node_mask = torch.ones(num_nodes, dtype=torch.bool)
    node_mask[loser] = 0
    node_map = torch.zeros(num_nodes, dtype=torch.int64)
    node_map[node_mask] = torch.arange(num_nodes - 1)
    node_map[loser] = winner
    return node_map


class MergeDirect(object):
    def __init__(self, onehot=False):
        super().__init__()
        self.onehot = onehot

    def __call__(self, graph):
        num_nodes = graph.num_nodes
        winner, loser = select_nodes(graph)

        node_map = make_merging_map(num_nodes, winner, loser)
        new_x = torch.zeros(num_nodes - 1, graph.num_features)
        new_x.index_add_(0, node_map, graph.x)
        new_x[winner] /= 2
        if self.onehot:
            new_x[winner] = make_onehot(new_x[winner])

        edge_index = remove_triangles(graph.edge_index, num_nodes, winner, loser)
        edge_index = node_map[edge_index]
        edge_index, _ = remove_self_loops(edge_index)
        return Data(x=new_x, edge_index=edge_index, y=graph.y)


class MergeEdge(object):
    def __init__(self, graphs, onehot=False):
        super().__init__()
        self.graphs = graphs
        self.core = MergeDirect(onehot)

    def __call__(self, index):
        return self.core(self.graphs[index])
