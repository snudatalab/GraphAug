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
import numpy as np
from torch_geometric.utils import degree


def get_degrees(graph):
    return degree(graph.edge_index[0], num_nodes=graph.num_nodes)


def to_directed(edge_index):
    return edge_index[:, edge_index[0] < edge_index[1]]


def to_undirected(edge_index):
    assert (edge_index[0] == edge_index[1]).sum() == 0  # no self-loops
    return torch.cat([edge_index, edge_index.flip([0])], dim=1)


def get_neighbors(graph):
    # Assume the edge_index is undirected and sorted.
    num_nodes = graph.num_nodes
    edge_index = graph.edge_index.numpy()

    diff = np.diff(np.insert(edge_index[0, :], 0, 0))
    start_index = np.full(num_nodes, fill_value=-1, dtype=np.int64)
    start_index[(diff[diff > 0]).cumsum()] = np.nonzero(diff)[0]
    start_index[0] = 0
    for i in range(num_nodes - 1, 0, -1):
        if start_index[i] < 0:
            if i == num_nodes - 1:
                start_index[i] = num_nodes
            else:
                start_index[i] = start_index[i + 1]

    neighbors = []
    for node in range(num_nodes):
        if node < num_nodes - 1:
            end_index = start_index[node + 1]
        else:
            end_index = edge_index.shape[1]
        neighbors.append(edge_index[1, start_index[node]:end_index])
    return neighbors
