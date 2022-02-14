"""
Model-Agnostic Augmentation for Accurate Graph Classification (WWW 2022)

Authors:
- Jaemin Yoo (jaeminyoo@snu.ac.kr), Seoul National University
- Sooyeon Shim (syshim77@snu.ac.kr), Seoul National University
- U Kang (ukang@snu.ac.kr), Seoul National University

This software may be used only for research evaluation purposes.
For other purposes (e.g., commercial), please contact the authors.
"""
import math

import torch
import numpy as np
from torch_geometric.data import Data
from torch_geometric import utils as pyg_utils

from augment import utils


def split(array, p=0.5):
    if isinstance(array, list):
        array = np.array(array)
    elif isinstance(array, np.ndarray):
        array = array.copy()
    else:
        raise ValueError()
    np.random.shuffle(array)
    n = np.random.binomial(len(array), p)
    return array[:n], array[n:]


def find_triangles(nn_sets, node):
    tri_counts = 0
    tri_nodes = set()
    for n1 in nn_sets[node]:
        for n2 in nn_sets[node].intersection(nn_sets[n1]):
            if n1 < n2:
                tri_counts += 1
                tri_nodes.add(n1)
                tri_nodes.add(n2)
    return tri_counts, list(tri_nodes)


def split_with_adjustment(graph, tri_count, tri_nodes, neighbors):
    if tri_count == 0:
        return split(neighbors)
    else:
        v = graph.num_nodes
        e = graph.num_edges // 2
        t = tri_count
        d = len(neighbors)
        c = e - 2 - 3 * t / d
        n = (math.sqrt(c ** 2 + 4 * t * v - 6 * t) - c) / 2

        tri_nodes, _ = split(tri_nodes, p=min(n / len(tri_nodes), 1))
        nset1, nset2 = split(np.setdiff1d(neighbors, tri_nodes, assume_unique=True))
        nset1 = np.concatenate([nset1, tri_nodes])
        nset2 = np.concatenate([nset2, tri_nodes])
        return nset1, nset2


class SplitNode(object):
    def __init__(self, graphs, adjustment=False):
        super().__init__()
        self.adjustment = adjustment
        self.graphs = graphs
        self.nn_lists = [utils.get_neighbors(g) for g in graphs]
        self.nn_sets = [[set(e) for e in nn] for nn in self.nn_lists]
        self.di_edges = [utils.to_directed(g.edge_index) for g in graphs]

    def __call__(self, index):
        graph = self.graphs[index]
        di_edges = self.di_edges[index]
        old_node = np.random.randint(graph.num_nodes)
        new_node = graph.num_nodes

        nn_sets = self.nn_sets[index]
        neighbors = self.nn_lists[index][old_node]
        if self.adjustment:
            tri_count, tri_nodes = find_triangles(nn_sets, old_node)
            nset1, nset2 = split_with_adjustment(
                graph, tri_count, tri_nodes, neighbors)
        else:
            nset1, nset2 = split(neighbors)

        new_edges = [[old_node, new_node]]
        new_edges.extend([n, old_node] for n in nset1)
        new_edges.extend([n, new_node] for n in nset2)
        new_edges = torch.tensor(new_edges, dtype=torch.long).t()

        edge_index = di_edges[:, (di_edges != old_node).all(0)]
        edge_index = torch.cat([edge_index, new_edges], dim=1)
        edge_index = pyg_utils.to_undirected(edge_index)

        new_x = torch.cat([graph.x, graph.x[old_node].view(1, -1)])
        return Data(x=new_x, edge_index=edge_index, y=graph.y)
