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
from torch_geometric.data import Data
import numpy as np
from augment import utils


def find_motifs(graph):
    motifs = []
    neighbors = utils.get_neighbors(graph)
    neighbors = [nn[nn > n].astype(np.int32) for n, nn in enumerate(neighbors)]
    for u, u_neighbors in enumerate(neighbors):
        for v in u_neighbors:
            for t in np.setdiff1d(neighbors[v], u_neighbors):
                motifs.append((u, v, t))  # u < v < t always holds.
    return np.array(motifs, dtype=np.int32)


def search_edge(edge_index, src, dst):
    return torch.nonzero((edge_index[0] == src) & (edge_index[1] == dst), as_tuple=True)[0]


class MotifSwap(object):
    def __init__(self, graphs):
        self.graphs = graphs
        self.motifs = [find_motifs(graph) for graph in graphs]

    def __call__(self, index):
        graph = self.graphs[index]
        motifs = self.motifs[index]

        if len(motifs) == 0:
            return graph

        u1, u2, u3 = motifs[np.random.randint(len(motifs))]
        if np.random.random() < 0.5:
            v1, v2 = u1, u2
        else:
            v1, v2 = u2, u3

        di_edges = utils.to_directed(graph.edge_index)
        target = search_edge(di_edges, v1, v2)
        di_edges[0, target] = int(u1)
        di_edges[1, target] = int(u3)
        edge_index = utils.to_undirected(di_edges)
        return Data(x=graph.x, edge_index=edge_index, y=graph.y)
