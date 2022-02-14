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
from torch_geometric.transforms import GDC
from torch_geometric.utils import subgraph

from augment import utils
from augment.submix import PPR


def make_subgraphs(graphs, prob):
    neighbors = []
    for graph in graphs:
        k = int(graph.num_nodes * prob)
        diffuse = GDC(normalization_out=None,
                      sparsification_kwargs=dict(method='topk', dim=0, k=k))
        new_graph = Data(graph.x, graph.edge_index, y=graph.y)
        diffuse(new_graph)
        n_list = utils.get_neighbors(new_graph)
        neighbors.append([torch.tensor(n) for n in n_list])
    return neighbors


def make_subgraph(graph, root, max_iters=10):
    diffuse = PPR()
    out = torch.zeros(graph.num_nodes)
    out[root] = 1.0
    for _ in range(max_iters):
        out = diffuse(out, graph.edge_index)
    return out.argsort(descending=True)


class GraphCrop(object):
    def __init__(self, graphs, prob=0.7, cached=True):
        super().__init__()
        self.graphs = graphs
        self.cached = cached
        self.prob = prob
        if cached:
            self.neighbors = make_subgraphs(graphs, prob)

    def __call__(self, index):
        graph = self.graphs[index]
        root = torch.randint(graph.num_nodes, size=(1,))
        if self.cached:
            nodes = self.neighbors[index][root]
        else:
            target_size = int(graph.num_nodes * self.prob)
            nodes = make_subgraph(graph, root)[:target_size]
        edge_index, _ = subgraph(nodes, graph.edge_index,
                                 relabel_nodes=True,
                                 num_nodes=graph.num_nodes)
        return Data(graph.x[nodes], edge_index, y=graph.y)
