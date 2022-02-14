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
from torch_geometric.data import Data
from torch_geometric.utils import subgraph

from augment import utils


class DropEdge(object):
    def __init__(self, graphs):
        super().__init__()
        self.graphs = graphs
        self.di_edges = [utils.to_directed(g.edge_index) for g in graphs]

    def __call__(self, index):
        graph = self.graphs[index]
        di_edges = self.di_edges[index]
        num_edges = di_edges.size(1)
        target = np.random.randint(num_edges)
        di_edges = di_edges[:, torch.arange(num_edges) != target]
        edge_index = utils.to_undirected(di_edges)
        return Data(x=graph.x, edge_index=edge_index, y=graph.y)


class DropNode(object):
    def __init__(self, graphs):
        super().__init__()
        self.graphs = graphs

    def __call__(self, index):
        graph = self.graphs[index]
        target = np.random.randint(graph.num_nodes)
        node_mask = torch.ones(graph.num_nodes, dtype=torch.bool)
        node_mask[target] = 0
        node_map = torch.zeros(graph.num_nodes, dtype=torch.int64)
        node_map[node_mask] = torch.arange(graph.num_nodes - 1)
        edge_index, _ = subgraph(node_mask, graph.edge_index, num_nodes=graph.num_nodes)
        edge_index = node_map[edge_index]
        return Data(x=graph.x[node_mask], edge_index=edge_index, y=graph.y)


class ChangeAttr(object):
    def __init__(self, graphs):
        super().__init__()
        self.graphs = graphs

    def __call__(self, index):
        graph = self.graphs[index]
        target = np.random.randint(graph.num_nodes)
        old_attr = graph.x[target].argmax(dim=0)
        new_attr = np.random.choice(np.delete(np.arange(graph.num_features), old_attr))
        new_x = graph.x.clone()
        new_x[target] = 0
        new_x[target, new_attr] = 1
        return Data(x=new_x, edge_index=graph.edge_index, y=graph.y)


class AddEdge(object):
    def __init__(self, graphs, target='node'):
        super().__init__()
        self.graphs = graphs
        self.target = target
        self.candidates = []
        for graph in graphs:
            candidates = []
            all_nodes = np.arange(graph.num_nodes)
            neighbors = utils.get_neighbors(graph)
            for node, n_list in enumerate(neighbors):
                n_rest = np.delete(all_nodes, n_list)
                pairs = [[node, e] for e in n_rest[n_rest > node]]
                candidates.extend(pairs)
            self.candidates.append(np.array(candidates))

    def __call__(self, index):
        graph = self.graphs[index]
        candidates = self.candidates[index]
        if len(candidates) == 0:  # The graph is a clique.
            return graph
        n1, n2 = candidates[np.random.randint(len(candidates))]
        new_edges = torch.tensor([[n1, n2], [n2, n1]])
        edge_index = torch.cat([graph.edge_index, new_edges], dim=1)
        return Data(x=graph.x, edge_index=edge_index, y=graph.y)
