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

from augment import utils


class NodeAug(object):
    def __init__(self, graphs, prob=0.3, max_levels=3):
        super().__init__()
        self.graphs = graphs
        self.degrees, self.neighbors, self.directed_edges = [], [], []
        for graph in graphs:
            self.degrees.append(utils.get_degrees(graph))
            self.neighbors.append(utils.get_neighbors(graph))
            self.directed_edges.append(utils.to_directed(graph.edge_index))
        self.prob = prob
        self.max_levels = max_levels

    def get_node_levels(self, index, node):
        num_nodes = self.graphs[index].num_nodes
        neighbors = self.neighbors[index]
        levels = np.full(num_nodes, self.max_levels + 1, dtype=np.int64)
        levels[node] = 0
        curr_nodes = [node]
        for level in range(self.max_levels):
            nn_list = [neighbors[n] for n in curr_nodes]
            curr_nodes = np.unique(np.concatenate(nn_list))
            if len(curr_nodes) == 0:
                break
            levels[curr_nodes] = np.minimum(levels[curr_nodes], level + 1)
        return levels

    def get_edge_scores(self, index, node_levels):
        edges = self.directed_edges[index]
        lower_idx = node_levels[edges].argmax(axis=0)
        lower_nodes = edges[lower_idx, torch.arange(edges.size(1))]
        return torch.log(self.degrees[index][lower_nodes])

    def remove_edges(self, index, node_levels, min_levels=1):
        edges = self.directed_edges[index]
        edge_scores = self.get_edge_scores(index, node_levels)
        edge_levels = node_levels[edges].max(axis=0)
        probs = torch.zeros(edges.size(1))
        for level in range(min_levels, self.max_levels + 1):
            targets = edge_levels == level
            if targets.sum() > 0:
                s_max = edge_scores[targets].max()
                s_avg = edge_scores[targets].mean()
                if s_max == s_avg:  # Not described in the paper.
                    s_max = 2 * s_avg
                s_arr = (s_max - edge_scores[targets]) / (s_max - s_avg)
                probs[targets] = np.minimum(self.prob * level * s_arr, 1)
        return edges[:, torch.rand(edges.size(1)) > probs]

    def add_edges(self, index, node, node_levels, min_levels=2):
        node_scores = self.degrees[index].log()
        probs = torch.zeros(len(node_scores))
        for level in range(min_levels, self.max_levels + 1):
            targets = node_levels == level
            if targets.sum() > 0:
                s_min = node_scores[targets].min()
                s_avg = node_scores[targets].mean()
                if s_min == s_avg:  # Not described in the paper.
                    s_min = 0
                s_arr = (node_scores[targets] - s_min) / (s_avg - s_min)
                probs[targets] = np.minimum(self.prob / level * s_arr, 1)

        new_edges = []
        for n in torch.nonzero(torch.rand(size=(len(node_scores),)) < probs):
            new_edges.append(sorted([node, n]))
        return torch.tensor(new_edges, dtype=torch.int64).t()

    def __call__(self, index):
        graph = self.graphs[index]
        node = np.random.randint(graph.num_nodes)
        node_levels = self.get_node_levels(index, node)
        edges1 = self.remove_edges(index, node_levels)
        edges2 = self.add_edges(index, node, node_levels)
        edge_index = utils.to_undirected(torch.cat([edges1, edges2], dim=1))
        return Data(graph.x, edge_index, y=graph.y)
