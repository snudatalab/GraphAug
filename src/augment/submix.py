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

import numpy as np
import torch
from torch_geometric.data import Data
from torch_geometric.nn import MessagePassing
from torch_geometric.nn.conv.gcn_conv import gcn_norm
from torch_geometric.transforms import GDC
from torch_geometric.utils import to_dense_adj

from augment import utils


class PPR(MessagePassing):
    def __init__(self, self_loops=True, alpha=0.15):
        super().__init__()
        self.alpha = alpha
        self.self_loops = self_loops
        self.edge_index = None

    def forward(self, x, edge_index):
        if self.edge_index is None:
            self.edge_index = gcn_norm(edge_index, add_self_loops=self.self_loops)
        edge_index, edge_weight = self.edge_index
        out = self.propagate(edge_index, x=x.view(-1, 1), edge_weight=edge_weight)
        out = out.view(-1) / out.sum()
        return self.alpha * x + (1 - self.alpha) * out

    def message(self, x_j, edge_weight):
        return edge_weight.view(-1, 1) * x_j


def gather_graphs_by_labels(graphs):
    out = []
    for i, graph in enumerate(graphs):
        y = graph.y.item()
        while y >= len(out):
            out.append([])
        out[y].append(i)
    return out


class GraphSelector(object):
    def __init__(self, graphs, same_label=False):
        self.graphs = graphs
        self.graphs_by_labels = gather_graphs_by_labels(graphs)
        self.same_label = same_label

    def __call__(self, index):
        if self.same_label:
            label = self.graphs[index].y.item()
            candidates = self.graphs_by_labels[label]
        else:
            candidates = np.arange(len(self.graphs))
        candidates = np.delete(candidates, index)
        if len(candidates) < 1:
            raise ValueError('Insufficient graphs')
        return int(np.random.choice(candidates))


class RootSelector(object):
    def __init__(self, graphs, mode='random'):
        if mode not in ['random', 'positional', 'important']:
            raise ValueError(mode)
        self.graphs = graphs
        self.mode = mode

        if mode == 'positional':
            self.degrees = []
            for graph in graphs:
                self.degrees.append(utils.get_degrees(graph))
        elif mode == 'important':
            self.degree_dists = []
            for graph in graphs:
                d = utils.get_degrees(graph).numpy()
                self.degree_dists.append(d / d.sum())

    def get_positional_target(self, idx, position):
        num_nodes = self.graphs[idx].num_nodes
        order = (self.degrees[idx] + torch.rand(num_nodes)).argsort()
        return order[int(position * num_nodes)]

    def __call__(self, idx1, idx2):
        if self.mode == 'random':
            target1 = torch.randint(self.graphs[idx1].num_nodes, size=(1,))
            target2 = torch.randint(self.graphs[idx2].num_nodes, size=(1,))
        elif self.mode == 'positional':
            position = torch.rand(size=(1,))
            target1 = self.get_positional_target(idx1, position)
            target2 = self.get_positional_target(idx2, position)
        elif self.mode == 'important':
            target1 = torch.randint(self.graphs[idx1].num_nodes, size=(1,))
            target2 = np.random.choice(self.graphs[idx2].num_nodes, p=self.degree_dists[idx2])
        else:
            raise ValueError(self.mode)
        return target1, target2


def run_bfs(neighbor_sets, node):
    seen = set()
    next_level = {node}
    while next_level:
        this_level = next_level
        next_level = set()
        for v in this_level:
            if v not in seen:
                seen.add(v)
                next_level.update(neighbor_sets[v])
    return seen


def get_cc_sizes(graph):
    nn_sets = [set(e) for e in utils.get_neighbors(graph)]
    num_nodes = graph.num_nodes
    out = np.zeros(num_nodes, dtype=int)
    seen = set()
    for u in range(num_nodes):
        if u not in seen:
            cc = run_bfs(nn_sets, u)
            seen.update(cc)
            for v in cc:
                out[v] = len(cc)
    return out


def make_subgraph(graph, root, cc_sizes, max_iters=10):
    diffuse = PPR()
    out = torch.zeros(graph.num_nodes)
    out[root] = 1.0
    for _ in range(max_iters):
        out = diffuse(out, graph.edge_index)
    return out.argsort(descending=True)[:cc_sizes[root]]


def make_subgraphs(graphs, norm='sym'):
    if norm not in ['row', 'sym']:
        raise ValueError(norm)
    diffuse = GDC(normalization_in=norm, normalization_out=None,
                  sparsification_kwargs=dict(method='threshold', eps=0))
    subgraphs = []
    for graph in graphs:
        new_graph = Data(graph.x, graph.edge_index, y=graph.y)
        diffuse(new_graph)
        diffused = to_dense_adj(new_graph.edge_index,
                                edge_attr=new_graph.edge_attr).squeeze(0)
        diffused = diffused.argsort(dim=1, descending=True)
        cc_sizes = get_cc_sizes(graph)
        out = [diffused[n, :cc_sizes[n]] for n in range(graph.num_nodes)]
        subgraphs.append(out)
    return subgraphs


def to_mask(num_nodes, subset):
    mask = torch.zeros(num_nodes, dtype=torch.bool)
    mask[subset] = 1
    return mask


def mix_labels(dst_graph, src_graph, weight, num_labels):
    out = torch.zeros(num_labels)
    out[dst_graph.y] += weight
    out[src_graph.y] += 1 - weight
    return out


def mix_graphs(dst_graph, dst_nodes, src_graph, src_nodes, num_labels,
               label_by='edges'):
    node_map = torch.zeros(src_graph.num_nodes, dtype=torch.long)
    node_map[src_nodes] = dst_nodes
    dst_mask = to_mask(dst_graph.num_nodes, dst_nodes)
    src_mask = to_mask(src_graph.num_nodes, src_nodes)

    edges1 = dst_graph.edge_index
    edges1 = edges1[:, ~(dst_mask[edges1[0]] & dst_mask[edges1[1]])]
    edges2 = src_graph.edge_index
    edges2 = node_map[edges2[:, src_mask[edges2[0]] & src_mask[edges2[1]]]]

    new_x = dst_graph.x.clone()
    new_x[dst_nodes] = src_graph.x[src_nodes]
    new_y = torch.zeros(num_labels)
    new_y[dst_graph.y] = 1

    new_edges = torch.cat([edges1, edges2], dim=1)
    if new_edges.size(1) == 0:
        return Data(dst_graph.x, dst_graph.edge_index, y=new_y)

    if label_by == 'nodes':
        ratio = len(dst_nodes) / dst_graph.num_nodes
    elif label_by == 'edges':
        ratio = edges1.size(1) / new_edges.size(1)
    else:
        raise ValueError(label_by)

    new_y[dst_graph.y] *= ratio
    new_y[src_graph.y] += 1 - ratio
    return Data(new_x, new_edges, y=new_y)


class SubMix(object):
    def __init__(self, graphs, aug_size=0.4, root='random', cached=True):
        self.graphs = graphs
        self.num_labels = max(g.y for g in graphs) + 1
        self.aug_size = aug_size
        self.cached = cached

        if cached:
            self.subgraphs = make_subgraphs(graphs, norm='sym')
            self.cc_sizes = []
        else:
            self.cc_sizes = [get_cc_sizes(g) for g in graphs]
            self.subgraphs = []

        if len(graphs) == 1:
            self.graphs *= 2
            self.subgraphs *= 2
            self.cc_sizes *= 2
            graphs = [graphs[0], graphs[0]]

        self.root_selector = RootSelector(graphs, root)
        self.graph_selector = GraphSelector(graphs, same_label=False)

    def get_subgraph(self, index, root):
        if self.cached:
            return self.subgraphs[index][root]
        else:
            return make_subgraph(self.graphs[index], root, self.cc_sizes[index])

    def __call__(self, index):
        target = self.graph_selector(index)
        graph1 = self.graphs[index]
        graph2 = self.graphs[target]
        root1, root2 = self.root_selector(index, target)

        subgraph1 = self.get_subgraph(index, root1)
        subgraph2 = self.get_subgraph(target, root2)
        aug_size = np.random.uniform(high=self.aug_size)
        aug_size = math.ceil(aug_size * min(len(subgraph1), len(subgraph2)))
        subgraph1 = subgraph1[:aug_size]
        subgraph2 = subgraph2[:aug_size]

        out = mix_graphs(graph1, subgraph1, graph2, subgraph2,
                         self.num_labels, label_by='edges')
        out.y = out.y.unsqueeze(0)
        return out


class SubMixBase(object):
    def __init__(self, graphs, aug_size=0.4):
        self.graphs = graphs
        self.num_labels = max(g.y for g in graphs) + 1
        self.aug_size = aug_size
        self.graph_selector = GraphSelector(graphs, same_label=False)

    def __call__(self, index):
        target = self.graph_selector(index)
        graph1 = self.graphs[index]
        graph2 = self.graphs[target]

        aug_size = np.random.uniform(high=self.aug_size)
        aug_size = math.ceil(aug_size * min(graph1.num_nodes, graph2.num_nodes))
        subgraph1 = torch.randperm(graph1.num_nodes)[:aug_size]
        subgraph2 = torch.randperm(graph2.num_nodes)[:aug_size]

        out = mix_graphs(graph1, subgraph1, graph2, subgraph2,
                         self.num_labels, label_by='edges')
        out.y = out.y.unsqueeze(0)
        return out
