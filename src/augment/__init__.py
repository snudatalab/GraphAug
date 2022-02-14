"""
Model-Agnostic Augmentation for Accurate Graph Classification (WWW 2022)

Authors:
- Jaemin Yoo (jaeminyoo@snu.ac.kr), Seoul National University
- Sooyeon Shim (syshim77@snu.ac.kr), Seoul National University
- U Kang (ukang@snu.ac.kr), Seoul National University

This software may be used only for research evaluation purposes.
For other purposes (e.g., commercial), please contact the authors.
"""
import inspect
from torch_geometric.data import Batch

from augment.merge import MergeEdge
from augment.split import SplitNode
from augment.submix import SubMix, SubMixBase
from augment.nodesam import NodeSam, NodeSamBase

from augment.baselines.motif import MotifSwap
from augment.baselines.simple import DropEdge, DropNode, ChangeAttr, AddEdge
from augment.baselines.nodeaug import NodeAug
from augment.baselines.graphcrop import GraphCrop


class Augment(object):
    def __init__(self, graphs, method=None, **kwargs):
        super().__init__()
        self.graphs = graphs
        if method is None:
            self.model = None
        else:
            model_class = eval(method)
            parameters = inspect.signature(model_class).parameters
            args = {k: v for k, v in kwargs.items() if k in parameters}
            self.model = model_class(graphs, **args)

    def augment(self, indices):
        if self.model is None:
            return [self.graphs[i] for i in indices]
        else:
            return [self.model(i) for i in indices]

    def __call__(self, indices, as_batch=True):
        data = self.augment(indices)
        if as_batch:
            data = Batch().from_data_list(data)
        return data
