"""
Model-Agnostic Augmentation for Accurate Graph Classification (WWW 2022)

Authors:
- Jaemin Yoo (jaeminyoo@snu.ac.kr), Seoul National University
- Sooyeon Shim (syshim77@snu.ac.kr), Seoul National University
- U Kang (ukang@snu.ac.kr), Seoul National University

This software may be used only for research evaluation purposes.
For other purposes (e.g., commercial), please contact the authors.
"""
from augment.merge import MergeDirect
from augment.split import SplitNode


class NodeSam(object):
    def __init__(self, graphs, adjustment=True):
        self.split = SplitNode(graphs, adjustment=adjustment)
        self.merge = MergeDirect()

    def __call__(self, index):
        return self.merge(self.split(index))


class NodeSamBase(NodeSam):
    def __init__(self, index):
        super().__init__(index, adjustment=False)
