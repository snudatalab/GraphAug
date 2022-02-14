"""
Model-Agnostic Augmentation for Accurate Graph Classification (WWW 2022)

Authors:
- Jaemin Yoo (jaeminyoo@snu.ac.kr), Seoul National University
- Sooyeon Shim (syshim77@snu.ac.kr), Seoul National University
- U Kang (ukang@snu.ac.kr), Seoul National University

This software may be used only for research evaluation purposes.
For other purposes (e.g., commercial), please contact the authors.
"""
import argparse

import numpy as np


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ['true']:
        return True
    elif v.lower() in ['false']:
        return False
    else:
        raise argparse.ArgumentTypeError()


def find_best_epoch(test_loss, test_acc):
    if isinstance(test_loss, np.ndarray) and test_loss.ndim > 1:
        test_loss = test_loss.mean(axis=1)
    if isinstance(test_acc, np.ndarray) and test_acc.ndim > 1:
        test_acc = test_acc.mean(axis=1)

    best_epoch, best_loss, best_acc = -1, np.inf, 0
    for i in range(len(test_acc)):
        if (test_acc[i] > best_acc) or (test_acc[i] == best_acc and test_loss[i] < best_loss):
            best_epoch = i
            best_loss = test_loss[i]
            best_acc = test_acc[i]
    return best_epoch
