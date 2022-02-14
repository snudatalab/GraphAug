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
import json
import math

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import f1_score, accuracy_score
from torch_geometric.data import DataLoader

from augment import Augment
from data import load_data_fold, load_data
from gin import GIN
from utils import str2bool


class SoftCELoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.ce_loss = nn.CrossEntropyLoss()
        self.kl_loss = nn.KLDivLoss(reduction='batchmean')
        self.log_softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, target):
        if target.ndim == 1:
            return self.ce_loss(input, target)
        elif input.size() == target.size():
            input = self.log_softmax(input)
            return self.kl_loss(input, target)
        else:
            raise ValueError()


def parse_args():
    parser = argparse.ArgumentParser()

    # Data
    parser.add_argument('--data', type=str, default='MUTAG')
    parser.add_argument('--degree-x', type=str2bool, default=True)
    parser.add_argument('--fold', type=int, default=0)
    parser.add_argument('--seed', type=int, default=2021)

    # Experimental setting
    parser.add_argument('--gpu', type=int, default=None)
    parser.add_argument('--verbose', type=int, default=0)
    parser.add_argument('--print-all', action='store_true', default=False)
    parser.add_argument('--metric', type=str, default='acc')

    # Augmentation
    parser.add_argument('--augment', type=str, default='none')
    parser.add_argument('--aug-size', type=float, default=0.4)  # SubMix

    # Training setup
    parser.add_argument('--epochs', type=int, default=350)
    parser.add_argument('--iters', type=str, default='50')
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--lr', type=float, default=1e-2)
    parser.add_argument('--decay', type=float, default=0)
    parser.add_argument('--schedule', type=str2bool, default=True)

    # Classifier
    parser.add_argument('--model', type=str, default='GIN')
    parser.add_argument('--units', type=int, default=64)
    parser.add_argument('--layers', type=int, default=5)
    parser.add_argument('--dropout', type=float, default=0.5)

    return parser.parse_args()


def to_device(gpu):
    if gpu is not None and torch.cuda.is_available():
        return torch.device('cuda:{}'.format(gpu))
    else:
        return torch.device('cpu')


@torch.no_grad()
def eval_acc(model, loader, device, metric='acc'):
    model.eval()
    y_true, y_pred = [], []
    for data in loader:
        data = data.to(device)
        output = model(data.x, data.edge_index, data.batch)
        y_pred.append(output.argmax(dim=1).cpu())
        y_true.append(data.y.cpu())
    y_true = torch.cat(y_true).numpy()
    y_pred = torch.cat(y_pred).numpy()
    if metric == 'acc':
        return accuracy_score(y_true, y_pred)
    elif metric == 'f1':
        return f1_score(y_true, y_pred, average='macro')
    else:
        raise ValueError(metric)


@torch.no_grad()
def eval_loss(model, loss_func, loader, device):
    model.eval()
    count_sum, loss_sum = 0, 0
    for data in loader:
        data = data.to(device)
        output = model(data.x, data.edge_index, data.batch)
        loss = loss_func(output, data.y).item()
        loss_sum += loss * len(data.y)
        count_sum += len(data.y)
    return loss_sum / count_sum


def main():
    args = parse_args()
    if args.augment.lower() == 'none':
        args.augment = None
    device = to_device(args.gpu)

    args.seed = args.seed + args.fold
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    data = load_data(args.data, args.degree_x)
    num_features = data.num_features
    num_classes = data.num_classes

    trn_graphs, test_graphs = load_data_fold(args.data, args.fold, args.degree_x)
    trn_loader = DataLoader(trn_graphs, batch_size=args.batch_size)
    test_loader = DataLoader(test_graphs, batch_size=args.batch_size)

    if args.iters == 'auto':
        args.iters = math.ceil(len(trn_graphs) / args.batch_size)
    else:
        args.iters = int(args.iters)

    model = GIN(num_features, num_classes, args.units, args.layers, args.dropout)
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.decay)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)
    loss_func = SoftCELoss()

    augment = Augment(trn_graphs, args.augment, aug_size=args.aug_size)

    if args.verbose > 0:
        print(' epochs\t   loss\ttrn_acc\tval_acc')

    out_list = dict(trn_loss=[], trn_acc=[], test_loss=[], test_acc=[])
    for epoch in range(args.epochs):
        model.train()
        loss_sum = 0
        for _ in range(args.iters):
            idx = torch.randperm(len(trn_graphs))[:args.batch_size]
            data = augment(idx).to(device)
            output = model(data.x, data.edge_index, data.batch)
            loss = loss_func(output, data.y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_sum += loss.item()

        if args.schedule:
            scheduler.step(epoch)

        trn_loss = loss_sum / args.iters
        trn_acc = eval_acc(model, trn_loader, device, args.metric)
        test_loss = eval_loss(model, loss_func, test_loader, device)
        test_acc = eval_acc(model, test_loader, device, args.metric)

        out_list['trn_loss'].append(trn_loss)
        out_list['trn_acc'].append(trn_acc)
        out_list['test_loss'].append(test_loss)
        out_list['test_acc'].append(test_acc)

        if args.verbose > 0 and (epoch + 1) % args.verbose == 0:
            print(f'{epoch + 1:7d}\t{trn_loss:7.4f}\t{trn_acc:7.4f}\t{test_acc:7.4f}')

    if args.print_all:
        out = {arg: getattr(args, arg) for arg in vars(args)}
        out['all'] = out_list
        print(json.dumps(out))
    else:
        print(f'Training accuracy: {out_list["trn_acc"][-1]}')
        print(f'Test accuracy: {out_list["test_acc"][-1]}')


if __name__ == '__main__':
    main()
