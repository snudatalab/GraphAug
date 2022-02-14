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
import itertools
import json
import multiprocessing
import subprocess
from collections import defaultdict
from multiprocessing import Pool

import torch
import tqdm
import numpy as np
import pandas as pd

from data import DATASETS
from utils import find_best_epoch


def parse_args():
    parser = argparse.ArgumentParser()

    # Experimental setup
    parser.add_argument('--data', type=str, nargs='+', default=['all'])
    parser.add_argument('--folds', type=int, nargs='+', default=list(range(10)))
    parser.add_argument('--gpu', type=int, nargs='+', default=[])
    parser.add_argument('--workers', type=int, default=4)  # The number of workers for each GPU

    # Hyperparameters
    parser.add_argument('--batch-size', type=int, nargs='+', default=[32, 128])
    parser.add_argument('--dropout', type=float, nargs='+', default=[0, 0.5])
    parser.add_argument('--layers', type=int, nargs='+', default=[5])
    parser.add_argument('--units', type=int, nargs='+', default=[64])
    return parser.parse_known_args()


def run_command(args):
    command, gpu_list = args
    if gpu_list:
        gpu_idx = int(multiprocessing.current_process().name.split('-')[-1]) - 1
        gpu = gpu_list[gpu_idx % len(gpu_list)]
        command += ['--gpu', str(gpu)]
    return subprocess.check_output(command)


def main():
    args, unknown = parse_args()
    if 'all' in [e.lower() for e in args.data]:
        args.data = DATASETS
    if torch.cuda.is_available() and not args.gpu:
        args.gpu = list(range(torch.cuda.device_count()))

    args.batch_size = sorted(args.batch_size)
    args.units = sorted(args.units, reverse=True)
    args.layers = sorted(args.layers, reverse=True)

    args_list = []
    for data, fold in itertools.product(args.data, args.folds):
        for u, l, b, d in itertools.product(args.units, args.layers, args.batch_size, args.dropout):
            command = ['python', 'main.py', '--data', data, '--print-all',
                       '--fold', str(fold),
                       '--units', str(u),
                       '--layers', str(l),
                       '--batch-size', str(b),
                       '--dropout', str(d)]
            args_list.append((command + unknown, args.gpu))

    out_list = []
    num_gpu = max(len(args.gpu), 1)
    with Pool(num_gpu * args.workers) as pool:
        for out in tqdm.tqdm(pool.imap_unordered(run_command, args_list), total=len(args_list)):
            out_list.append(out)

    out_dict = defaultdict(lambda: [])
    for line in out_list:
        out = json.loads(line)
        arguments = []
        for k in sorted(out.keys()):
            if k not in ['seed', 'fold', 'gpu', 'all']:
                arguments.append((k, out[k]))
        out_dict[tuple(arguments)].append(out['all'])

    values = []
    for k1, v1 in out_dict.items():
        result = defaultdict(lambda: [])
        for fold_dict in v1:
            for k2, v2 in fold_dict.items():
                result[k2].append(v2)
        result = {k: np.array(v).transpose() for k, v in result.items()}
        best_epoch = find_best_epoch(result['test_loss'], result['test_acc'])
        trn_acc = result['trn_acc'][best_epoch]
        test_acc = result['test_acc'][best_epoch]

        key_dict = dict(k1)
        values.append([key_dict['data'],
                       key_dict['batch_size'],
                       key_dict['dropout'],
                       best_epoch + 1,
                       trn_acc.mean(),
                       trn_acc.std(),
                       test_acc.mean(),
                       test_acc.std()])

    df_columns = ['data', 'batch_size', 'dropout', 'best_epoch', 'trn_mean', 'trn_std', 'test_mean', 'test_std']
    df = pd.DataFrame(values, columns=df_columns)
    df = df.sort_values(by='test_mean', ascending=False).drop_duplicates(['data'])
    df = df.sort_values(by='data')

    print(df.to_string(index=False))


if __name__ == '__main__':
    main()
