# Model-Agnostic Graph Augmentation

This project is a PyTorch implementation of [Model-Agnostic Augmentation for
Accurate Graph Classification](https://arxiv.org/abs/2202.10107)
(WWW 2022). This paper proposes NodeSam and SubMix, two novel algorithms for
model-agnostic graph augmentation.

## Prerequisites

Our implementation is based on Python 3.7 and PyTorch Geometric. Please see the
full list of packages required to run our codes in `requirements.txt`.

- Python 3.7
- PyTorch 1.4.0
- PyTorch Geometric 1.6.3

PyTorch Geometric requires a separate installation process from the other
packages. We included `install.sh` to guide the installation process of PyTorch
Geometric based on the OS and CUDA version. The code includes the cases for
`Linux + CUDA 10.0`, `Linux + CUDA 10.1`, and `MacOS + CPU`.

## Datasets

We use 9 datasets in our work, which are not included in this repository due to
their size but can be downloaded easily by PyTorch Geometric. You can run
`data.py` in the `src` directory to download the datasets in the `data/graphs`
directory. Our split indices in `data/splits` are also based on these datasets.

|Name    | Graphs|  Nodes|     Edges|Features|Labels|
|:-------|------:|------:|---------:|-------:|-----:|
|DD      |  1,178|334,925|   843,046|      89|     2|
|ENZYMES |    600| 19,580|    37,282|       3|     6|
|MUTAG   |    188|  3,371|     3,721|       7|     2|
|NCI1    |  4,110|122,747|   132,753|      37|     2|
|NCI109  |  4,127|122,494|   132,604|      38|     2|
|PROTEINS|  1,113| 43,471|    81,044|       3|     2|
|PTC_MR  |    334|  4,915|     5,054|      18|     2|
|COLLAB  |  5,000|372,474|12,286,079|       3|     2|
|Twitter |144,033|580,768|   717,558|      18|     2|

## Usage

We included `demo.sh`, which reproduces the experimental results of our paper.
The code automatically downloads the datasets and trains a GIN classifier with
all of our proposed approaches for graph augmentation. In other words, you just
have to type the following command.
```
bash demo.sh
```

This demo script uses all of your GPUs by default and runs four workers for each
GPU to reduce the running time. You can change experimental arguments such as
the number of workers in `run.py` and the other hyperparameters such as the
number of epochs, batch size, or the initial learning rate in `main.py`. Since
`run.py` is a wrapper script for the parallel execution of `main.py`, all
optional arguments given to `run.py` are passed also to `main.py`.

## Citation

Please cite the following paper if you use our code:
```
@inproceedings{DBLP:conf/www/YooSK22,
  author    = {Jaemin Yoo and
               Sooyeon Shim and
               U Kang},
  title     = {Model-Agnostic Augmentation for Accurate Graph Classification},
  booktitle = {{WWW} '22: The {ACM} Web Conference 2022, Virtual Event, Lyon, France,
               April 25 - 29, 2022},
  pages     = {1281--1291},
  publisher = {{ACM}},
  year      = {2022},
}
```
