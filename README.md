# PyTorch Implementation of Correlated Graph Neural Networks

[Outcome Correlation in Graph Neural Network Regression](https://arxiv.org/abs/2002.08274)</br>

Junteng Jia and Austin Benson</br>

arXiv:2002.08274, 2020.<br/>

## Overview
Correlated Graph Neural Networks model the correlations among labels as well as features of each node:
- C-GNN models the correlation as a multivariate Gaussian and learns the correlation structure in O(m) per optimization step, where m is the number of edges.
- LP-GNN assumes positive correlation among neighboring vertices, and runs label propagation to interpolate GNN residuals on the testing vertices.

## Requirements
- Python 3.7+
- PyTorch 1.2.0+
- [DGL](https://github.com/dmlc/dgl)
- [GPyTorch](https://github.com/cornellius-gp/gpytorch)

## Usage

### Download this Repository
```git clone``` this repo to your local machine.

### Dataset
[US Election](https://projects.fivethirtyeight.com/2016-election-forecast/) dataset is used as a running example. The dataset is included in this repo.

### Train and Test

We so far implemented 2 graph neural network structures: GCN and GraphSAGE, for LP-GNN as well as C-GNN.

To train and test GCN-based LP-GNN, use the following script:
```bash
scripts/run_lp_gcn.sh
```
To train and test GraphSAGE-based LP-GNN, use the following script:
```bash
scripts/run_lp_graphsage.sh
```
To train and test GCN-based C-GNN, use the following script:

```bash
scripts/run_cgnn_gcn.sh
```

To train and test GraphSAGE-based C-GNN, use the following script:

```bash
scripts/run_cgnn_graphsage.sh
```

The default hyper-parameters should give reasonably good results.

If you have any questions, feel free to open an issue.

## References
[C-GNN](https://github.com/000Justin000/gnn-residual-correlation) (original implementation in Julia)</br>
[GCN](https://github.com/tkipf/pygcn)</br>
[GraphSAGE](https://github.com/dmlc/dgl/tree/master/examples/pytorch/graphsage)</br>
