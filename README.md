# STA663 Final Project: Optimization and Demonstration of Biclustering using Sparse Singular Value Decomposition

author: Chengxin Yang, Guanqi Zeng

The project disects the paper Biclustering via Sparse Singular Value Decomposition and implement the SSVD algorithm in the original paper with optimized functions.

## Functions 

We created three main functions: the `SSVD`, `SSVD_numba`, and `ClusterPlot`. All three functions can be imported by the following commands:

`from SSVD.SSVD import SSVD`

`from SSVD.SSVD_numba import SSVD_numba`

`from SSVD.ClusterPlot import ClusterPlot`

There are some dependency modules. Please make sure you have the scipy, numpy, sparsesvd and seaborn modules.

## Package Installation
The package can be installed via the following command:

`pip install git+https://github.com/GuanqizEng/STA663-final-biclustering.git@main`

or  `pip install -i https://test.pypi.org/simple/ SSVD-cxgq2==2.0` (recommended)