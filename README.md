# Integrating Social and Knowledge Graphs in GNN-Based Recommender Systems

This is our PYG implementation for the paper:

> 2024. IJCNN

Author: Anonymous Authors

## Introduction

We introduce a novel recommendation framework,named Integrating Social and Knowledge Graphs (ISKG), de-signed for use within Graph Neural Network (GNN)-based Recommender Systems.

## Citation

If you want to use our codes and datasets in your research, please cite:

```
@inproceedings{IJCNN,
  author    = {},
  title     = {{ISKG:} Integrating Social and Knowledge Graphs in GNN-Based Recommender Systems},
  booktitle = {{IJCNN}},
  pages     = {},
  year      = {2024}
}
```

## Environment Requirement

The code has been tested running under Python 3.6.5. The required packages are as follows:

use conda to install.
```conda
conda create -n pyG python=3.6
conda activate pyG
```

```pip
* tensorflow == 1.12.0
* numpy == 1.15.4
* scipy == 1.1.0
* sklearn == 0.20.0
$ python -c "import torch; print(torch.__version__)"
$ python -c "import torch; print(torch.version.cuda)"
pip install torch-scatter -f https://pytorch-geometric.com/whl/torch-${TORCH}+${CUDA}.html
pip install torch-sparse -f https://pytorch-geometric.com/whl/torch-${TORCH}+${CUDA}.html
pip install torch-cluster -f https://pytorch-geometric.com/whl/torch-${TORCH}+${CUDA}.html
pip install torch-spline-conv -f https://pytorch-geometric.com/whl/torch-${TORCH}+${CUDA}.html
pip install torch-geometric
```

