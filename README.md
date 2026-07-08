# Topological Spatial Filtering (TriCo)

A PyTorch-based method for discovering and optimizing linear spatial filters for multi-channel data (e.g., EEG/MEG) by matching 1D projected topologies with high-dimensional reference graphs using UMAP fuzzy cross-entropy.

## Overview

The core objective of this algorithm is to find spatial filters $\mathbf{W}$ that project Covariance Matrices ($C_i$) into a 1D log-power space where the resulting pairwise proximities accurately reflect a predefined high-dimensional graph (Fuzzy Simplicial Set).

It iteratively discovers components, using either random initializations or pre-calculated starting states (`w_init`), optimizing the weights natively via PyTorch using the robust AdamW framework.

For the detailed mathematical reference, see [Mathematical Reference](docs/mathematical_reference.md).

## Installation

Ensure you have PyTorch, NumPy, and umap-learn installed.

```bash
pip install torch numpy umap-learn scipy scikit-learn
```

## Usage

```python
import torch
import numpy as np
from topological_spatial_filter import fit_filters

N_epochs, M_channels, K_filters = 200, 16, 5

# Tangent Space vectors (N_epochs, D)
T_features = np.random.randn(N_epochs, 136)

# Symmetric Positive Definite Covariances (N_epochs, M_channels, M_channels)
C_matrices = torch.randn(N_epochs, M_channels, M_channels)
C_matrices = torch.matmul(C_matrices, C_matrices.transpose(1, 2))

# Optional pre-calculated initializations (e.g. from SSD)
w_init = torch.randn(K_filters, M_channels, 1)

# Fit the K parallel topological filters
w_opt, final_losses, loss_history = fit_filters(
    C=C_matrices,
    T_features=T_features,
    K=K_filters,
    w_init=w_init,
    n_neighbors=15,
    epochs=100,
    lr=0.05
)
```
