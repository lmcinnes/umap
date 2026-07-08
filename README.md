# Topological Spatial Filtering (TriCo)

A PyTorch-based method for discovering and optimizing linear spatial filters for multi-channel data (e.g., EEG/MEG) by matching 1D projected topologies with high-dimensional reference graphs using UMAP fuzzy cross-entropy.

## Algorithm Concept

Standard dimensionality reduction techniques (like PCA, ICA, or even standard UMAP) operate either by maximizing variance, ensuring independence, or mapping unconstrained embeddings directly.

**Topological Spatial Filtering (TriCo)** takes a different approach for covariance-based data (like EEG/MEG):
1.  **Reference Graph:** It builds a high-dimensional neighborhood graph (Fuzzy Simplicial Set) representing the true manifold of your data. This is typically done on Tangent Space features of covariance matrices using standard UMAP logic.
2.  **Constrained 1D Projection:** Instead of finding an unconstrained set of points (like t-SNE or UMAP), it forces the 1D projection to be the result of a physical, linear spatial filter $\mathbf{w}$ applied to the raw Covariance Matrices ($y_i = \log(\mathbf{w}^T C_i \mathbf{w})$).
3.  **Topological Alignment:** It optimizes the weights $\mathbf{w}$ directly using gradient descent (AdamW in PyTorch) so that the pairwise proximities of the resulting 1D log-powers match the connections of the high-dimensional reference graph.

The result is a set of physical spatial filters that extract brain sources whose power fluctuations perfectly mirror the underlying manifold of the complex dataset.

## How it differs from Standard UMAP

| Feature | Standard UMAP | Topological Spatial Filter (TriCo) |
| :--- | :--- | :--- |
| **Output** | Unconstrained free-floating points in low-dimensional space. | Linear spatial filters ($\mathbf{w} \in \mathbb{R}^{M \times 1}$) applicable to new covariance data. |
| **Mapping Function** | Non-parametric embedding. | Parametric bilinear form: $y_i = \log(\mathbf{w}^T C_i \mathbf{w})$. |
| **Engine** | Numba / CPU / Stochastic Gradient Descent. | **PyTorch** / GPU acceleration / Batched `AdamW`. |
| **Use Case** | Visualization, clustering. | Source separation, feature extraction in BCI/EEG. |

## Files Added / Changed

Unlike standard UMAP which relies on heavily optimized Numba code across multiple files, this algorithm acts as an extension and does **not** modify the core `umap` library internals.

The following files represent the entirely new PyTorch-based functionality added to this repository:

1.  **`topological_spatial_filter.py`**: The core engine. Contains the PyTorch `TopologicalFilterBatch` module, the vectorized `umap_cross_entropy_loss` calculation without loops, and the main `fit_filters` training loop with GPU support, normalization, and LR scheduling.
2.  **`example.py`**: A standalone testing and demonstration script showing how to map NumPy arrays through the pipeline, pick the best independent components, and visualize convergence loss and log-power sources.
3.  **`docs/mathematical_reference.md`**: A detailed mathematical breakdown (using LaTeX) explaining the log-power projections, 1D UMAP distance conversions, Cross-Entropy formulations, and initialization strategies.

*(The original `umap-learn` files located in the `umap/` folder remain completely unchanged).*

## Installation

Ensure you have PyTorch, NumPy, and umap-learn installed.

```bash
pip install torch numpy umap-learn scipy scikit-learn matplotlib tqdm
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
# C_matrices can be numpy array or torch Tensor
A = np.random.randn(N_epochs, M_channels, M_channels)
C_matrices = A @ A.transpose(0, 2, 1)

# Optional pre-calculated initializations (e.g. from SSD)
w_init = np.random.randn(K_filters, 2, M_channels)

# Fit the K parallel topological filters finding a 2D embedding space
w_opt, final_losses, loss_history = fit_filters(
    C=C_matrices,
    T_features=T_features,
    N_dim=2,
    K_restarts=K_filters,
    w_init=w_init,
    n_neighbors=15,
    epochs=100,
    lr=0.05,
    verbose=True
)
```

For the detailed mathematical reference, see [Mathematical Reference](docs/mathematical_reference.md).
