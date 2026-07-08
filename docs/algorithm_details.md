# Detailed Algorithm Implementation Guide: Topological Spatial Filtering (TriCo)

This document provides a comprehensive, step-by-step breakdown of the exact processes, mathematical formulas, and computational mechanics currently implemented in the `topological_spatial_filter.py` PyTorch codebase.

---

## 1. Input Data & Graph Construction

The algorithm requires two sets of inputs derived from raw multidimensional data (e.g., EEG/MEG epochs):
1. **$C_i$**: Epoch-wise Covariance Matrices of shape $(N_{epochs}, M_{channels}, M_{channels})$.
2. **$T\_features$**: Tangent Space features of shape $(N_{epochs}, D_{tangent})$, representing the covariance matrices mapped onto a flat Euclidean manifold.

### The High-Dimensional Reference Graph
Before optimization begins, we build a reference topology representing the "true" relationships between the data epochs. We use the standard UMAP algorithm on the $T\_features$:
1. Find nearest neighbors in the high-dimensional Tangent Space.
2. Compute the **Fuzzy Simplicial Set**: an $N \times N$ dense adjacency matrix of connection probabilities, denoted as $V_{ij}$.
3. Extract the UMAP curve parameters $a$ and $b$, which control the mapping of distances to probabilities.

---

## 2. Batched Spatial Filtering

The core of the algorithm is a custom PyTorch module (`TopologicalFilterBatch`). Instead of looking for a single spatial filter, we simultaneously optimize a batch of $K$ independent linear spatial filters $\mathbf{W} \in \mathbb{R}^{K \times M \times 1}$.

For every restart $k$ and every epoch $i$, the algorithm projects the $M \times M$ covariance matrix $C_i$ using $N_{dim}$ parallel spatial filters to form an **$N_{dim}$-dimensional Log-Power coordinate** vector $\mathbf{y}_{k,i}$:

$$y_{k,i,d} = \log(\mathbf{w}_{k,d}^T C_i \mathbf{w}_{k,d})$$

**Implementation Details:**
- **Vectorization:** Instead of slow loops, this multi-dimensional projection is executed entirely in parallel using `torch.einsum('kdm, nml, kdl -> knd')`.
- **Clamping:** To prevent numerical instabilities (like $\log(0)$ or negative values causing `NaN`s), the raw power is clamped to a minimum value of `1e-8` before the logarithm is applied.

---

## 3. Distance & Low-Dimensional Graph

Once projected, we measure the topology of the newly formed $N_{dim}$-dimensional spaces. For each restart $k$, we compute the squared Euclidean distance between every pair of epochs $(i, j)$ by summing across all embedded dimensions $d$:

$$d_{k,ij}^2 = \|\mathbf{y}_{k,i} - \mathbf{y}_{k,j}\|^2 = \sum_d (y_{k,i,d} - y_{k,j,d})^2$$

These multi-dimensional distances are then translated into low-dimensional connection probabilities ($w_{k,ij}$) using the UMAP family of curves (with parameters $a$ and $b$):

$$w_{k,ij} = \frac{1}{1 + a \cdot (d_{k,ij}^2 + \epsilon)^b}$$

**Implementation Details:**
- **Epsilon ($\epsilon$):** A tiny constant (`1e-8`) is added to the squared distance to prevent division by zero or NaN gradients during the exponentiation step (if $d^2$ is exactly $0$).
- **Diagonal Masking:** Self-connections (where $i = j$) are masked out of the final loss calculation using a boolean identity matrix `~torch.eye`.

---

## 4. The Error Function (UMAP Cross-Entropy Loss)

The optimization objective is to adjust the filter weights $\mathbf{w}_k$ so that the 1D probabilities ($w_{k,ij}$) match the pre-calculated high-dimensional reference probabilities ($V_{ij}$). This is achieved by minimizing the **Fuzzy Set Cross-Entropy Loss**:

$$\mathcal{L}_k = \frac{1}{N(N-1)} \sum_{i \neq j} \left[ V_{ij} \log\left(\frac{V_{ij}}{w_{k,ij}}\right) + (1 - V_{ij}) \log\left(\frac{1 - V_{ij}}{1 - w_{k,ij}}\right) \right]$$

**Implementation Details:**
- **Clamping:** Both $V_{ij}$ and $w_{k,ij}$ are tightly clamped between `[1e-7, 1.0 - 1e-7]` before calculating the logarithms to guarantee the gradient descent never encounters a `NaN` crash.
- **Normalization (Valid Pairs):** The summation of the loss matrix is explicitly divided by $N(N-1)$ (the number of valid non-diagonal pairs). This makes the magnitude of the loss scale-invariant. Without this division, datasets with thousands of epochs would produce massive loss values (in the millions), causing the optimizer to explode or require erratic, dataset-specific learning rates.

---

## 5. Gradient Descent & Optimization Mechanics

The algorithm uses PyTorch's Autograd engine to calculate the gradients of the Cross-Entropy loss with respect to the filter weights $\mathbf{w}_k$.

The optimization loop incorporates several modern machine learning practices:
1. **AdamW Optimizer:** We use the `AdamW` optimizer, which handles weight decay perfectly and handles the highly non-convex topography of the UMAP cross-entropy landscape better than standard `Adam` or `SGD`.
2. **Learning Rate Scheduler:** We wrap the optimizer in a `ReduceLROnPlateau` scheduler. If the loss plateaus for 10 epochs, the learning rate is automatically halved, allowing for fine-grained convergence tuning.
3. **L2 Normalization (Crucial Step):** UMAP distances are inherently scale-invariant. Because we take the logarithm of the power, scaling $\mathbf{w}_k$ shifts the coordinates $y_i$ uniformly, leaving pairwise distances exactly the same. Without constraints, gradient descent would cause the weights to grow to infinity. To prevent this, **after every optimizer step**, the weights are explicitly L2-normalized:
   $$\mathbf{w}_k = \frac{\mathbf{w}_k}{\|\mathbf{w}_k\|_2}$$

---

## 6. Independent Multi-Start Strategy

The UMAP error landscape is riddled with local minima. Instead of using complex deflation methods (like Gram-Schmidt orthogonalization) to find subsequent components, the current implementation uses a **brute-force parallelized multi-start search**:

1. We initialize $K$ completely independent filters.
2. The user has the option to pass `w_init`—a predefined matrix of "smart" initial guesses (for example, derived from an earlier linear method).
3. The remaining filters in the batch are initialized with pure random noise ($\mathcal{N}(0, 1)$).
4. All $K$ filters traverse the loss landscape independently through the epochs.
5. At the very end of training, the algorithm evaluates the final Cross-Entropy loss of each filter, **sorts the results in ascending order**, and outputs them.

The filter sitting at index `0` is mathematically guaranteed to be the spatial filter that best preserves the high-dimensional graph structure.
