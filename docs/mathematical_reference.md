# Mathematical Reference for Topological Spatial Filtering

This document details the mathematical framework for optimizing linear spatial filters using UMAP Cross-Entropy loss.

## 1. Log-Power Projection

Given a set of covariance matrices $C_i \in \mathbb{R}^{M \times M}$ (where $M$ is the number of channels) and a linear spatial filter $\mathbf{w} \in \mathbb{R}^{M \times 1}$, the log-power projection $y_i$ for the $i$-th epoch is computed as:

$$y_i = \log(\mathbf{w}^T C_i \mathbf{w})$$

## 2. 1D UMAP Probabilities

Once the data is projected into the 1D space, we compute the squared pairwise Euclidean distances between epochs:

$$d_{ij}^2 = (y_i - y_j)^2$$

Using the UMAP curve parameters $a$ and $b$, these distances are converted into low-dimensional connection probabilities $w_{ij}$:

$$w_{ij} = \frac{1}{1 + a \cdot (d_{ij}^2)^b}$$

*(A small constant $\epsilon$ is typically added to the denominator or the squared distance to ensure numerical stability and prevent division by zero or NaN gradients).*

## 3. UMAP Cross-Entropy Loss

The spatial filter $\mathbf{w}$ is optimized by minimizing the Fuzzy Set Cross-Entropy between the predefined high-dimensional reference probabilities $V_{ij}$ and the 1D probabilities $w_{ij}$:

$$\mathcal{L} = \frac{1}{N(N-1)} \sum_{i \neq j} \left[ V_{ij} \log\left(\frac{V_{ij}}{w_{ij}}\right) + (1 - V_{ij}) \log\left(\frac{1 - V_{ij}}{1 - w_{ij}}\right) \right]$$

This objective forces the spatial filter to arrange the log-powers $y_i$ such that their 1D topology matches the high-dimensional manifold of the original data.

## 4. Deflation (Gram-Schmidt Orthogonalization)

To find $K > 1$ distinct components, we must ensure that each new spatial filter $\mathbf{w}_k$ captures unique information not explained by the previous filters $\{\mathbf{w}_1, \dots, \mathbf{w}_{k-1}\}$. We use an iterative deflation process based on Gram-Schmidt orthogonalization.

Given a newly optimized weight vector $\mathbf{v}$, we project it out of the subspace spanned by the previously found filters. If we arrange the normalized previously found filters in a matrix $B$, the orthogonalized component $\mathbf{w}_k$ is obtained by:

$$\mathbf{w}_{k, \text{ortho}} = \mathbf{v} - B B^T \mathbf{v}$$

Followed by L2-normalization:

$$\mathbf{w}_k = \frac{\mathbf{w}_{k, \text{ortho}}}{\|\mathbf{w}_{k, \text{ortho}}\|_2}$$

*Alternatively, orthogonal penalties can be added directly to the loss function during simultaneous batched optimization.*

## 5. Pre-calculated Initialization ($\mathbf{W}_{init}$)

Gradient descent on the highly non-convex UMAP Cross-Entropy landscape is susceptible to local minima. Providing a pre-calculated matrix $\mathbf{W}_{init}$ (e.g., from SSD or mSPoC) significantly accelerates convergence.

When $\mathbf{W}_{init}$ is provided, the algorithm begins its search in a structurally meaningful region of the parameter space rather than a random state. The gradient descent then acts as a fine-tuning mechanism—adjusting the pre-calculated filter weights to "stretch" and "compress" the spatial projections so they perfectly align with the target graph topology without needing to escape massive local minima plateaus.
