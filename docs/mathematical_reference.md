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

## 4. Independent Multi-Start Optimization

The UMAP Cross-Entropy loss landscape is highly non-convex and features many local minima. To combat this, the algorithm trains $K$ completely independent spatial filters in parallel.

At the end of the optimization process, the algorithm evaluates all $K$ independent filters, ranks them by their final Cross-Entropy loss, and selects the one that achieved the absolute minimum loss as the optimal spatial filter $\mathbf{w}_{best}$.

## 5. Pre-calculated Initialization ($\mathbf{W}_{init}$)

Providing a pre-calculated matrix $\mathbf{W}_{init}$ (e.g., from SSD or mSPoC) for some of the $K$ initializations significantly accelerates convergence.

When $\mathbf{W}_{init}$ is provided, the algorithm begins its search in a structurally meaningful region of the parameter space rather than a random state. The gradient descent then acts as a fine-tuning mechanism—adjusting the pre-calculated filter weights to "stretch" and "compress" the spatial projections so they perfectly align with the target graph topology without needing to escape massive local minima plateaus.
