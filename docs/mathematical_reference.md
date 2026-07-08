# Mathematical Reference: Topological Spatial Filtering

The proposed methodology aims to discover linear spatial filters that extract source-power envelopes whose topological structure maximally matches the intrinsic Riemannian geometry of the high-dimensional data. This document outlines the process for mapping data into an $N_{dim}$-dimensional embedding space.

## 1. Signal Enhancement via Spatio-Spectral Decomposition (SSD)
To maximize the signal-to-noise ratio (SNR) of the oscillatory activity of interest, we first apply Spatio-Spectral Decomposition (SSD). The raw multi-channel EEG/MEG signals are filtered into a target frequency band to obtain the signal matrix $\mathbf{X}_S$, and into flanking broadband/stopband frequencies to obtain the noise matrix $\mathbf{X}_N$. We compute the corresponding covariance matrices $\mathbf{C}_S$ and $\mathbf{C}_N$ and solve the generalized eigenvalue problem:
$$ \mathbf{C}_S \mathbf{W}_{SSD} = \mathbf{C}_N \mathbf{W}_{SSD} \mathbf{\Lambda} $$
We retain the first $N_{comp}$ components that exhibit the highest SNR. The continuous data is then projected into the reduced SSD space: $\mathbf{X}_{proj} = \mathbf{W}_{SSD}^T \mathbf{X}_{S}$.

## 2. Epoching and Riemannian Tangent Space Mapping
The continuous projected signals are segmented into $N_{epochs}$ overlapping epochs. For each epoch $i \in \{1, \dots, N_{epochs}\}$, we estimate the spatial covariance matrix $\mathbf{C}_i \in \mathbb{R}^{N_{comp} \times N_{comp}}$. Since these matrices lie on the non-Euclidean Symmetric Positive Definite (SPD) manifold, we map them to a Euclidean tangent space at the Riemannian Fréchet mean $\bar{\mathbf{C}}$:
$$ \mathbf{S}_i = \log \left( \bar{\mathbf{C}}^{-1/2} \mathbf{C}_i \bar{\mathbf{C}}^{-1/2} \right) $$
The upper triangular elements of the symmetric matrix $\mathbf{S}_i$ are vectorized to form the feature vectors $\mathbf{t}_i \in \mathbb{R}^{D}$.

## 3. Topological Graph Construction
Using the Euclidean tangent space vectors $\mathbf{t}_i$, we construct a high-dimensional fuzzy simplicial set (a weighted k-nearest neighbors graph) according to the UMAP algorithm. Let $d(\mathbf{t}_i, \mathbf{t}_j)$ be the Euclidean distance. We compute the directed edge weights as:
$$ v_{i|j} = \exp\left(-\frac{\max(0, d(\mathbf{t}_i, \mathbf{t}_j) - \rho_i)}{\sigma_i}\right) $$
where $\rho_i$ is the distance to the nearest neighbor of $\mathbf{t}_i$, and $\sigma_i$ is a scaling parameter. The final symmetric adjacency matrix $V$ representing the topological similarity between epoch $i$ and epoch $j$ is computed as:
$$ v_{ij} = v_{i|j} + v_{j|i} - v_{i|j}v_{j|i} $$

## 4. Multi-Dimensional Topological Spatial Filtering
Instead of relying on an external behavioral variable or searching for unconstrained low-dimensional points, we force the $N_{dim}$-dimensional embedding to be a direct physical projection of the covariance matrices. We seek a set of spatial filters (forming a matrix $\mathbf{W} \in \mathbb{R}^{N_{comp} \times N_{dim}}$) such that the logarithmic power of the extracted components for epoch $i$ forms a vector $\mathbf{y}_i \in \mathbb{R}^{N_{dim}}$:
$$ y_{i,d}(\mathbf{W}) = \log(\mathbf{w}_d^T \mathbf{C}_i \mathbf{w}_d), \quad \text{for } d=1,\dots,N_{dim} $$
This multidimensional projection must preserve the topological structure of $V$. We define the pairwise squared Euclidean distance in this $N_{dim}$-projected space as:
$$ d_{ij}^2(\mathbf{W}) = \|\mathbf{y}_i - \mathbf{y}_j\|^2 = \sum_{d=1}^{N_{dim}} \left(y_{i,d} - y_{j,d}\right)^2 $$

Following UMAP's Student t-distribution approximation, the low-dimensional connection probabilities $q_{ij}(\mathbf{W})$ are modeled as:
$$ q_{ij}(\mathbf{W}) = \frac{1}{1 + a \left( d_{ij}^2(\mathbf{W}) \right)^b} $$
where $a$ and $b$ are hyper-parameters controlling the tightness of the embedding. The optimal filter matrix $\mathbf{W}$ is found by minimizing the fuzzy set cross-entropy loss $\mathcal{L}(\mathbf{W})$:
$$ \mathcal{L}(\mathbf{W}) = \frac{1}{N_{pairs}} \sum_{i \neq j} \left[ v_{ij} \log \left( \frac{v_{ij}}{q_{ij}(\mathbf{W})} \right) + (1 - v_{ij}) \log \left( \frac{1 - v_{ij}}{1 - q_{ij}(\mathbf{W})} \right) \right] $$

## 5. Optimization Strategy and Multi-Start
The objective function $\mathcal{L}(\mathbf{W})$ represents the fuzzy set cross-entropy. Optimizing this loss directly is challenging because the projection function combined with the Student-t distribution yields a highly non-convex loss landscape.

To address this, the optimization is solved using Automatic Differentiation (PyTorch). The training process involves:
1. **Forward Pass:** Batches of covariance matrices $\mathbf{C}_i$ are multiplied by the filters $\mathbf{w}_d$ to yield $N_{dim}$-dimensional log-power coordinates $\mathbf{y}_i$. Pairwise distances and probabilities $q_{ij}(\mathbf{W})$ are computed.
2. **Backpropagation:** The cross-entropy loss $\mathcal{L}(\mathbf{W})$ is evaluated against $v_{ij}$. Gradients are computed.
3. **Weight Update:** The spatial filters are updated using AdamW.
4. **L2 Normalization:** To prevent scale explosions due to the scale-invariance of the log-distances, each filter $\mathbf{w}_d$ is independently L2-normalized after each step.

**Independent Multi-Start Mechanism:**
To ensure we find the global minimum for the $N_{dim}$-dimensional subspace, we evaluate $K_{restarts}$ completely independent filter matrices in parallel. The algorithm returns the matrix $\mathbf{W}$ that produced the absolute lowest final Cross-Entropy loss.

## 6. Self-Explained Features
A fundamental advantage of Topological Spatial Filtering is its ability to directly answer *why* data separates into specific clusters. While standard embeddings are black boxes, our approach guarantees that the resulting axes are driven by the power fluctuations of distinct, linearly mixed neural sources.

By mapping the spatial filters back to the sensor space, we compute the corresponding spatial patterns (forward models) $\mathbf{a}_d$:
$$ \mathbf{a}_d = \frac{\mathbf{C}_x \mathbf{w}_d}{\mathbf{w}_d^T \mathbf{C}_x \mathbf{w}_d} $$
where $\mathbf{C}_x$ is the average sensor-space covariance matrix.
