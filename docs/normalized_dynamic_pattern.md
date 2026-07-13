# Normalized Dynamic Pattern Filtering

## Conceptual Overview

The original Topological Spatial Filtering algorithm (`TopologicalFilterBatch`) focused on finding a global spatial filter $\mathbf{w}$ to optimize the projected power of the signal across epochs:
$$y_i = \log(\mathbf{w}^T C_i \mathbf{w})$$

However, due to the non-stationarity of background noise in EEG/MEG signals, a single global filter often performs poorly across the entire dataset. To address this, the new algorithm (`NormalizedPatternFilterBatch`) shifts from searching for a global filter to searching for a **global source pattern** $\mathbf{a}$ (a forward model).

Instead of applying the same filter to every epoch, a **dynamic filter** $\mathbf{w}_i$ is constructed for each epoch $i$:
$$\mathbf{w}_i = \frac{B_i^{-1} \mathbf{a}}{\mathbf{a}^T B_i^{-1} \mathbf{a}}$$
where $B_i$ is the averaged covariance matrix of the local neighbors of epoch $i$ (determined via the UMAP graph).

## Mathematical Formulation

To optimize the pattern $\mathbf{a}$ via gradient descent in PyTorch, the unit-gain constraint is embedded directly into the power formula. This constraint normalizes the power into a consistent coordinate system across all epochs.

The normalized power $y_i$ for epoch $i$ is calculated as:
$$y_i = \frac{\mathbf{a}^T (B_i^{-1} C_i B_i^{-1}) \mathbf{a}}{(\mathbf{a}^T B_i^{-1} \mathbf{a})^2}$$

In the implementation, we expect two matrices for each epoch $i$:
*   **Numerator** ($C_{num,i}$): $B_i^{-1} C_i B_i^{-1}$ (the pseudo-covariance)
*   **Denominator** ($C_{den,i}$): $B_i^{-1}$ (the inverse local background)

The computation is thus:
$$y_i = \frac{\mathbf{a}^T C_{num,i} \mathbf{a}}{(\mathbf{a}^T C_{den,i} \mathbf{a})^2}$$

The UMAP loss is then calculated on the logarithms of these normalized powers $\log(y_i)$, utilizing both the attractive forces between neighbors (where the geodesic distance indicates similarity) and the repulsive forces for non-neighbors, fully matching the original UMAP cross-entropy logic.

## L2 Normalization (Computational Stability)

Although the power formula $y_i$ is mathematically invariant to the scaling of the pattern vector $\mathbf{a}$ (if $\mathbf{a}$ is multiplied by a scalar $\lambda$, the scalar cancels out in the UMAP loss because it simply shifts all $\log(y_i)$ by a constant), gradient descent can lead to unstable parameter magnitudes. To prevent the values in $\mathbf{a}$ from exploding or vanishing (which could cause `float32` precision issues like `NaN` or `Inf`), an explicit L2 normalization is applied to $\mathbf{a}$ after every optimization step. This keeps the norm of the pattern vector equal to 1, maintaining numerical stability without altering the theoretical outcome.
