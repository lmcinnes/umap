import torch
import torch.nn as nn
import torch.optim as optim
import umap
from umap.umap_ import nearest_neighbors, fuzzy_simplicial_set, find_ab_params
import numpy as np
import scipy.sparse

def get_umap_graph(X: np.ndarray, n_neighbors: int = 15, metric: str = 'euclidean') -> tuple[scipy.sparse.coo_matrix, float, float]:
    """
    Constructs the high-dimensional UMAP connectivity graph.

    Args:
        X: Feature matrix of shape (N, D).
        n_neighbors: The size of local neighborhood (in terms of number of neighboring sample points)
                     used for manifold approximation.
        metric: The metric to use to compute distances in high dimensional space.

    Returns:
        v_ij: A sparse matrix representing the fuzzy simplicial set (connection probabilities).
        a: The a parameter of the UMAP layout curve.
        b: The b parameter of the UMAP layout curve.
    """
    random_state = np.random.RandomState(42)

    # 1. Find nearest neighbors
    knn_indices, knn_dists, forest = nearest_neighbors(
        X,
        n_neighbors=n_neighbors,
        metric=metric,
        metric_kwds={},
        angular=False,
        random_state=random_state,
    )

    # 2. Compute fuzzy simplicial set
    v_ij, sigmas, rhos = fuzzy_simplicial_set(
        X=X,
        n_neighbors=n_neighbors,
        random_state=random_state,
        metric=metric,
        metric_kwds={},
        knn_indices=knn_indices,
        knn_dists=knn_dists,
        angular=False,
        set_op_mix_ratio=1.0,
        local_connectivity=1.0,
    )

    # 3. Get layout curve parameters for optimization
    # Note: UMAP typically uses spread=1.0, min_dist=0.1
    a, b = find_ab_params(spread=1.0, min_dist=0.1)

    return v_ij, a, b


class TopologicalFilter(nn.Module):
    """
    Topological spatial filter for EEG/MEG data.
    Finds an optimal linear spatial filter w in R^(Mx1) such that the extracted log-powers
    preserve a given graph topology (UMAP connectivity matrix).
    """
    def __init__(self, num_channels: int, w_init: torch.Tensor = None):
        super().__init__()
        if w_init is not None:
            self.w = nn.Parameter(w_init.clone())
        else:
            self.w = nn.Parameter(torch.randn(num_channels, 1))

    def forward(self, C: torch.Tensor) -> torch.Tensor:
        """
        Calculates log-power projections for given covariance matrices.

        Args:
            C: Tensor of shape (N, M, M), SPD covariance matrices.

        Returns:
            y: Tensor of shape (N, 1), the 1D log-power coordinate for each epoch.
        """
        # Calculate C @ w -> (N, M, 1)
        Cw = torch.matmul(C, self.w)

        # Calculate w^T @ (C @ w) -> (N, 1, 1)
        # Using batched dot product for efficiency
        w_t = self.w.t().unsqueeze(0)  # (1, 1, M)
        p = torch.matmul(w_t, Cw).squeeze(-1)  # (N, 1) keeping y as (N, 1) output vector as requested

        # Protect against negative values or exact zeros from numerical instability
        p_clamped = torch.clamp(p, min=1e-8)

        # Calculate coordinate y_k = log(p_k)
        y = torch.log(p_clamped)
        return y


def umap_cross_entropy_loss(y: torch.Tensor, v_ij: torch.Tensor, a: float, b: float) -> torch.Tensor:
    """
    Computes the UMAP Fuzzy Set Cross Entropy Loss based on 1D coordinates.

    Args:
        y: Tensor of shape (N, 1), 1D coordinates.
        v_ij: Tensor of shape (N, N), high-dimensional connections (probabilities, dense).
        a: UMAP constant a.
        b: UMAP constant b.

    Returns:
        Scalar tensor representing the total loss.
    """
    N = v_ij.shape[0]

    # Distance calculation: d_ij^2 = (y_i - y_j)^2
    # y is (N, 1), y.t() is (1, N)
    dist_sq = torch.pow(y - y.t(), 2)  # Broadcasting -> (N, N)

    # Low-dimensional probabilities w_ij = 1 / (1 + a * (d_ij^2)^b)
    # Add a tiny epsilon before power to avoid nan gradients when distance is exactly 0
    w_ij = 1.0 / (1.0 + a * torch.pow(dist_sq + 1e-12, b))

    # We do not use w_ij.fill_diagonal_(0.0) here as it modifies the tensor in-place,
    # crashing PyTorch's backward pass. The diagonal is naturally ignored later by the mask anyway.

    # Clamp w_ij and v_ij to prevent log(0) resulting in NaN
    w_ij_clamped = torch.clamp(w_ij, min=1e-7, max=1.0 - 1e-7)
    v_ij_clamped = torch.clamp(v_ij, min=1e-7, max=1.0 - 1e-7)

    # Fuzzy Set Cross Entropy calculation
    term1 = v_ij_clamped * torch.log(v_ij_clamped / w_ij_clamped)
    term2 = (1.0 - v_ij_clamped) * torch.log((1.0 - v_ij_clamped) / (1.0 - w_ij_clamped))

    loss_matrix = term1 + term2

    # Mask out the diagonal elements completely to ignore them from sum
    mask = ~torch.eye(N, dtype=torch.bool, device=v_ij.device)

    # Sum over all i != j
    total_loss = torch.sum(loss_matrix[mask])
    return total_loss


def fit_topological_filter(
    C: torch.Tensor,
    v_ij: torch.Tensor,
    a: float,
    b: float,
    w_init: torch.Tensor = None,
    n_restarts: int = 5,
    epochs: int = 500,
    lr: float = 0.01
) -> tuple[torch.Tensor, float]:
    """
    Trains the topological spatial filter using multistart optimization.

    Args:
        C: Tensor of shape (N, M, M), SPD covariance matrices.
        v_ij: Tensor of shape (N, N), fuzzy simplicial set from UMAP (dense).
        a: UMAP curve parameter.
        b: UMAP curve parameter.
        w_init: Optional initial weights tensor of shape (M, 1).
        n_restarts: Number of independent training restarts to avoid local minima.
        epochs: Number of training epochs per restart.
        lr: Learning rate for Adam optimizer.

    Returns:
        w_best: Tensor of shape (M, 1), the best trained spatial filter weights.
        best_loss: The final loss achieved by w_best.
    """
    N, M, _ = C.shape
    device = C.device
    v_ij = v_ij.to(device)

    best_loss = float('inf')
    w_best = None

    for restart in range(n_restarts):
        # Determine initialization
        if restart == 0 and w_init is not None:
            init_tensor = w_init.clone().to(device)
        else:
            init_tensor = torch.randn(M, 1, device=device)

        model = TopologicalFilter(num_channels=M, w_init=init_tensor)
        model.to(device)

        optimizer = optim.Adam(model.parameters(), lr=lr)

        final_loss = None

        for epoch in range(epochs):
            optimizer.zero_grad()

            # Forward pass
            y = model(C)

            # Loss calculation
            loss = umap_cross_entropy_loss(y, v_ij, a, b)

            # Backward pass
            loss.backward()
            optimizer.step()

            # Normalize weights to prevent them from blowing up: w = w / ||w||_2
            with torch.no_grad():
                w_norm = torch.norm(model.w, p=2)
                model.w.div_(w_norm + 1e-8)

            final_loss = loss.item()

        # Record best
        if final_loss is not None and final_loss < best_loss:
            best_loss = final_loss
            w_best = model.w.detach().cpu().clone()

        print(f"Restart {restart+1}/{n_restarts} completed with loss: {final_loss:.4f}")

    return w_best, best_loss


if __name__ == "__main__":
    # Integration test
    torch.manual_seed(42)
    np.random.seed(42)

    N, M = 150, 16

    # 1. Create Mock data for UMAP Graph (flat features, e.g. tangent space features)
    D_flat = M * (M + 1) // 2
    X_mock = np.random.randn(N, D_flat)

    print("Computing UMAP Graph...")
    v_sparse, param_a, param_b = get_umap_graph(X_mock, n_neighbors=15)

    # Convert sparse scipy matrix to dense PyTorch tensor
    v_dense = torch.tensor(v_sparse.toarray(), dtype=torch.float32)

    # 2. Create Mock Covariance matrices (N, M, M) - must be SPD
    A = torch.randn(N, M, M)
    C_mock = torch.matmul(A, A.transpose(1, 2))

    # 3. Fit filter
    print("Fitting Topological Filter...")
    w_opt, min_loss = fit_topological_filter(
        C_mock, v_dense, param_a, param_b,
        n_restarts=3, epochs=100, lr=0.05
    )

    print(f"Best weight vector shape: {w_opt.shape}")
    print(f"Minimum loss achieved: {min_loss:.4f}")
    print("Pipeline finished successfully.")
