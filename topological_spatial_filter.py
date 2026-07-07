import torch
import torch.nn as nn
import torch.optim as optim
import umap
from umap.umap_ import nearest_neighbors, fuzzy_simplicial_set, find_ab_params
import numpy as np
from tqdm.auto import tqdm
import matplotlib.pyplot as plt

def get_umap_graph(T_features: np.ndarray, n_neighbors: int = 15, metric: str = 'euclidean') -> tuple[torch.Tensor, float, float]:
    """
    Constructs the high-dimensional UMAP connectivity graph.

    Args:
        T_features: Feature matrix of shape (N_epochs, D), representing vectorized
                    covariance features in tangent space.
        n_neighbors: The size of local neighborhood (in terms of number of neighboring sample points)
                     used for manifold approximation.
        metric: The metric to use to compute distances in high dimensional space.

    Returns:
        v_ij: A dense PyTorch tensor representing the fuzzy simplicial set (connection probabilities).
        a: The a parameter of the UMAP layout curve.
        b: The b parameter of the UMAP layout curve.
    """
    random_state = np.random.RandomState(42)

    # 1. Find nearest neighbors
    knn_indices, knn_dists, forest = nearest_neighbors(
        T_features,
        n_neighbors=n_neighbors,
        metric=metric,
        metric_kwds={},
        angular=False,
        random_state=random_state,
    )

    # 2. Compute fuzzy simplicial set
    v_ij_sparse, sigmas, rhos = fuzzy_simplicial_set(
        X=T_features,
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

    # Convert sparse scipy matrix to dense PyTorch tensor
    v_ij = torch.tensor(v_ij_sparse.toarray(), dtype=torch.float32)

    # 3. Get layout curve parameters for optimization
    a, b = find_ab_params(spread=1.0, min_dist=0.1)

    return v_ij, a, b

class TopologicalFilterBatch(nn.Module):
    """
    Batched topological spatial filter for EEG/MEG data.
    Evaluates K independent spatial filters W in R^(K x M x 1) in parallel.
    """
    def __init__(self, M: int, K: int, w_init: torch.Tensor = None):
        super().__init__()
        self.M = M
        self.K = K

        # Initialize filters
        w_tensor = torch.randn(K, M, 1)
        if w_init is not None:
            K_init = w_init.shape[0]
            if K_init > K:
                raise ValueError("w_init has more filters than K.")
            # Overwrite the first K_init filters with the provided ones
            w_tensor[:K_init] = w_init.clone()

        self.w = nn.Parameter(w_tensor)

    def forward(self, C: torch.Tensor) -> torch.Tensor:
        """
        Calculates batched log-power projections for given covariance matrices.

        Args:
            C: Tensor of shape (N_epochs, M, M), SPD covariance matrices.

        Returns:
            y: Tensor of shape (K, N_epochs), 1D log-power coordinates.
        """
        # We need to compute p_{k,i} = w_k^T C_i w_k
        # C shape: (N, M, M)
        # w shape: (K, M, 1)

        # Using einsum for efficient batched matrix multiplication over independent dimensions
        # 'nml' -> n=N, m=M, l=M (C_i matrix)
        # 'kmj' -> k=K, m=M, j=1 (w_k vector)
        # Result of C_i w_k for all k: (K, N, M, 1)
        # However, we can compute the bilinear form directly:
        # p_{k,i} = sum_{m,l} w_{k,m,1} * C_{i,m,l} * w_{k,l,1}

        # In einsum:
        # w_k^T: 'km...' (using just 'km' since the last dim is 1)
        # C_i: 'nml'
        # w_k: 'kl...'

        # Flatten w to (K, M) for easier einsum
        w_flat = self.w.squeeze(-1) # (K, M)

        # Equation: k: filter index, n: epoch index, m: row index, l: col index
        p = torch.einsum('km, nml, kl -> kn', w_flat, C, w_flat)

        # Protect against negative values or exact zeros from numerical instability
        p_clamped = torch.clamp(p, min=1e-8)

        # Calculate coordinate y_{k,i} = log(p_{k,i})
        y = torch.log(p_clamped)
        return y

def umap_cross_entropy_loss(y: torch.Tensor, v_ij: torch.Tensor, a: float, b: float) -> torch.Tensor:
    """
    Computes the UMAP Fuzzy Set Cross Entropy Loss for K batched filters.

    Args:
        y: Tensor of shape (K, N_epochs), 1D coordinates.
        v_ij: Tensor of shape (N_epochs, N_epochs), high-dimensional connection probabilities.
        a: UMAP constant a.
        b: UMAP constant b.

    Returns:
        losses: Tensor of shape (K,), representing the loss for each filter.
    """
    K, N = y.shape

    # Calculate pairwise squared distances d_{k,ij}^2 = (y_{k,i} - y_{k,j})^2
    # y.unsqueeze(2): (K, N, 1)
    # y.unsqueeze(1): (K, 1, N)
    # dist_sq: (K, N, N)
    dist_sq = torch.pow(y.unsqueeze(2) - y.unsqueeze(1), 2)

    # Low-dimensional probabilities w_{k,ij} = 1 / (1 + a * (d_{k,ij}^2)^b)
    # Add a tiny epsilon before power to avoid nan gradients when distance is exactly 0
    w_kij = 1.0 / (1.0 + a * torch.pow(dist_sq + 1e-8, b))

    # Clamp w_kij and v_ij to prevent log(0) resulting in NaN
    w_kij_clamped = torch.clamp(w_kij, min=1e-7, max=1.0 - 1e-7)
    v_ij_clamped = torch.clamp(v_ij, min=1e-7, max=1.0 - 1e-7)

    # Broadcast v_ij (N, N) -> (1, N, N) to match w_kij (K, N, N)
    v_ij_bc = v_ij_clamped.unsqueeze(0)

    # Fuzzy Set Cross Entropy calculation
    term1 = v_ij_bc * torch.log(v_ij_bc / w_kij_clamped)
    term2 = (1.0 - v_ij_bc) * torch.log((1.0 - v_ij_bc) / (1.0 - w_kij_clamped))

    loss_matrix = term1 + term2  # Shape: (K, N, N)

    # Mask out the diagonal elements (where i == j) to ignore them from the sum
    mask = ~torch.eye(N, dtype=torch.bool, device=y.device) # Shape: (N, N)

    # Compute sum over i != j for each k
    # Apply mask over the NxN dimensions and sum them up
    # We can multiply by mask then sum over dim 1 and 2
    masked_loss = loss_matrix * mask.unsqueeze(0)

    # Take the mean over all valid pairs (N * (N - 1)) to keep the loss scale
    # invariant to the number of epochs (N). This prevents massive loss values
    # and stabilizes gradients.
    valid_pairs = N * (N - 1)
    losses = masked_loss.sum(dim=(1, 2)) / valid_pairs # Shape: (K,)

    return losses

def fit_filters(
    C: torch.Tensor,
    T_features: np.ndarray,
    K: int,
    w_init: torch.Tensor = None,
    n_neighbors: int = 15,
    metric: str = 'euclidean',
    epochs: int = 500,
    lr: float = 0.01,
    device: str = None,
    verbose: bool = True
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Trains K batched topological spatial filters.

    Args:
        C: Tensor of shape (N_epochs, M, M), SPD covariance matrices.
        T_features: Numpy array of shape (N_epochs, D), tangent space features for graph building.
        K: Number of independent filters to optimize.
        w_init: Optional initial weights tensor of shape (K_init, M, 1).
        n_neighbors: UMAP graph neighbors parameter.
        epochs: Number of training epochs.
        lr: Learning rate for Adam optimizer.
        device: Hardware device to run optimization ('cuda', 'cpu', etc.). Auto-detects if None.
        verbose: Whether to display a tqdm progress bar with loss logs.

    Returns:
        w_final: Tensor of shape (K, M, 1), the trained spatial filter weights (sorted by loss).
        final_losses: Tensor of shape (K,), the final loss achieved by each filter (sorted).
        loss_history: Tensor of shape (epochs, K), the recorded loss history.
    """
    N, M, _ = C.shape

    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Move C to correct device
    C = C.to(device)

    if verbose:
        print(f"Building UMAP graph from tangent space features (moving to {device})...")

    v_ij, a, b = get_umap_graph(T_features, n_neighbors=n_neighbors, metric=metric)
    v_ij = v_ij.to(device)

    model = TopologicalFilterBatch(M=M, K=K, w_init=w_init)
    model.to(device)

    optimizer = optim.AdamW(model.parameters(), lr=lr)
    # verbose argument was deprecated/removed in newer PyTorch versions for schedulers in favor of logging
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)

    loss_history = torch.zeros(epochs, K)

    pbar = tqdm(range(epochs), disable=not verbose, desc="Optimizing Filters")
    for epoch in pbar:
        optimizer.zero_grad()

        # Forward pass -> y shape: (K, N)
        y = model(C)

        # Loss calculation -> losses shape: (K,)
        losses = umap_cross_entropy_loss(y, v_ij, a, b)

        # We can optimize all filters independently by taking the mean (or sum) of their losses
        total_loss = losses.mean()

        # Backward pass
        total_loss.backward()
        optimizer.step()

        # Normalize weights to prevent them from blowing up: w_k = w_k / ||w_k||_2
        # self.w shape is (K, M, 1)
        with torch.no_grad():
            w_norms = torch.norm(model.w, p=2, dim=1, keepdim=True) # Shape: (K, 1, 1)
            model.w.div_(w_norms + 1e-8)

        # Update loss history
        loss_history[epoch] = losses.detach().cpu()

        # Update scheduler and progress bar
        current_loss_val = total_loss.item()
        scheduler.step(current_loss_val)
        pbar.set_postfix(mean_loss=f"{current_loss_val:.4f}")

    # Process and sort the final outputs
    with torch.no_grad():
        final_y = model(C)
        final_losses = umap_cross_entropy_loss(final_y, v_ij, a, b).detach().cpu()
        w_final = model.w.detach().cpu().clone()

        # Sort filters based on their final loss ascending
        sorted_indices = torch.argsort(final_losses)
        w_opt = w_final[sorted_indices]
        final_losses_sorted = final_losses[sorted_indices]
        loss_history_sorted = loss_history[:, sorted_indices]

    return w_opt, final_losses_sorted, loss_history_sorted

if __name__ == "__main__":
    # Integration test for the entire batched pipeline
    torch.manual_seed(42)
    np.random.seed(42)

    N_epochs = 200
    M_channels = 16
    K_filters = 5
    K_init = 2
    D_tangent = 136 # equivalent to 16 * (16 + 1) // 2

    print("Generating mock data...")
    # 1. Create Mock Tangent Space Features
    T_mock = np.random.randn(N_epochs, D_tangent)

    # 2. Create Mock Covariance matrices (N, M, M) - must be SPD
    A = torch.randn(N_epochs, M_channels, M_channels)
    C_mock = torch.matmul(A, A.transpose(1, 2))

    # 3. Create partial w_init for the first K_init filters
    w_init_mock = torch.randn(K_init, M_channels, 1)

    # 4. Fit all filters concurrently
    print(f"Fitting {K_filters} Topological Filters in parallel...")
    w_opt, final_losses, loss_history = fit_filters(
        C=C_mock,
        T_features=T_mock,
        K=K_filters,
        w_init=w_init_mock,
        n_neighbors=15,
        epochs=100,
        lr=0.05,
        verbose=True
    )

    print("\n--- Training Results ---")
    print(f"Optimized Weight Tensor Shape: {w_opt.shape}")
    print(f"Loss History Tensor Shape: {loss_history.shape}")

    for k in range(K_filters):
        print(f"Filter {k+1} (Best #{k+1}) Final Loss: {final_losses[k].item():.4f}")

    print("Plotting Loss History...")
    plt.figure(figsize=(10, 6))
    for k in range(K_filters):
        plt.plot(loss_history[:, k].numpy(), label=f'Filter {k+1} (Final: {final_losses[k].item():.1f})')

    plt.title('Topological Spatial Filter Training Convergence')
    plt.xlabel('Epochs')
    plt.ylabel('UMAP Fuzzy Cross-Entropy Loss')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('convergence_plot.png')
    print("Plot saved to 'convergence_plot.png'.")

    print("Batched pipeline finished successfully.")
