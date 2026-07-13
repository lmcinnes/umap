import torch
import torch.nn as nn
import torch.optim as optim
import umap
from umap.umap_ import nearest_neighbors, fuzzy_simplicial_set, find_ab_params
import numpy as np
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
from typing import Union, Tuple

def get_umap_graph(
    T_features: np.ndarray = None,
    D_matrix: np.ndarray = None,
    n_neighbors: int = 15,
    metric: str = 'euclidean'
) -> tuple[torch.Tensor, float, float]:
    """
    Constructs the high-dimensional UMAP connectivity graph from either features or a precomputed distance matrix.
    """
    if T_features is None and D_matrix is None:
        raise ValueError("Must provide either T_features or D_matrix.")
    if T_features is not None and D_matrix is not None:
        raise ValueError("Cannot provide both T_features and D_matrix.")

    random_state = np.random.RandomState(42)

    if D_matrix is not None:
        # Precomputed distance matrix pathway
        knn_indices, knn_dists, forest = nearest_neighbors(
            D_matrix, n_neighbors=n_neighbors, metric='precomputed',
            metric_kwds={}, angular=False, random_state=random_state,
        )
        v_ij_sparse, sigmas, rhos = fuzzy_simplicial_set(
            X=D_matrix, n_neighbors=n_neighbors, random_state=random_state,
            metric='precomputed', metric_kwds={}, knn_indices=knn_indices,
            knn_dists=knn_dists, angular=False, set_op_mix_ratio=1.0,
            local_connectivity=1.0,
        )
    else:
        # Standard features pathway
        knn_indices, knn_dists, forest = nearest_neighbors(
            T_features, n_neighbors=n_neighbors, metric=metric,
            metric_kwds={}, angular=False, random_state=random_state,
        )
        v_ij_sparse, sigmas, rhos = fuzzy_simplicial_set(
            X=T_features, n_neighbors=n_neighbors, random_state=random_state,
            metric=metric, metric_kwds={}, knn_indices=knn_indices,
            knn_dists=knn_dists, angular=False, set_op_mix_ratio=1.0,
            local_connectivity=1.0,
        )

    v_ij = torch.tensor(v_ij_sparse.toarray(), dtype=torch.float32)
    a, b = find_ab_params(spread=1.0, min_dist=0.1)
    return v_ij, a, b

class TopologicalFilterBatch(nn.Module):
    """
    Batched multi-dimensional topological spatial filter.
    Evaluates K independent restarts, each finding an N_dim-dimensional embedding space.
    W shape: (K_restarts, N_dim, M_channels)
    """
    def __init__(self, M: int, N_dim: int, K_restarts: int, w_init: torch.Tensor = None):
        super().__init__()
        self.M = M
        self.N_dim = N_dim
        self.K = K_restarts

        # Initialize filters (K, N_dim, M)
        w_tensor = torch.randn(K_restarts, N_dim, M)
        if w_init is not None:
            # w_init shape should be (K_init, N_dim, M)
            K_init = w_init.shape[0]
            if K_init > K_restarts:
                raise ValueError("w_init has more restarts than K_restarts.")
            w_tensor[:K_init] = w_init.clone()

        self.w = nn.Parameter(w_tensor)

    def forward(self, C: torch.Tensor) -> torch.Tensor:
        """
        Calculates batched log-power projections.
        C shape: (N_epochs, M, M)
        Returns y shape: (K_restarts, N_epochs, N_dim)
        """
        # p_{k, i, d} = w_{k,d}^T C_i w_{k,d}
        # w shape: (K, N_dim, M)
        # C shape: (N_epochs, M, M)

        # 'kdm' -> k=K, d=N_dim, m=M
        # 'nml' -> n=N_epochs, m=M, l=M
        # 'kdl' -> k=K, d=N_dim, l=M
        p = torch.einsum('kdm, nml, kdl -> knd', self.w, C, self.w)

        p_clamped = torch.clamp(p, min=1e-8)
        y = torch.log(p_clamped) # shape: (K, N_epochs, N_dim)
        return y


class NormalizedPatternFilterBatch(nn.Module):
    """
    Batched multi-dimensional normalized pattern spatial filter.
    Evaluates K independent restarts, each finding an N_dim-dimensional embedding space.
    a shape: (K_restarts, N_dim, M_channels)
    """
    def __init__(self, M: int, N_dim: int, K_restarts: int, a_init: torch.Tensor = None):
        super().__init__()
        self.M = M
        self.N_dim = N_dim
        self.K = K_restarts

        # Initialize patterns (K, N_dim, M)
        a_tensor = torch.randn(K_restarts, N_dim, M) * 0.1
        if a_init is not None:
            # a_init shape should be (K_init, N_dim, M)
            K_init = a_init.shape[0]
            if K_init > K_restarts:
                raise ValueError("a_init has more restarts than K_restarts.")
            a_tensor[:K_init] = a_init.clone()

        self.a = nn.Parameter(a_tensor)

    def forward(self, C_num: torch.Tensor, C_den: torch.Tensor) -> torch.Tensor:
        """
        Calculates batched normalized log-power projections.
        C_num: (N_epochs, M, M) (Numerator pseudo-covariance)
        C_den: (N_epochs, M, M) (Denominator inverse local background)
        Returns y shape: (K_restarts, N_epochs, N_dim)
        """
        # num = a^T C_num a -> shape (K, N_epochs, N_dim)
        # den = a^T C_den a -> shape (K, N_epochs, N_dim)

        # 'kdm' -> k=K, d=N_dim, m=M
        # 'nml' -> n=N_epochs, m=M, l=M
        # 'kdl' -> k=K, d=N_dim, l=M
        num = torch.einsum('kdm, nml, kdl -> knd', self.a, C_num, self.a)
        den = torch.einsum('kdm, nml, kdl -> knd', self.a, C_den, self.a) ** 2

        power = num / (den + 1e-8)

        y = torch.log(power + 1e-8)
        return y

def umap_cross_entropy_loss(y: torch.Tensor, v_ij: torch.Tensor, a: float, b: float) -> torch.Tensor:
    """
    Computes UMAP loss for multi-dimensional embeddings across K restarts.
    y shape: (K, N_epochs, N_dim)
    Returns shape: (K,)
    """
    K, N_epochs, N_dim = y.shape

    # Euclidean distance squared: d_{ij}^2 = sum_d (y_{i,d} - y_{j,d})^2
    # y unsqueeze(2) -> (K, N_epochs, 1, N_dim)
    # y unsqueeze(1) -> (K, 1, N_epochs, N_dim)
    diff = y.unsqueeze(2) - y.unsqueeze(1) # (K, N, N, N_dim)
    dist_sq = torch.sum(diff ** 2, dim=-1) # (K, N, N)

    w_kij = 1.0 / (1.0 + a * torch.pow(dist_sq + 1e-12, b))

    w_kij_clamped = torch.clamp(w_kij, min=1e-7, max=1.0 - 1e-7)
    v_ij_clamped = torch.clamp(v_ij, min=1e-7, max=1.0 - 1e-7)

    v_ij_bc = v_ij_clamped.unsqueeze(0) # (1, N, N)

    term1 = v_ij_bc * torch.log(v_ij_bc / w_kij_clamped)
    term2 = (1.0 - v_ij_bc) * torch.log((1.0 - v_ij_bc) / (1.0 - w_kij_clamped))

    loss_matrix = term1 + term2

    mask = ~torch.eye(N_epochs, dtype=torch.bool, device=y.device)
    masked_loss = loss_matrix * mask.unsqueeze(0)

    valid_pairs = N_epochs * (N_epochs - 1)
    losses = masked_loss.sum(dim=(1, 2)) / valid_pairs # (K,)

    return losses

def fit_filters(
    C: Union[np.ndarray, torch.Tensor],
    N_dim: int,
    K_restarts: int,
    T_features: np.ndarray = None,
    D_matrix: np.ndarray = None,
    w_init: Union[np.ndarray, torch.Tensor] = None,
    n_neighbors: int = 15,
    metric: str = 'euclidean',
    epochs: int = 500,
    lr: float = 0.01,
    device: str = None,
    verbose: bool = True
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Trains K restarts of an N_dim-dimensional topological spatial filter.

    Returns:
        w_final: Numpy array of shape (K_restarts, N_dim, M)
        final_losses: Numpy array of shape (K_restarts,)
        loss_history: Numpy array of shape (epochs, K_restarts)
    """
    if T_features is None and D_matrix is None:
        raise ValueError("Must provide either T_features or D_matrix to fit_filters.")

    if isinstance(C, np.ndarray):
        C = torch.tensor(C, dtype=torch.float32)
    if w_init is not None and isinstance(w_init, np.ndarray):
        w_init = torch.tensor(w_init, dtype=torch.float32)

    N_epochs, M_channels, _ = C.shape

    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    C = C.to(device)

    if verbose:
        print(f"Building UMAP graph (moving to {device})...")

    v_ij, a, b = get_umap_graph(T_features=T_features, D_matrix=D_matrix, n_neighbors=n_neighbors, metric=metric)
    v_ij = v_ij.to(device)

    model = TopologicalFilterBatch(M=M_channels, N_dim=N_dim, K_restarts=K_restarts, w_init=w_init)
    model.to(device)

    optimizer = optim.AdamW(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)

    loss_history = torch.zeros(epochs, K_restarts)

    pbar = tqdm(range(epochs), disable=not verbose, desc="Optimizing Filters")
    for epoch in pbar:
        optimizer.zero_grad()

        y = model(C) # (K, N_epochs, N_dim)
        losses = umap_cross_entropy_loss(y, v_ij, a, b)
        total_loss = losses.mean()

        total_loss.backward()
        optimizer.step()

        # Independent L2 Normalization across the M_channels dimension for each filter
        # model.w shape: (K, N_dim, M)
        with torch.no_grad():
            w_norms = torch.norm(model.w, p=2, dim=2, keepdim=True) # (K, N_dim, 1)
            model.w.div_(w_norms + 1e-8)

        loss_history[epoch] = losses.detach().cpu()
        current_loss_val = total_loss.item()
        scheduler.step(current_loss_val)
        pbar.set_postfix(mean_loss=f"{current_loss_val:.4f}")

    with torch.no_grad():
        final_y = model(C)
        final_losses = umap_cross_entropy_loss(final_y, v_ij, a, b).detach().cpu()
        w_final = model.w.detach().cpu().clone()

        sorted_indices = torch.argsort(final_losses)
        w_opt = w_final[sorted_indices].numpy()
        final_losses_sorted = final_losses[sorted_indices].numpy()
        loss_history_sorted = loss_history[:, sorted_indices].numpy()

    return w_opt, final_losses_sorted, loss_history_sorted

def fit_patterns(
    C_num: Union[np.ndarray, torch.Tensor],
    C_den: Union[np.ndarray, torch.Tensor],
    N_dim: int,
    K_restarts: int,
    T_features: np.ndarray = None,
    D_matrix: np.ndarray = None,
    a_init: Union[np.ndarray, torch.Tensor] = None,
    n_neighbors: int = 15,
    metric: str = 'euclidean',
    epochs: int = 500,
    lr: float = 0.01,
    device: str = None,
    verbose: bool = True
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Trains K restarts of an N_dim-dimensional normalized pattern spatial filter.

    Returns:
        a_final: Numpy array of shape (K_restarts, N_dim, M)
        final_losses: Numpy array of shape (K_restarts,)
        loss_history: Numpy array of shape (epochs, K_restarts)
    """
    if T_features is None and D_matrix is None:
        raise ValueError("Must provide either T_features or D_matrix to fit_patterns.")

    if isinstance(C_num, np.ndarray):
        C_num = torch.tensor(C_num, dtype=torch.float32)
    if isinstance(C_den, np.ndarray):
        C_den = torch.tensor(C_den, dtype=torch.float32)
    if a_init is not None and isinstance(a_init, np.ndarray):
        a_init = torch.tensor(a_init, dtype=torch.float32)

    N_epochs, M_channels, _ = C_num.shape

    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    C_num = C_num.to(device)
    C_den = C_den.to(device)

    if verbose:
        print(f"Building UMAP graph (moving to {device})...")

    v_ij, a, b = get_umap_graph(T_features=T_features, D_matrix=D_matrix, n_neighbors=n_neighbors, metric=metric)
    v_ij = v_ij.to(device)

    model = NormalizedPatternFilterBatch(M=M_channels, N_dim=N_dim, K_restarts=K_restarts, a_init=a_init)
    model.to(device)

    optimizer = optim.AdamW(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)

    loss_history = torch.zeros(epochs, K_restarts)

    pbar = tqdm(range(epochs), disable=not verbose, desc="Optimizing Patterns")
    for epoch in pbar:
        optimizer.zero_grad()

        y = model(C_num, C_den) # (K, N_epochs, N_dim)
        losses = umap_cross_entropy_loss(y, v_ij, a, b)
        total_loss = losses.mean()

        total_loss.backward()
        optimizer.step()

        # Independent L2 Normalization across the M_channels dimension for each pattern
        # model.a shape: (K, N_dim, M)
        with torch.no_grad():
            a_norms = torch.norm(model.a, p=2, dim=2, keepdim=True) # (K, N_dim, 1)
            model.a.div_(a_norms + 1e-8)

        loss_history[epoch] = losses.detach().cpu()
        current_loss_val = total_loss.item()
        scheduler.step(current_loss_val)
        pbar.set_postfix(mean_loss=f"{current_loss_val:.4f}")

    with torch.no_grad():
        final_y = model(C_num, C_den)
        final_losses = umap_cross_entropy_loss(final_y, v_ij, a, b).detach().cpu()
        a_final = model.a.detach().cpu().clone()

        sorted_indices = torch.argsort(final_losses)
        a_opt = a_final[sorted_indices].numpy()
        final_losses_sorted = final_losses[sorted_indices].numpy()
        loss_history_sorted = loss_history[:, sorted_indices].numpy()

    return a_opt, final_losses_sorted, loss_history_sorted
