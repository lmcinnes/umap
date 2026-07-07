import torch
import torch.nn as nn
import torch.optim as optim

class TopologicalFilter(nn.Module):
    """
    Topological spatial filter for EEG/MEG data.
    Finds an optimal linear spatial filter w in R^(Mx1) such that the extracted log-powers
    preserve a given graph topology (UMAP connectivity matrix).
    """
    def __init__(self, num_channels: int):
        super().__init__()
        # Initialize trainable weight vector w (M x 1) with normal distribution
        self.w = nn.Parameter(torch.randn(num_channels, 1))

    def forward(self, C: torch.Tensor) -> torch.Tensor:
        """
        Calculates log-power projections for given covariance matrices.

        Args:
            C: Tensor of shape (N, M, M), SPD covariance matrices.
               N is the number of epochs (windows), M is the number of channels.

        Returns:
            y: Tensor of shape (N,), the 1D log-power coordinate for each epoch.
        """
        # Calculate C @ w -> (N, M, 1)
        Cw = torch.matmul(C, self.w)

        # Calculate w^T @ (C @ w) -> (N, 1, 1)
        # Using batched dot product for efficiency
        w_t = self.w.t().unsqueeze(0)  # (1, 1, M)
        p = torch.matmul(w_t, Cw).squeeze(-1).squeeze(-1)  # (N,) safely squeezing specific dimensions

        # Protect against negative values or exact zeros from numerical instability
        p_clamped = torch.clamp(p, min=1e-8)

        # Calculate coordinate y_k = log(p_k)
        y = torch.log(p_clamped)
        return y


def compute_w_ij(y: torch.Tensor, a: float = 1.5769, b: float = 0.8950) -> torch.Tensor:
    """
    Computes low-dimensional connections probabilities based on UMAP formula.

    Args:
        y: Tensor of shape (N,), 1D coordinates.
        a: UMAP constant a.
        b: UMAP constant b.

    Returns:
        w_ij: Tensor of shape (N, N), low-dimensional connection probabilities.
    """
    # Vectorized computation of distance matrix |y_i - y_j|
    # Broadcasting: (N, 1) - (1, N) -> (N, N)
    dist = torch.abs(y.unsqueeze(1) - y.unsqueeze(0))

    # Calculate w_ij = 1 / (1 + a * dist^(2b))
    # Added a small epsilon in the denominator (though mathematically it's 1 + >=0, so always >=1)
    w_ij = 1.0 / (1.0 + a * torch.pow(dist, 2 * b))

    # Mask diagonal (self-connections probability should be 0)
    w_ij.fill_diagonal_(0.0)

    return w_ij


def umap_cross_entropy_loss(v_ij: torch.Tensor, w_ij: torch.Tensor) -> torch.Tensor:
    """
    Computes the UMAP Fuzzy Set Cross Entropy Loss.

    Args:
        v_ij: Tensor of shape (N, N), high-dimensional connections (probabilities).
        w_ij: Tensor of shape (N, N), low-dimensional connections (probabilities).

    Returns:
        Scalar tensor representing the total loss.
    """
    N = v_ij.shape[0]

    # Clamp w_ij and v_ij to prevent log(0) resulting in NaN
    w_ij_clamped = torch.clamp(w_ij, min=1e-7, max=1.0 - 1e-7)
    v_ij_clamped = torch.clamp(v_ij, min=1e-7, max=1.0 - 1e-7)

    # Fuzzy Set Cross Entropy calculation
    term1 = v_ij_clamped * torch.log(v_ij_clamped / w_ij_clamped)
    term2 = (1.0 - v_ij_clamped) * torch.log((1.0 - v_ij_clamped) / (1.0 - w_ij_clamped))

    loss_matrix = term1 + term2

    # Mask out the diagonal elements
    mask = ~torch.eye(N, dtype=torch.bool, device=v_ij.device)

    # Sum over all i != j
    total_loss = torch.sum(loss_matrix[mask])
    return total_loss


def train_topological_filter(
    C: torch.Tensor,
    v_ij: torch.Tensor,
    num_epochs: int = 500,
    lr: float = 0.01,
    tol: float = 1e-5
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Trains the topological spatial filter.

    Args:
        C: Tensor of shape (N, M, M), SPD covariance matrices.
        v_ij: Tensor of shape (N, N), fuzzy simplicial set from UMAP.
        num_epochs: Maximum number of training epochs.
        lr: Learning rate for AdamW optimizer.
        tol: Tolerance for convergence checking.

    Returns:
        w: Tensor of shape (M, 1), the trained spatial filter weights.
        y: Tensor of shape (N,), the resulting log-power coordinates.
    """
    N, M, _ = C.shape

    model = TopologicalFilter(num_channels=M)
    model.to(C.device)

    optimizer = optim.AdamW(model.parameters(), lr=lr)

    prev_loss = float('inf')

    for epoch in range(num_epochs):
        optimizer.zero_grad()

        # Forward pass
        y = model(C)
        w_ij = compute_w_ij(y)
        loss = umap_cross_entropy_loss(v_ij, w_ij)

        # Backward pass
        loss.backward()
        optimizer.step()

        # Normalize weights to prevent them from blowing up
        with torch.no_grad():
            w_norm = torch.norm(model.w)
            model.w.div_(w_norm + 1e-8)

        current_loss = loss.item()

        # Check for convergence
        if abs(prev_loss - current_loss) < tol:
            print(f"Converged at epoch {epoch} with loss: {current_loss:.4f}")
            break

        prev_loss = current_loss

    # Final normalization just to be safe
    with torch.no_grad():
        w_norm = torch.norm(model.w)
        model.w.div_(w_norm + 1e-8)

    # Return detached final w and y
    final_w = model.w.detach().cpu()

    with torch.no_grad():
        final_y = model(C).detach().cpu()

    return final_w, final_y

if __name__ == "__main__":
    # Simple test case to verify everything works end-to-end
    torch.manual_seed(42)

    N, M = 100, 16

    # Generate random positive-definite covariance matrices
    A = torch.randn(N, M, M)
    C_mock = torch.matmul(A, A.transpose(1, 2))

    # Generate mock symmetric UMAP connectivity matrix
    v_mock = torch.rand(N, N)
    v_mock = (v_mock + v_mock.T) / 2
    v_mock.fill_diagonal_(0.0)

    # Train the filter
    w_opt, y_opt = train_topological_filter(C_mock, v_mock, num_epochs=200, lr=0.05)

    print(f"Trained weight vector shape: {w_opt.shape}")
    print(f"Coordinates shape: {y_opt.shape}")
    print("Training finished successfully.")
