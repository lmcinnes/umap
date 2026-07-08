import numpy as np
import matplotlib.pyplot as plt
from topological_spatial_filter import fit_filters

def run_example():
    print("=== Generating mock EEG data ===")
    N_epochs = 200
    M_channels = 38
    N_dim = 2 # 2D Embedding space
    K_restarts = 5 # 5 parallel independent multi-starts
    D_tangent = M_channels * (M_channels + 1) // 2

    # 1. Mock Tangent Space features
    ts_current = np.random.randn(N_epochs, D_tangent)

    # 2. Mock Covariance matrices
    A = np.random.randn(N_epochs, M_channels, M_channels)
    covmats_current = A @ A.transpose(0, 2, 1)

    print(f"\n=== Fitting {K_restarts} Parallel Topological {N_dim}D Embeddings ===")

    w_opt, final_losses, loss_history = fit_filters(
        C=covmats_current,
        T_features=ts_current,
        N_dim=N_dim,
        K_restarts=K_restarts,
        n_neighbors=15,
        epochs=100,
        lr=0.05,
        verbose=True
    )

    print("\n=== Training Results ===")
    for k in range(K_restarts):
        print(f"Restart {k+1} Final Loss: {final_losses[k]:.4f}")

    # Best filter matrix (N_dim x M_channels)
    best_W = w_opt[0]
    print(f"\nНаилучшее {N_dim}D вложение найдено с loss: {final_losses[0]:.4f}")

    # === ВИЗУАЛИЗАЦИЯ ===

    # 1. График истории сходимости
    plt.figure(figsize=(10, 6))
    for k in range(K_restarts):
        plt.plot(loss_history[:, k], label=f'Restart {k+1} (Loss: {final_losses[k]:.2f})')

    plt.title('Convergence of Topological Spatial Filters')
    plt.xlabel('Epochs')
    plt.ylabel('UMAP Cross-Entropy Loss')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('example_loss_history.png')
    print("Сохранен график сходимости: example_loss_history.png")

    # 2. Scatter plot of the 2D projection
    y_points = np.zeros((N_epochs, N_dim))
    for i in range(N_epochs):
        for d in range(N_dim):
            y_points[i, d] = np.log(best_W[d].T @ covmats_current[i] @ best_W[d])

    plt.figure(figsize=(8, 6))
    plt.scatter(y_points[:, 0], y_points[:, 1], c=np.arange(N_epochs), cmap='viridis', s=20)
    plt.title(f'Optimized {N_dim}D Log-Power Projection (Colored by Epoch)')
    plt.xlabel('Dimension 1 (Log-Power)')
    plt.ylabel('Dimension 2 (Log-Power)')
    plt.colorbar(label='Epoch Index')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('example_2d_projection.png')
    print("Сохранен график 2D вложения: example_2d_projection.png")

if __name__ == '__main__':
    run_example()
