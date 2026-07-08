# -*- coding: utf-8 -*-
"""
Example usage of Topological Spatial Filter (TriCo)
"""

import numpy as np
import matplotlib.pyplot as plt

# Using mock data instead of local paths to make the example runnable out-of-the-box
# In real scenarios, you would use mne to load and preprocess the data:
# import mne
# from pyriemann.estimation import Covariances
# from pyriemann.tangentspace import TangentSpace

from topological_spatial_filter import fit_filters

def run_example():
    print("=== Generating mock EEG data ===")
    N_epochs = 200
    M_channels = 38
    K_filters = 5  # We want to find 5 independent components
    D_tangent = M_channels * (M_channels + 1) // 2

    # 1. Mock Tangent Space features (would normally be output of pyriemann TangentSpace)
    ts_current = np.random.randn(N_epochs, D_tangent)

    # 2. Mock Covariance matrices (would normally be output of pyriemann Covariances)
    A = np.random.randn(N_epochs, M_channels, M_channels)
    covmats_current = A @ A.transpose(0, 2, 1)

    print(f"\n=== Fitting {K_filters} Parallel Topological Filters ===")

    # We fit K_filters in parallel using the batched implementation.
    # No deflation loop needed; we just pick the best one (or top N) at the end.
    w_opt, final_losses, loss_history = fit_filters(
        C=covmats_current,
        T_features=ts_current,
        K=K_filters,
        n_neighbors=15,
        epochs=100,
        lr=0.05,
        verbose=True
    )

    print("\n=== Training Results ===")
    for k in range(K_filters):
        print(f"Filter {k+1} Final Loss: {final_losses[k]:.4f}")

    # Select the best filter (index 0, since w_opt is sorted by loss)
    best_w_np = w_opt[0, :, 0]
    print(f"\nНаилучший фильтр найден с loss: {final_losses[0]:.4f}")

    # === ВИЗУАЛИЗАЦИЯ ===

    # 1. График истории сходимости Loss (Loss History)
    plt.figure(figsize=(10, 6))
    for k in range(K_filters):
        plt.plot(loss_history[:, k], label=f'Filter {k+1} (Loss: {final_losses[k]:.2f})')

    plt.title('Convergence of Topological Spatial Filters')
    plt.xlabel('Epochs')
    plt.ylabel('UMAP Cross-Entropy Loss')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('example_loss_history.png')
    print("Сохранен график сходимости: example_loss_history.png")

    # 2. График лог-мощностей лучшего фильтра
    p_source = np.zeros(covmats_current.shape[0])
    for i in range(covmats_current.shape[0]):
        p_source[i] = np.log(best_w_np.T @ covmats_current[i] @ best_w_np)

    plt.figure(figsize=(10, 4))
    plt.plot(p_source)
    plt.title('Log-Power Source over Epochs (Best Filter)')
    plt.xlabel('Epoch')
    plt.ylabel('Log-Power')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('example_log_power.png')
    print("Сохранен график мощности источника: example_log_power.png")

if __name__ == '__main__':
    run_example()
