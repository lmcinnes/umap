"""
Benchmark: fastmath=True removal from layouts.py optimization functions.

Compares performance of UMAP fit/transform with and without fastmath=True
on the numba-compiled layout optimization functions.

Usage:
    uv run python benchmarks/bench_fastmath_removal.py
"""

import time
import warnings

import numba
import numpy as np
from sklearn.datasets import make_blobs, fetch_openml

import umap
from umap.layouts import (
    _optimize_layout_euclidean_single_epoch,
    _optimize_layout_generic_single_epoch,
    _optimize_layout_inverse_single_epoch,
)

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _compile_variants(func):
    """Return (no_fastmath, with_fastmath) compiled versions of func."""
    no_fm = numba.njit(func, fastmath=False, parallel=False)
    with_fm = numba.njit(func, fastmath=True, parallel=False)
    return no_fm, with_fm


def time_umap(X, n_runs=3, **kwargs):
    """Time UMAP fit over n_runs, return list of elapsed seconds."""
    times = []
    for _ in range(n_runs):
        reducer = umap.UMAP(**kwargs)
        t0 = time.perf_counter()
        reducer.fit(X)
        times.append(time.perf_counter() - t0)
    return times


def time_transform(X_train, X_test, n_runs=3, **kwargs):
    """Time UMAP transform over n_runs."""
    reducer = umap.UMAP(**kwargs)
    reducer.fit(X_train)
    times = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        reducer.transform(X_test)
        times.append(time.perf_counter() - t0)
    return times


def report(label, times):
    arr = np.array(times)
    print(
        f"  {label:40s}  mean={arr.mean():.4f}s  std={arr.std():.4f}s  min={arr.min():.4f}s  (n={len(arr)})"
    )


# ---------------------------------------------------------------------------
# Datasets
# ---------------------------------------------------------------------------


def get_datasets():
    datasets = {}

    # Small synthetic
    X_small, _ = make_blobs(n_samples=1_000, n_features=50, centers=10, random_state=42)
    datasets["blobs_1k_d50"] = X_small

    # Medium synthetic
    X_med, _ = make_blobs(n_samples=10_000, n_features=100, centers=20, random_state=42)
    datasets["blobs_10k_d100"] = X_med

    # Large synthetic
    X_large, _ = make_blobs(
        n_samples=50_000, n_features=50, centers=30, random_state=42
    )
    datasets["blobs_50k_d50"] = X_large

    # MNIST (real-world)
    try:
        mnist = fetch_openml("mnist_784", version=1, as_frame=False, parser="auto")
        X_mnist = mnist.data[:20_000].astype(np.float32)
        datasets["mnist_20k"] = X_mnist
    except Exception:
        print("  (skipping MNIST – fetch failed)")

    return datasets


# ---------------------------------------------------------------------------
# Benchmark 1: End-to-end UMAP fit
# ---------------------------------------------------------------------------


def bench_fit(datasets, n_runs=3):
    print("\n" + "=" * 72)
    print("BENCHMARK 1: UMAP.fit() — end-to-end (current code, fastmath=False)")
    print("=" * 72)
    print("(fastmath is already removed in this branch; this measures current perf)\n")

    configs = [
        {"n_neighbors": 15, "n_epochs": 200, "metric": "euclidean"},
        {"n_neighbors": 30, "n_epochs": 500, "metric": "euclidean"},
        {"n_neighbors": 15, "n_epochs": 200, "metric": "cosine"},
    ]

    for name, X in datasets.items():
        print(f"\nDataset: {name} (shape={X.shape})")
        for cfg in configs:
            label = f"nn={cfg['n_neighbors']} ep={cfg['n_epochs']} m={cfg['metric']}"
            times = time_umap(X, n_runs=n_runs, random_state=42, **cfg)
            report(label, times)


# ---------------------------------------------------------------------------
# Benchmark 2: UMAP transform
# ---------------------------------------------------------------------------


def bench_transform(datasets, n_runs=3):
    print("\n" + "=" * 72)
    print("BENCHMARK 2: UMAP.transform()")
    print("=" * 72)

    for name, X in datasets.items():
        if X.shape[0] < 2000:
            continue
        X_train, X_test = X[: int(0.8 * len(X))], X[int(0.8 * len(X)) :]
        print(f"\nDataset: {name} (train={X_train.shape}, test={X_test.shape})")
        times = time_transform(
            X_train,
            X_test,
            n_runs=n_runs,
            n_neighbors=15,
            n_epochs=200,
            random_state=42,
        )
        report("transform", times)


# ---------------------------------------------------------------------------
# Benchmark 3: Micro-benchmark — isolated epoch functions
# ---------------------------------------------------------------------------


def bench_epoch_micro():
    print("\n" + "=" * 72)
    print(
        "BENCHMARK 3: Micro-benchmark — single epoch function (fastmath=True vs False)"
    )
    print("=" * 72)
    print("Compiles both variants and compares raw epoch time.\n")

    no_fm, with_fm = _compile_variants(_optimize_layout_euclidean_single_epoch)

    # Prepare inputs matching the actual function signature:
    # head_embedding, tail_embedding, head, tail, n_vertices,
    # epochs_per_sample, a, b, rng_state_per_sample, gamma, dim, move_other,
    # alpha, epochs_per_negative_sample, epoch_of_next_negative_sample,
    # epoch_of_next_sample, n, densmap_flag, dens_phi_sum, dens_re_sum,
    # dens_re_cov, dens_re_std, dens_re_mean, dens_lambda, dens_R,
    # dens_mu, dens_mu_tot
    np.random.seed(42)
    n_points = 5_000
    n_edges = 50_000
    d = 2

    head_embedding = np.random.randn(n_points, d).astype(np.float32)
    tail_embedding = head_embedding  # same for standard UMAP
    head = np.random.randint(0, n_points, size=n_edges).astype(np.int32)
    tail = np.random.randint(0, n_points, size=n_edges).astype(np.int32)
    epochs_per_sample = np.ones(n_edges, dtype=np.float32)
    epoch_of_next_sample = np.zeros(n_edges, dtype=np.float32)
    epoch_of_next_negative_sample = np.zeros(n_edges, dtype=np.float32)
    epochs_per_negative_sample = np.full(n_edges, 5.0, dtype=np.float32)
    # Per-vertex RNG state: 3 int64s per vertex (tau_rand_int state)
    rng_state = np.random.randint(1, 2**31, size=(n_points, 3)).astype(np.int64)

    # densmap placeholders (disabled)
    dens_phi_sum = np.zeros(n_points, dtype=np.float32)
    dens_re_sum = np.zeros(n_points, dtype=np.float32)
    dens_re_cov = 0.0
    dens_re_std = 1.0
    dens_re_mean = 0.0
    dens_lambda = 0.0
    dens_R = np.zeros(n_points, dtype=np.float32)
    dens_mu = np.zeros(n_points, dtype=np.float32)
    dens_mu_tot = 0.0

    def make_args(emb):
        return (
            emb,
            emb,  # head_embedding, tail_embedding
            head,
            tail,
            n_points,
            epochs_per_sample,
            1.929,
            0.7915,
            rng_state,
            1.0,  # gamma
            d,
            True,  # dim, move_other
            1.0,  # alpha
            epochs_per_negative_sample,
            epoch_of_next_negative_sample.copy(),
            epoch_of_next_sample.copy(),
            1,  # n (current epoch)
            False,  # densmap_flag
            dens_phi_sum,
            dens_re_sum,
            dens_re_cov,
            dens_re_std,
            dens_re_mean,
            dens_lambda,
            dens_R,
            dens_mu,
            dens_mu_tot,
        )

    # Warmup / compile
    for fn in (no_fm, with_fm):
        try:
            fn(*make_args(head_embedding.copy()))
        except Exception as e:
            print(f"  Compile error: {e}")
            return

    n_iters = 20
    for label, fn in [
        ("fastmath=False (current)", no_fm),
        ("fastmath=True  (old)", with_fm),
    ]:
        times = []
        for _ in range(n_iters):
            args = make_args(head_embedding.copy())
            t0 = time.perf_counter()
            fn(*args)
            times.append(time.perf_counter() - t0)
        report(label, times)


# ---------------------------------------------------------------------------
# Benchmark 4: Numerical correctness
# ---------------------------------------------------------------------------


def bench_correctness():
    print("\n" + "=" * 72)
    print("BENCHMARK 4: Numerical correctness — trustworthiness & continuity")
    print("=" * 72)

    from sklearn.manifold import trustworthiness

    X, y = make_blobs(n_samples=2_000, n_features=50, centers=10, random_state=42)

    scores = []
    for i in range(5):
        reducer = umap.UMAP(n_neighbors=15, n_epochs=200, random_state=i)
        X_emb = reducer.fit_transform(X)
        tw = trustworthiness(X, X_emb, n_neighbors=15)
        scores.append(tw)
        print(f"  Run {i}: trustworthiness={tw:.4f}")

    print(f"\n  Mean trustworthiness: {np.mean(scores):.4f} (std={np.std(scores):.4f})")
    print("  (Expected range: 0.95–0.99 for well-separated blobs)")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("UMAP fastmath=True Removal — Performance Benchmark")
    print(f"NumPy {np.__version__}, Numba {numba.__version__}")
    print(f"UMAP version: {umap.__version__}")

    datasets = get_datasets()

    bench_epoch_micro()
    bench_fit(datasets, n_runs=3)
    bench_transform(datasets, n_runs=3)
    bench_correctness()

    print("\n" + "=" * 72)
    print("DONE")
    print("=" * 72)
