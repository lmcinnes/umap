"""Benchmark: Delaunay adjacency matrix construction in inverse_transform.

Compares old (nested Python loops + lil_matrix) vs new (direct-fill, no intermediates).
"""

import time
import tracemalloc

import numpy as np
import scipy.spatial
import scipy.sparse


def build_adjmat_old(simplices, n_embed):
    """Original: nested Python loops + lil_matrix."""
    adjmat = scipy.sparse.lil_matrix((n_embed, n_embed), dtype=int)
    for i in np.arange(0, simplices.shape[0]):
        for j in simplices[i]:
            if j < n_embed:
                idx = simplices[i][simplices[i] < n_embed]
                adjmat[j, idx] = 1
                adjmat[idx, j] = 1
    return scipy.sparse.csr_matrix(adjmat)


def build_adjmat_new(simplices, n_embed):
    """Direct-fill: pre-allocated int32 arrays, no intermediates, no filter."""
    k = simplices.shape[1]
    n_s = simplices.shape[0]
    rows = np.empty(n_s * k * k, dtype=np.int32)
    cols = np.empty(n_s * k * k, dtype=np.int32)
    for a in range(k):
        for b in range(k):
            off = (a * k + b) * n_s
            rows[off : off + n_s] = simplices[:, a]
            cols[off : off + n_s] = simplices[:, b]
    return scipy.sparse.csr_matrix(
        (np.ones(n_s * k * k, dtype=np.int8), (rows, cols)),
        shape=(n_embed, n_embed),
    )


def verify_equivalence(old, new):
    old_bin = old.copy().astype(bool).astype(int)
    new_bin = new.copy().astype(bool).astype(int)
    diff = old_bin - new_bin
    diff.eliminate_zeros()
    assert diff.nnz == 0, f"Matrices differ at {diff.nnz} entries"


def bench_fn(fn, simplices, n_embed, n_runs):
    times = []
    peaks = []
    for _ in range(n_runs):
        tracemalloc.start()
        t0 = time.perf_counter()
        fn(simplices, n_embed)
        times.append(time.perf_counter() - t0)
        _, peak = tracemalloc.get_traced_memory()
        peaks.append(peak)
        tracemalloc.stop()
    return np.median(times) * 1000, np.median(peaks) / 1024 / 1024


def run_benchmark():
    sizes = [500, 2_000, 5_000, 10_000]

    header = f"{'N':>8} {'Simplices':>10} {'Old ms':>8} {'New ms':>8} {'Speedup':>8} {'Old MB':>8} {'New MB':>8} {'RAM':>8}"
    print(header)
    print("-" * len(header))

    for n in sizes:
        rng = np.random.default_rng(42)
        points = rng.standard_normal((n, 2))
        deltri = scipy.spatial.Delaunay(points, qhull_options="QJ")
        simplices = deltri.simplices
        n_embed = n
        n_runs = 3 if n >= 5_000 else 5

        old_result = build_adjmat_old(simplices, n_embed)
        new_result = build_adjmat_new(simplices, n_embed)
        verify_equivalence(old_result, new_result)

        old_ms, old_mb = bench_fn(build_adjmat_old, simplices, n_embed, n_runs)
        new_ms, new_mb = bench_fn(build_adjmat_new, simplices, n_embed, n_runs)

        speedup = old_ms / new_ms
        ram = new_mb / old_mb if old_mb > 0 else float("inf")

        print(
            f"{n:>8} {simplices.shape[0]:>10} "
            f"{old_ms:>8.1f} {new_ms:>8.1f} {speedup:>7.1f}x "
            f"{old_mb:>7.1f} {new_mb:>7.1f} {ram:>7.2f}x"
        )


if __name__ == "__main__":
    run_benchmark()
