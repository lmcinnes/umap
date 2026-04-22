"""Benchmark compute_membership_strengths: old (sequential) vs new (parallel split)."""

import time
import numpy as np
import numba


# Old implementation (from HEAD) - sequential despite parallel=True decorator
@numba.njit(
    locals={
        "knn_dists": numba.types.float32[:, ::1],
        "sigmas": numba.types.float32[::1],
        "rhos": numba.types.float32[::1],
        "val": numba.types.float32,
    },
    parallel=True,
)
def compute_membership_strengths_old(
    knn_indices,
    knn_dists,
    sigmas,
    rhos,
    return_dists=False,
    bipartite=False,
):
    n_samples = knn_indices.shape[0]
    n_neighbors = knn_indices.shape[1]

    rows = np.zeros(knn_indices.size, dtype=np.int32)
    cols = np.zeros(knn_indices.size, dtype=np.int32)
    vals = np.zeros(knn_indices.size, dtype=np.float32)
    if return_dists:
        dists = np.zeros(knn_indices.size, dtype=np.float32)
    else:
        dists = None

    for i in range(n_samples):  # <-- sequential!
        for j in range(n_neighbors):
            if knn_indices[i, j] == -1:
                continue
            if (bipartite == False) & (knn_indices[i, j] == i):
                val = 0.0
            elif knn_dists[i, j] - rhos[i] <= 0.0 or sigmas[i] == 0.0:
                val = 1.0
            else:
                val = np.exp(-((knn_dists[i, j] - rhos[i]) / (sigmas[i])))

            rows[i * n_neighbors + j] = i
            cols[i * n_neighbors + j] = knn_indices[i, j]
            vals[i * n_neighbors + j] = val
            if return_dists:
                dists[i * n_neighbors + j] = knn_dists[i, j]

    return rows, cols, vals, dists


# New implementation - parallel with split functions
@numba.njit(
    locals={
        "knn_dists": numba.types.float32[:, ::1],
        "sigmas": numba.types.float32[::1],
        "rhos": numba.types.float32[::1],
        "val": numba.types.float32,
    },
    parallel=True,
)
def _compute_membership_strengths_new(
    knn_indices,
    knn_dists,
    sigmas,
    rhos,
    bipartite=False,
):
    n_samples = knn_indices.shape[0]
    n_neighbors = knn_indices.shape[1]

    rows = np.zeros(knn_indices.size, dtype=np.int32)
    cols = np.zeros(knn_indices.size, dtype=np.int32)
    vals = np.zeros(knn_indices.size, dtype=np.float32)

    for i in numba.prange(n_samples):  # <-- parallel!
        row_offset = i * n_neighbors
        rho = rhos[i]
        sigma = sigmas[i]
        for j in range(n_neighbors):
            neighbor = knn_indices[i, j]
            if neighbor == -1:
                continue
            dist = knn_dists[i, j]
            if (not bipartite) and (neighbor == i):
                val = 0.0
            elif dist - rho <= 0.0 or sigma == 0.0:
                val = 1.0
            else:
                val = np.exp(-((dist - rho) / sigma))

            rows[row_offset + j] = i
            cols[row_offset + j] = neighbor
            vals[row_offset + j] = val

    return rows, cols, vals


@numba.njit(
    locals={
        "knn_dists": numba.types.float32[:, ::1],
        "sigmas": numba.types.float32[::1],
        "rhos": numba.types.float32[::1],
        "val": numba.types.float32,
    },
    parallel=True,
)
def _compute_membership_strengths_with_dists_new(
    knn_indices,
    knn_dists,
    sigmas,
    rhos,
    bipartite=False,
):
    n_samples = knn_indices.shape[0]
    n_neighbors = knn_indices.shape[1]

    rows = np.zeros(knn_indices.size, dtype=np.int32)
    cols = np.zeros(knn_indices.size, dtype=np.int32)
    vals = np.zeros(knn_indices.size, dtype=np.float32)
    dists = np.zeros(knn_indices.size, dtype=np.float32)

    for i in numba.prange(n_samples):
        row_offset = i * n_neighbors
        rho = rhos[i]
        sigma = sigmas[i]
        for j in range(n_neighbors):
            neighbor = knn_indices[i, j]
            if neighbor == -1:
                continue
            dist = knn_dists[i, j]
            if (not bipartite) and (neighbor == i):
                val = 0.0
            elif dist - rho <= 0.0 or sigma == 0.0:
                val = 1.0
            else:
                val = np.exp(-((dist - rho) / sigma))

            rows[row_offset + j] = i
            cols[row_offset + j] = neighbor
            vals[row_offset + j] = val
            dists[row_offset + j] = dist

    return rows, cols, vals, dists


def compute_membership_strengths_new(
    knn_indices, knn_dists, sigmas, rhos, return_dists=False, bipartite=False
):
    if return_dists:
        return _compute_membership_strengths_with_dists_new(
            knn_indices, knn_dists, sigmas, rhos, bipartite
        )
    rows, cols, vals = _compute_membership_strengths_new(
        knn_indices, knn_dists, sigmas, rhos, bipartite
    )
    return rows, cols, vals, None


def generate_test_data(n_samples, n_neighbors):
    rng = np.random.default_rng(42)
    knn_indices = np.zeros((n_samples, n_neighbors), dtype=np.int32)
    for i in range(n_samples):
        candidates = np.arange(n_samples)
        candidates = candidates[candidates != i]
        knn_indices[i] = rng.choice(candidates, n_neighbors, replace=False)
    knn_dists = rng.random((n_samples, n_neighbors)).astype(np.float32)
    sigmas = rng.random(n_samples).astype(np.float32) + 0.1
    rhos = rng.random(n_samples).astype(np.float32) * 0.5
    return knn_indices, knn_dists, sigmas, rhos


def benchmark(fn, knn_indices, knn_dists, sigmas, rhos, return_dists, n_runs=5):
    # Warmup (JIT compile)
    fn(knn_indices, knn_dists, sigmas, rhos, return_dists=return_dists)

    times = []
    for _ in range(n_runs):
        start = time.perf_counter()
        fn(knn_indices, knn_dists, sigmas, rhos, return_dists=return_dists)
        times.append(time.perf_counter() - start)
    return np.median(times), np.std(times)


def main():
    print("=" * 60)
    print("Benchmark: compute_membership_strengths")
    print("Old (sequential range) vs New (parallel prange + split)")
    print("=" * 60)

    configs = [
        (10_000, 15),
        (50_000, 15),
        (100_000, 15),
        (100_000, 30),
    ]

    for n_samples, n_neighbors in configs:
        print(f"\nn_samples={n_samples:,}, n_neighbors={n_neighbors}")
        print("-" * 50)

        knn_indices, knn_dists, sigmas, rhos = generate_test_data(
            n_samples, n_neighbors
        )

        for return_dists in [False, True]:
            label = "with dists" if return_dists else "no dists  "

            old_time, old_std = benchmark(
                compute_membership_strengths_old,
                knn_indices,
                knn_dists,
                sigmas,
                rhos,
                return_dists,
            )
            new_time, new_std = benchmark(
                compute_membership_strengths_new,
                knn_indices,
                knn_dists,
                sigmas,
                rhos,
                return_dists,
            )

            speedup = old_time / new_time
            print(
                f"  {label}: old={old_time * 1000:6.2f}ms, new={new_time * 1000:6.2f}ms, speedup={speedup:.2f}x"
            )


if __name__ == "__main__":
    main()
