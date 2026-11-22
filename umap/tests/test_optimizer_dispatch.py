"""Cross-cutting verification for optimizer dispatch boundaries.

These tests fill combinations not exercised by the focused kernel suites. Some
matrix dimensions are intentionally not parametrized: inverse transform has a
dedicated optimizer-independent objective, aligned UMAP has one established
optimizer, and DensMAP is only implemented for Euclidean output embeddings.
"""

import numpy as np
import pytest
import scipy.sparse
from sklearn.datasets import make_blobs

from umap import UMAP
import umap.umap_ as umap_module

MODERN_OPTIMIZERS = ("adam", "momentum")


def sparse_blobs(n_samples):
    data, _ = make_blobs(
        n_samples=n_samples,
        n_features=24,
        centers=6,
        random_state=42,
    )
    data[np.abs(data) < 4.0] = 0.0
    return scipy.sparse.csr_matrix(data.astype(np.float32))


@pytest.mark.parametrize("optimizer", MODERN_OPTIMIZERS)
def test_modern_densmap_memberships_follow_csr_order(monkeypatch, optimizer):
    captured = {}

    def capture_layout(head_embedding, *args, **kwargs):
        captured["memberships"] = kwargs["densmap_kwds"]["mu"].copy()
        captured["csr_data"] = kwargs["csr_data"].copy()
        return head_embedding

    monkeypatch.setattr(umap_module, "optimize_layout_euclidean", capture_layout)
    data, _ = make_blobs(
        n_samples=80,
        n_features=6,
        centers=4,
        random_state=42,
    )

    UMAP(
        n_neighbors=10,
        n_epochs=20,
        init="random",
        optimizer=optimizer,
        densmap=True,
        random_state=42,
    ).fit(data)

    np.testing.assert_array_equal(captured["memberships"], captured["csr_data"])


@pytest.mark.parametrize("optimizer", MODERN_OPTIMIZERS)
def test_large_sparse_hard_negative_fit_and_transform(optimizer):
    data = sparse_blobs(1050)
    model = UMAP(
        n_neighbors=10,
        n_epochs=10,
        init="random",
        optimizer=optimizer,
        negative_selection_range=250,
        negative_sample_scale_adaptation_samples=0,
        exclude_graph_neighbors=True,
        random_state=42,
    ).fit(data)

    transformed = model.transform(data[:25])

    assert model.embedding_.shape == (1050, 2)
    assert transformed.shape == (25, 2)
    assert np.isfinite(model.embedding_).all()
    assert np.isfinite(transformed).all()


@pytest.mark.parametrize("optimizer", MODERN_OPTIMIZERS)
def test_large_generic_fit_and_transform(optimizer):
    data, _ = make_blobs(
        n_samples=1060,
        n_features=8,
        centers=6,
        random_state=42,
    )
    model = UMAP(
        n_neighbors=10,
        n_epochs=10,
        init="random",
        output_metric="haversine",
        optimizer=optimizer,
        negative_selection_range=265,
        random_state=42,
    ).fit(data[:1030])

    transformed = model.transform(data[1030:])

    assert model.embedding_.shape == (1030, 2)
    assert transformed.shape == (30, 2)
    assert np.isfinite(model.embedding_).all()
    assert np.isfinite(transformed).all()


@pytest.mark.parametrize("optimizer", MODERN_OPTIMIZERS)
def test_large_densmap_fit_produces_finite_density_radii(optimizer):
    data, _ = make_blobs(
        n_samples=1030,
        n_features=8,
        centers=6,
        cluster_std=[0.2, 0.4, 0.8, 1.2, 1.8, 2.5],
        random_state=42,
    )
    model = UMAP(
        n_neighbors=10,
        n_epochs=10,
        init="random",
        optimizer=optimizer,
        densmap=True,
        output_dens=True,
        random_state=42,
    ).fit(data)

    assert model.embedding_.shape == (1030, 2)
    assert np.isfinite(model.embedding_).all()
    assert np.isfinite(model.rad_orig_).all()
    assert np.isfinite(model.rad_emb_).all()


@pytest.mark.parametrize("optimizer", MODERN_OPTIMIZERS)
def test_unseeded_parallel_fit_is_finite(optimizer):
    data, _ = make_blobs(
        n_samples=180,
        n_features=6,
        centers=4,
        random_state=42,
    )
    embedding = UMAP(
        n_neighbors=10,
        n_epochs=10,
        init="random",
        optimizer=optimizer,
        random_state=None,
        n_jobs=2,
    ).fit_transform(data)

    assert embedding.shape == (180, 2)
    assert np.isfinite(embedding).all()


def test_sparse_compatibility_fit_and_transform():
    data = sparse_blobs(180)
    model = UMAP(
        n_neighbors=10,
        n_epochs=15,
        init="random",
        optimizer="compatibility",
        random_state=42,
    ).fit(data[:150])

    transformed = model.transform(data[150:])

    assert model.embedding_.shape == (150, 2)
    assert transformed.shape == (30, 2)
    assert np.isfinite(model.embedding_).all()
    assert np.isfinite(transformed).all()


@pytest.mark.parametrize("optimizer", ["adam", "momentum"])
def test_transform_many_points_into_tiny_reference_embedding(optimizer):
    # More than 1024 query points selects the large-input kernels while the
    # reference embedding holds fewer points than the minimum negative window.
    rng = np.random.RandomState(0)
    model = UMAP(
        n_neighbors=5,
        n_epochs=30,
        optimizer=optimizer,
        compatibility_layout=False,
        random_state=0,
    ).fit(rng.normal(size=(60, 8)).astype(np.float32))

    embedding = model.transform(rng.normal(size=(1500, 8)).astype(np.float32))

    assert embedding.shape == (1500, 2)
    assert np.all(np.isfinite(embedding))


@pytest.mark.parametrize("n_epochs", [30, 150, 500])
def test_momentum_schedules_decay_within_short_runs(n_epochs):
    from umap.layouts import _create_alpha_schedule, _create_momentum_schedule

    alpha = _create_alpha_schedule("momentum", n_epochs, 1.0, True)
    momentum = _create_momentum_schedule("momentum", n_epochs, True)
    assert alpha.shape == momentum.shape == (n_epochs,)
    # The warm-up must finish within the run so the step size reaches its floor.
    assert alpha[-1] < 0.25 * alpha[0]
    assert momentum[-1] > momentum[0]


def test_csr_row_sort_check():
    from umap.layouts import _csr_rows_sorted

    graph = scipy.sparse.random(40, 40, density=0.2, format="csr", random_state=0)
    graph.sort_indices()
    assert _csr_rows_sorted(graph.indptr, graph.indices)
    row = slice(graph.indptr[5], graph.indptr[6])
    graph.indices[row] = graph.indices[row][::-1].copy()
    assert not _csr_rows_sorted(graph.indptr, graph.indices)


@pytest.mark.parametrize("n_points", [1000, 40_007])
@pytest.mark.parametrize("with_ties", [False, True])
def test_projection_order_is_near_exact(n_points, with_ties):
    from umap.layouts import (
        PROJECTION_MAX_BUCKET,
        PROJECTION_POINTS_PER_BUCKET,
        _projection_order_and_source_ranks,
    )

    embedding = np.random.RandomState(0).randn(n_points, 2).astype(np.float32)
    if with_ties:
        embedding[n_points // 2 :] = embedding[: n_points - n_points // 2]
    random_state = np.random.RandomState(1)
    expected_state = np.random.RandomState(1)
    direction = expected_state.randn(2)
    direction /= np.linalg.norm(direction)
    projection = np.dot(embedding, direction)
    exact_ranks = np.empty(n_points, dtype=np.int64)
    exact_ranks[np.argsort(projection, kind="stable")] = np.arange(n_points)

    order, ranks = _projection_order_and_source_ranks(
        embedding, embedding, random_state
    )

    assert order.dtype == np.int32 and ranks.dtype == np.int32
    np.testing.assert_array_equal(np.sort(order), np.arange(n_points))
    np.testing.assert_array_equal(ranks[order], np.arange(n_points))
    # Only order within a bucket is lost, so ranks stay close to exact.
    assert np.abs(ranks - exact_ranks).max() <= PROJECTION_MAX_BUCKET
    assert np.abs(ranks - exact_ranks).max() <= 64
    # Out-of-order neighbours share a bucket, so they differ by under its width.
    bucket_width = np.ptp(projection) / (n_points // PROJECTION_POINTS_PER_BUCKET)
    assert np.all(np.diff(projection[order]) >= -bucket_width * (1 + 1e-9))
    assert random_state.randint(2**31) == expected_state.randint(2**31)


def test_projection_order_falls_back_to_exact_sort_for_outliers():
    from umap.layouts import _projection_order_and_source_ranks

    embedding = np.random.RandomState(2).randn(5000, 2).astype(np.float32)
    embedding[17] = 1e6  # squeezes every other point into a few buckets
    random_state = np.random.RandomState(3)
    direction = np.random.RandomState(3).randn(2)
    direction /= np.linalg.norm(direction)
    order, ranks = _projection_order_and_source_ranks(
        embedding, embedding, random_state
    )
    np.testing.assert_array_equal(order, np.argsort(np.dot(embedding, direction)))
    np.testing.assert_array_equal(ranks[order], np.arange(5000))


def test_projection_ranks_for_transform_are_near_insertion_points():
    from umap.layouts import _projection_order_and_source_ranks

    rng = np.random.RandomState(5)
    tail = rng.randn(20_000, 2).astype(np.float32)
    head = rng.randn(3_000, 2).astype(np.float32) * 1.5
    random_state = np.random.RandomState(6)
    direction = np.random.RandomState(6).randn(2)
    direction /= np.linalg.norm(direction)
    order, ranks = _projection_order_and_source_ranks(head, tail, random_state)
    np.testing.assert_array_equal(np.sort(order), np.arange(20_000))
    exact = np.minimum(
        np.searchsorted(np.sort(np.dot(tail, direction)), np.dot(head, direction)),
        19_999,
    )
    assert ranks.shape == (3_000,) and ranks.min() >= 0 and ranks.max() <= 19_999
    assert np.abs(ranks.astype(np.int64) - exact).max() <= 64


@pytest.mark.parametrize("n_chunks", [1, 3, 12])
@pytest.mark.parametrize("kind", ["normal", "clustered", "ties"])
def test_parallel_argsort_matches_numpy(n_chunks, kind):
    from umap.layouts import _parallel_argsort_if_distinct

    rng = np.random.RandomState(4)
    if kind == "normal":
        values = rng.normal(size=50_000)
    elif kind == "clustered":
        values = np.concatenate(
            [rng.normal(loc=c, scale=1e-3, size=5_000) for c in range(10)]
        )
    else:
        values = np.round(rng.normal(size=50_000), 2)
    order, all_distinct = _parallel_argsort_if_distinct(values, n_chunks)
    assert all_distinct == (kind != "ties")
    np.testing.assert_array_equal(np.sort(order), np.arange(values.shape[0]))
    assert np.all(np.diff(values[order]) >= 0)
    if all_distinct:
        np.testing.assert_array_equal(order, np.argsort(values))


def test_parallel_random_permutation_is_valid_deterministic_and_uniform():
    from collections import Counter

    from umap.layouts import _parallel_random_permutation

    for n_values in (1, 2, 511, 70_000):
        permutation = _parallel_random_permutation(7, n_values, 4)
        assert permutation.dtype == np.int32
        np.testing.assert_array_equal(np.sort(permutation), np.arange(n_values))
    reference = _parallel_random_permutation(123, 70_000, 12)
    for n_chunks in (1, 3, 7):
        np.testing.assert_array_equal(
            reference, _parallel_random_permutation(123, 70_000, n_chunks)
        )
    assert not np.array_equal(reference, _parallel_random_permutation(124, 70_000, 12))
    counts = Counter(
        tuple(_parallel_random_permutation(seed, 3, 2)) for seed in range(30_000)
    )
    assert len(counts) == 6
    assert all(abs(c / 30_000 - 1 / 6) < 0.01 for c in counts.values())
    # Two buckets: every value should be equally likely to come first.
    first = np.bincount(
        [_parallel_random_permutation(seed, 1030, 3)[0] for seed in range(20_000)],
        minlength=1030,
    )
    halves = first[:515].sum(), first[515:].sum()
    assert abs(halves[0] - halves[1]) < 600


def test_unsorted_graph_rejected_when_excluding_neighbors():
    from umap.layouts import optimize_layout_euclidean

    graph = scipy.sparse.random(40, 40, density=0.2, format="csr", random_state=0)
    graph = (graph + graph.T).tocsr()
    graph.sort_indices()
    row = slice(graph.indptr[5], graph.indptr[6])
    graph.indices[row] = graph.indices[row][::-1].copy()
    embedding = np.random.RandomState(0).normal(size=(40, 2)).astype(np.float32)
    with pytest.raises(ValueError, match="sorted"):
        optimize_layout_euclidean(
            embedding,
            embedding,
            None,
            None,
            5,
            40,
            np.ones(graph.nnz),
            1.0,
            1.0,
            np.array([1, 2, 3], dtype=np.int64),
            csr_indptr=graph.indptr,
            csr_indices=graph.indices,
            csr_data=graph.data.astype(np.float32),
            exclude_graph_neighbors=True,
        )


def test_aligned_kernel_is_compiled_once():
    from umap.layouts import _get_aligned_epoch_fn

    assert _get_aligned_epoch_fn(True) is _get_aligned_epoch_fn(True)
    assert _get_aligned_epoch_fn(False) is not _get_aligned_epoch_fn(True)
