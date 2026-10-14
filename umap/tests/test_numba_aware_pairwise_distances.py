import numba
import numpy as np
import pytest
import scipy.sparse
from sklearn.metrics import pairwise_distances

import umap.distances as dist
from umap import UMAP

METRICS = [
    "euclidean",
    "manhattan",
    "chebyshev",
    "cosine",
    "correlation",
    "canberra",
    "braycurtis",
]


@pytest.mark.parametrize("metric_name", METRICS)
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("shape", [(50, 7), (40, 64)])
def test_matches_sklearn_bit_for_bit(metric_name, dtype, shape):
    X = np.random.default_rng(0).normal(size=shape).astype(dtype)
    metric = dist.named_distances[metric_name]

    expected = pairwise_distances(X, metric=metric)
    result = dist.numba_aware_pairwise_distances(X, metric=metric)

    assert result.dtype == expected.dtype
    np.testing.assert_array_equal(result, expected)


def test_delegates_metric_kwds_to_sklearn():
    X = np.random.default_rng(1).normal(size=(20, 5))
    metric = dist.named_distances["minkowski"]

    np.testing.assert_array_equal(
        dist.numba_aware_pairwise_distances(X, metric=metric, p=3.0),
        pairwise_distances(X, metric=metric, p=3.0),
    )


def test_delegates_python_callables_and_strings_to_sklearn():
    X = np.random.default_rng(2).normal(size=(15, 4))

    def python_euclidean(x, y):
        return np.sqrt(np.sum((x - y) ** 2))

    np.testing.assert_array_equal(
        dist.numba_aware_pairwise_distances(X, metric=python_euclidean),
        pairwise_distances(X, metric=python_euclidean),
    )
    np.testing.assert_array_equal(
        dist.numba_aware_pairwise_distances(X, metric="cosine"),
        pairwise_distances(X, metric="cosine"),
    )


def test_delegates_sparse_input_to_sklearn():
    X = scipy.sparse.random(20, 10, density=0.3, format="csr", random_state=3)

    np.testing.assert_array_equal(
        dist.numba_aware_pairwise_distances(X, metric="euclidean"),
        pairwise_distances(X, metric="euclidean"),
    )


def test_non_finite_input_keeps_sklearn_error():
    X = np.random.default_rng(4).normal(size=(10, 4))
    X[0, 0] = np.nan

    with pytest.raises(ValueError):
        dist.numba_aware_pairwise_distances(X, metric=dist.named_distances["euclidean"])


def test_small_data_embedding_unchanged(monkeypatch):
    X = np.random.default_rng(5).normal(size=(300, 32))
    fast = UMAP(n_neighbors=10, n_epochs=50, random_state=42).fit(X)

    monkeypatch.setattr(
        dist,
        "numba_aware_pairwise_distances",
        lambda X, metric, **kwds: pairwise_distances(X, metric=metric, **kwds),
    )
    reference = UMAP(n_neighbors=10, n_epochs=50, random_state=42).fit(X)

    np.testing.assert_array_equal(fast.embedding_, reference.embedding_)
    assert (fast.graph_ != reference.graph_).nnz == 0


@numba.njit()
def _offset_manhattan(x, y):
    result = 1.0
    for i in range(x.shape[0]):
        result += np.abs(x[i] - y[i])
    return result


def test_diagonal_uses_the_metric_like_sklearn():
    X = np.random.default_rng(6).normal(size=(25, 6))

    result = dist.numba_aware_pairwise_distances(X, metric=_offset_manhattan)

    np.testing.assert_array_equal(
        result, pairwise_distances(X, metric=_offset_manhattan)
    )
    np.testing.assert_array_equal(np.diag(result), np.ones(25))
