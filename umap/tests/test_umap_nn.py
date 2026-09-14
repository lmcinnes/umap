import pickle

import numpy as np
import pytest
from numpy.testing import assert_array_almost_equal
from sklearn.neighbors import KDTree
from sklearn.preprocessing import normalize

from umap import distances as dist
from umap import umap_ as umap_module
from umap.umap_ import (
    _rinnd_eligibility,
    nearest_neighbors,
    smooth_knn_dist,
)

# ===================================================
#  Nearest Neighbour Test cases
# ===================================================


# nearest_neighbours metric parameter validation
# -----------------------------------------------
def test_nn_bad_metric(nn_data):
    with pytest.raises(ValueError):
        nearest_neighbors(nn_data, 10, 42, {}, False, np.random)


def test_nn_bad_metric_sparse_data(sparse_nn_data):
    with pytest.raises(ValueError):
        nearest_neighbors(
            sparse_nn_data,
            10,
            "seuclidean",
            {},
            False,
            np.random,
        )


def test_rinnd_eligibility_checks_supported_parameters(monkeypatch):
    monkeypatch.setattr(umap_module, "_HAVE_RINND", True)
    data = np.ones((32, 4), dtype=np.float32)

    eligible, reason = _rinnd_eligibility(data, "euclidean", {}, False, True, -1, False)

    assert eligible
    assert reason is None


@pytest.mark.parametrize(
    ("kwargs", "reason"),
    [
        ({"metric": "manhattan"}, "metric"),
        ({"metric_kwds": {"p": 2}}, "metric_kwds"),
        ({"low_memory": False}, "low_memory=False"),
        ({"n_jobs": 2}, "n_jobs"),
        ({"compatibility_layout": True}, "compatibility_layout"),
        ({"angular": True}, "angular"),
    ],
)
def test_rinnd_eligibility_rejects_unsupported_parameters(monkeypatch, kwargs, reason):
    monkeypatch.setattr(umap_module, "_HAVE_RINND", True)
    params = {
        "X": np.ones((32, 4), dtype=np.float32),
        "metric": "euclidean",
        "metric_kwds": {},
        "angular": False,
        "low_memory": True,
        "n_jobs": -1,
        "compatibility_layout": False,
    }
    params.update(kwargs)

    eligible, ineligible_reason = _rinnd_eligibility(**params)

    assert not eligible
    assert reason in ineligible_reason


def test_rinnd_eligibility_rejects_unsupported_input(monkeypatch, sparse_nn_data):
    monkeypatch.setattr(umap_module, "_HAVE_RINND", True)

    for data, reason in (
        (np.ones((32, 4), dtype=np.float64), "float32"),
        (np.asfortranarray(np.ones((32, 4), dtype=np.float32)), "C-contiguous"),
        (sparse_nn_data, "sparse"),
        (np.zeros((32, 4), dtype=np.float32), "zero vectors"),
    ):
        metric = "cosine" if reason == "zero vectors" else "euclidean"
        eligible, ineligible_reason = _rinnd_eligibility(
            data, metric, {}, metric == "cosine", True, -1, False
        )

        assert not eligible
        assert reason in ineligible_reason


def test_nearest_neighbors_uses_queryable_rinnd_without_mutating_input(monkeypatch):
    class FakeRINND:
        def __init__(self, data, n_neighbors, **kwargs):
            self.data = data.copy()
            self.kwargs = kwargs
            self.neighbor_graph = (
                np.tile(np.arange(n_neighbors), (data.shape[0], 1)),
                np.zeros((data.shape[0], n_neighbors), dtype=np.float32),
            )

        def query(self, data, k=10, epsilon=0.1):
            return (
                np.tile(np.arange(k), (data.shape[0], 1)),
                np.full((data.shape[0], k), epsilon, dtype=np.float32),
            )

    monkeypatch.setattr(umap_module, "RINND", FakeRINND)
    monkeypatch.setattr(umap_module, "_HAVE_RINND", True)
    data = np.arange(128, dtype=np.float32).reshape(32, 4) + 1.0
    original_data = data.copy()

    _, _, index = nearest_neighbors(
        data,
        5,
        "cosine",
        {},
        True,
        np.random.RandomState(42),
        nearest_neighbor_backend="rinnd",
    )
    indices, distances = index.query(data[:3], 2, epsilon=0.25)

    np.testing.assert_array_equal(data, original_data)
    assert index._angular_trees
    assert index._index.kwargs["normalize"]
    assert indices.shape == distances.shape == (3, 2)

    restored_index = pickle.loads(pickle.dumps(index))
    assert restored_index.neighbor_graph[0].shape == (32, 5)

    index.prepare()
    index.update(data[:4])
    assert index._raw_data.shape == (36, 4)
    assert index.neighbor_graph[0].shape == (36, 5)


def test_nearest_neighbors_falls_back_when_n_jobs_is_restricted(monkeypatch):
    class UnexpectedRINND:
        def __init__(self, *args, **kwargs):
            raise AssertionError("RINND should not be selected")

    class FakeNNDescent:
        def __init__(self, data, n_neighbors, **kwargs):
            self.kwargs = kwargs
            self.neighbor_graph = (
                np.tile(np.arange(n_neighbors), (data.shape[0], 1)),
                np.zeros((data.shape[0], n_neighbors), dtype=np.float32),
            )

    monkeypatch.setattr(umap_module, "RINND", UnexpectedRINND)
    monkeypatch.setattr(umap_module, "_HAVE_RINND", True)
    monkeypatch.setattr(umap_module, "NNDescent", FakeNNDescent)
    data = np.ones((32, 4), dtype=np.float32)

    _, _, index = nearest_neighbors(
        data, 5, "euclidean", {}, False, np.random.RandomState(42), n_jobs=2
    )

    assert isinstance(index, FakeNNDescent)
    assert index.kwargs["n_jobs"] == 2


def test_nearest_neighbors_forced_rinnd_reports_unsupported_parameter(monkeypatch):
    monkeypatch.setattr(umap_module, "_HAVE_RINND", True)
    data = np.ones((32, 4), dtype=np.float32)

    with pytest.raises(ValueError, match="restricted n_jobs"):
        nearest_neighbors(
            data,
            5,
            "euclidean",
            {},
            False,
            np.random.RandomState(42),
            n_jobs=1,
            nearest_neighbor_backend="rinnd",
        )


def test_nearest_neighbors_explicit_pynndescent(monkeypatch):
    class UnexpectedRINND:
        def __init__(self, *args, **kwargs):
            raise AssertionError("RINND should not be selected")

    class FakeNNDescent:
        def __init__(self, data, n_neighbors, **kwargs):
            self.neighbor_graph = (
                np.tile(np.arange(n_neighbors), (data.shape[0], 1)),
                np.zeros((data.shape[0], n_neighbors), dtype=np.float32),
            )

    monkeypatch.setattr(umap_module, "RINND", UnexpectedRINND)
    monkeypatch.setattr(umap_module, "_HAVE_RINND", True)
    monkeypatch.setattr(umap_module, "NNDescent", FakeNNDescent)

    _, _, index = nearest_neighbors(
        np.ones((32, 4), dtype=np.float32),
        5,
        "euclidean",
        {},
        False,
        np.random.RandomState(42),
        nearest_neighbor_backend="pynndescent",
    )

    assert isinstance(index, FakeNNDescent)


# -------------------------------------------------
#  Utility functions for Nearest Neighbour
# -------------------------------------------------


def knn(indices, nn_data):  # pragma: no cover
    tree = KDTree(nn_data)
    true_indices = tree.query(nn_data, 10, return_distance=False)
    num_correct = 0.0
    for i in range(nn_data.shape[0]):
        num_correct += np.sum(np.isin(true_indices[i], indices[i]))
    return num_correct / (nn_data.shape[0] * 10)


def smooth_knn(nn_data, local_connectivity=1.0):
    knn_indices, knn_dists, _ = nearest_neighbors(
        nn_data, 10, "euclidean", {}, False, np.random
    )
    sigmas, rhos = smooth_knn_dist(
        knn_dists, 10.0, local_connectivity=local_connectivity
    )
    shifted_dists = knn_dists - rhos[:, np.newaxis]
    shifted_dists[shifted_dists < 0.0] = 0.0
    vals = np.exp(-(shifted_dists / sigmas[:, np.newaxis]))
    norms = np.sum(vals, axis=1)
    return norms


@pytest.mark.skip()
def test_nn_descent_neighbor_accuracy(nn_data):  # pragma: no cover
    knn_indices, knn_dists, _ = nearest_neighbors(
        nn_data, 10, "euclidean", {}, False, np.random
    )
    percent_correct = knn(knn_indices, nn_data)
    assert (
        percent_correct >= 0.85
    ), "NN-descent did not get 89% accuracy on nearest neighbors"


@pytest.mark.skip()
def test_nn_descent_neighbor_accuracy_low_memory(nn_data):  # pragma: no cover
    knn_indices, knn_dists, _ = nearest_neighbors(
        nn_data, 10, "euclidean", {}, False, np.random, low_memory=True
    )
    percent_correct = knn(knn_indices, nn_data)
    assert (
        percent_correct >= 0.89
    ), "NN-descent did not get 89% accuracy on nearest neighbors"


@pytest.mark.skip()
def test_angular_nn_descent_neighbor_accuracy(nn_data):  # pragma: no cover
    knn_indices, knn_dists, _ = nearest_neighbors(
        nn_data, 10, "cosine", {}, True, np.random
    )
    angular_data = normalize(nn_data, norm="l2")
    percent_correct = knn(knn_indices, angular_data)
    assert (
        percent_correct >= 0.85
    ), "NN-descent did not get 89% accuracy on nearest neighbors"


@pytest.mark.skip()
def test_sparse_nn_descent_neighbor_accuracy(sparse_nn_data):  # pragma: no cover
    knn_indices, knn_dists, _ = nearest_neighbors(
        sparse_nn_data, 20, "euclidean", {}, False, np.random
    )
    percent_correct = knn(knn_indices, sparse_nn_data.todense())
    assert (
        percent_correct >= 0.75
    ), "Sparse NN-descent did not get 90% accuracy on nearest neighbors"


@pytest.mark.skip()
def test_sparse_nn_descent_neighbor_accuracy_low_memory(
    sparse_nn_data,
):  # pragma: no cover
    knn_indices, knn_dists, _ = nearest_neighbors(
        sparse_nn_data, 20, "euclidean", {}, False, np.random, low_memory=True
    )
    percent_correct = knn(knn_indices, sparse_nn_data.todense())
    assert (
        percent_correct >= 0.85
    ), "Sparse NN-descent did not get 90% accuracy on nearest neighbors"


@pytest.mark.skip()
def test_nn_descent_neighbor_accuracy_callable_metric(nn_data):  # pragma: no cover
    knn_indices, knn_dists, _ = nearest_neighbors(
        nn_data, 10, dist.euclidean, {}, False, np.random
    )

    percent_correct = knn(knn_indices, nn_data)
    assert (
        percent_correct >= 0.95
    ), "NN-descent did not get 95% accuracy on nearest neighbors with callable metric"


@pytest.mark.skip()
def test_sparse_angular_nn_descent_neighbor_accuracy(
    sparse_nn_data,
):  # pragma: no cover
    knn_indices, knn_dists, _ = nearest_neighbors(
        sparse_nn_data, 20, "cosine", {}, True, np.random
    )
    angular_data = normalize(sparse_nn_data, norm="l2").toarray()
    percent_correct = knn(knn_indices, angular_data)
    assert (
        percent_correct >= 0.90
    ), "Sparse NN-descent did not get 90% accuracy on nearest neighbors"


def test_smooth_knn_dist_l1norms(nn_data):
    norms = smooth_knn(nn_data)
    assert_array_almost_equal(
        norms,
        1.0 + np.log2(10) * np.ones(norms.shape[0]),
        decimal=3,
        err_msg="Smooth knn-dists does not give expected" "norms",
    )


def test_smooth_knn_dist_l1norms_w_connectivity(nn_data):
    norms = smooth_knn(nn_data, local_connectivity=1.75)
    assert_array_almost_equal(
        norms,
        1.0 + np.log2(10) * np.ones(norms.shape[0]),
        decimal=3,
        err_msg="Smooth knn-dists does not give expected"
        "norms for local_connectivity=1.75",
    )
