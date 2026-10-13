import pickle

import numpy as np
import pytest
from numpy.testing import assert_array_almost_equal
from sklearn.neighbors import KDTree
from sklearn.preprocessing import normalize

from umap import distances as dist
from umap import UMAP
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


class FakeRINND:
    instances = []

    def __init__(self, data, n_neighbors, **kwargs):
        self.data = np.array(data, copy=True)
        self.kwargs = kwargs
        self.is_prepared = False
        n_samples = data.shape[0]
        offsets = np.arange(1, n_neighbors + 1)
        # Like RINND, the graph never lists a point as its own neighbor.
        self.neighbor_graph = (
            ((np.arange(n_samples)[:, None] + offsets) % n_samples).astype(np.int32),
            np.tile(offsets.astype(np.float32), (n_samples, 1)),
        )
        FakeRINND.instances.append(self)

    def prepare(self):
        self.is_prepared = True

    def query(self, data, k=10, epsilon=0.1):
        return (
            np.tile(np.arange(k, dtype=np.int32), (data.shape[0], 1)),
            np.full((data.shape[0], k), epsilon, dtype=np.float32),
        )


class FakeNNDescent:
    def __init__(self, data, n_neighbors, **kwargs):
        self.kwargs = kwargs
        self.neighbor_graph = (
            np.tile(np.arange(n_neighbors, dtype=np.int32), (data.shape[0], 1)),
            np.zeros((data.shape[0], n_neighbors), dtype=np.float32),
        )


@pytest.fixture
def fake_rinnd(monkeypatch):
    FakeRINND.instances = []
    monkeypatch.setattr(umap_module, "RINND", FakeRINND)
    monkeypatch.setattr(umap_module, "_HAVE_RINND", True)
    monkeypatch.setattr(
        umap_module,
        "rinnd_named_distances",
        dict.fromkeys(umap_module.RINND_METRICS, True),
    )
    monkeypatch.setattr(umap_module, "NNDescent", FakeNNDescent)
    return FakeRINND


@pytest.mark.parametrize(
    "metric", ["euclidean", "manhattan", "cosine", "correlation", "jaccard"]
)
def test_rinnd_eligibility_accepts_verified_metrics(fake_rinnd, metric):
    data = np.asfortranarray(np.ones((32, 4), dtype=np.float32))

    eligible, reason = _rinnd_eligibility(data, 10, metric, {}, False)

    assert eligible
    assert reason is None


@pytest.mark.parametrize(
    ("kwargs", "reason"),
    [
        ({"metric": "dot"}, "metric"),
        ({"metric": "russellrao"}, "metric"),
        ({"metric": dist.euclidean}, "metric"),
        ({"metric_kwds": {"p": 2}}, "metric_kwds"),
        ({"compatibility_layout": True}, "compatibility_layout"),
        ({"n_neighbors": 1}, "n_neighbors"),
        ({"X": np.ones((32, 4), dtype=np.float64)}, "float32"),
    ],
)
def test_rinnd_eligibility_rejects_unsupported_parameters(fake_rinnd, kwargs, reason):
    params = {
        "X": np.ones((32, 4), dtype=np.float32),
        "n_neighbors": 10,
        "metric": "euclidean",
        "metric_kwds": {},
        "compatibility_layout": False,
    }
    params.update(kwargs)

    eligible, ineligible_reason = _rinnd_eligibility(**params)

    assert not eligible
    assert reason in ineligible_reason


def test_rinnd_eligibility_rejects_sparse_and_outdated_rinnd(
    fake_rinnd, monkeypatch, sparse_nn_data
):
    assert "sparse" in _rinnd_eligibility(sparse_nn_data, 10, "euclidean", {}, False)[1]

    monkeypatch.setattr(umap_module, "rinnd_named_distances", {"euclidean": True})
    data = np.ones((32, 4), dtype=np.float32)
    assert "installed RINND" in _rinnd_eligibility(data, 10, "cosine", {}, False)[1]

    monkeypatch.setattr(umap_module, "_HAVE_RINND", False)
    assert "0.3.1" in _rinnd_eligibility(data, 10, "euclidean", {}, False)[1]


def test_nearest_neighbors_rinnd_honors_n_jobs_and_includes_self(fake_rinnd):
    data = np.arange(128, dtype=np.float32).reshape(32, 4) + 1.0
    original_data = data.copy()

    indices, dists, index = nearest_neighbors(
        data,
        5,
        "cosine",
        {},
        True,
        np.random.RandomState(42),
        n_jobs=3,
        nearest_neighbor_backend="rinnd",
    )
    built = fake_rinnd.instances[-1]

    np.testing.assert_array_equal(data, original_data)
    assert built.kwargs["n_jobs"] == 3
    assert built.kwargs["normalize"]
    assert not built.kwargs["input_normalized"]
    assert not built.kwargs["graph_only"]
    assert built.neighbor_graph[0].shape == (32, 5)
    np.testing.assert_array_equal(indices[:, 0], np.arange(32))
    np.testing.assert_array_equal(indices[:, 1:], built.neighbor_graph[0][:, :4])
    np.testing.assert_array_equal(dists[:, 0], 0.0)
    assert indices.shape == dists.shape == (32, 5)
    assert index._angular_trees


def test_nearest_neighbors_rinnd_skips_normalizing_normalized_cosine_data(fake_rinnd):
    data = normalize(np.random.RandomState(0).normal(size=(32, 4))).astype(np.float32)

    nearest_neighbors(data, 5, "cosine", {}, True, 42, nearest_neighbor_backend="rinnd")

    assert fake_rinnd.instances[-1].kwargs["input_normalized"]


def test_nearest_neighbors_graph_only_returns_no_index(fake_rinnd):
    data = np.ones((32, 4), dtype=np.float32)

    indices, _, index = nearest_neighbors(
        data,
        5,
        "euclidean",
        {},
        False,
        42,
        nearest_neighbor_backend="rinnd",
        graph_only=True,
    )

    assert index is None
    assert fake_rinnd.instances[-1].kwargs["graph_only"]
    assert "normalize" not in fake_rinnd.instances[-1].kwargs
    assert indices.shape == (32, 5)


def test_rinnd_index_pickles_without_rebuilding_until_queried(fake_rinnd):
    data = np.arange(128, dtype=np.float32).reshape(32, 4) + 1.0
    _, _, index = nearest_neighbors(
        data, 5, "euclidean", {}, False, 42, nearest_neighbor_backend="rinnd"
    )
    n_builds = len(fake_rinnd.instances)

    restored = pickle.loads(pickle.dumps(index))

    assert len(fake_rinnd.instances) == n_builds
    np.testing.assert_array_equal(restored.neighbor_graph[0], index.neighbor_graph[0])
    indices, _ = restored.query(data[:3], 2, epsilon=0.25)
    assert len(fake_rinnd.instances) == n_builds + 1
    assert indices.shape == (3, 2)


def test_rinnd_index_prepare_delegates_and_rebuilds_after_unpickle(fake_rinnd):
    data = np.arange(128, dtype=np.float32).reshape(32, 4) + 1.0
    _, _, index = nearest_neighbors(
        data, 5, "euclidean", {}, False, 42, nearest_neighbor_backend="rinnd"
    )

    assert not index._index.is_prepared
    index.prepare()
    assert index._index.is_prepared

    restored = pickle.loads(pickle.dumps(index))
    n_builds = len(fake_rinnd.instances)
    restored.prepare()

    assert len(fake_rinnd.instances) == n_builds + 1
    assert restored._index.is_prepared


def test_rinnd_index_update_rebuilds_with_appended_data(fake_rinnd):
    data = normalize(np.random.RandomState(0).normal(size=(32, 4))).astype(np.float32)
    _, _, index = nearest_neighbors(
        data, 5, "cosine", {}, True, 42, nearest_neighbor_backend="rinnd"
    )

    index.update(4.0 * data[:4])

    assert index._raw_data.shape == (36, 4)
    assert index.neighbor_graph[0].shape == (36, 5)
    assert not fake_rinnd.instances[-1].kwargs["input_normalized"]
    assert not index._index.is_prepared


def test_nearest_neighbors_auto_selects_rinnd_by_size(fake_rinnd):
    small = np.ones((32, 4), dtype=np.float32)
    large = np.ones((umap_module.RINND_MIN_SAMPLES, 4), dtype=np.float32)

    _, _, small_index = nearest_neighbors(small, 5, "euclidean", {}, False, 42)
    _, _, large_index = nearest_neighbors(
        large, 5, "euclidean", {}, False, 42, n_jobs=2
    )

    assert isinstance(small_index, FakeNNDescent)
    assert isinstance(large_index, umap_module._RINNDIndex)
    assert fake_rinnd.instances[-1].kwargs["n_jobs"] == 2


def test_nearest_neighbors_forced_rinnd_reports_unsupported_parameter(fake_rinnd):
    data = np.ones((32, 4), dtype=np.float32)

    with pytest.raises(ValueError, match="metric_kwds"):
        nearest_neighbors(
            data,
            5,
            "euclidean",
            {"p": 2},
            False,
            42,
            nearest_neighbor_backend="rinnd",
        )


def test_nearest_neighbors_explicit_pynndescent(fake_rinnd):
    large = np.ones((umap_module.RINND_MIN_SAMPLES, 4), dtype=np.float32)

    _, _, index = nearest_neighbors(
        large, 5, "euclidean", {}, False, 42, nearest_neighbor_backend="pynndescent"
    )

    assert isinstance(index, FakeNNDescent)
    assert not fake_rinnd.instances


@pytest.mark.skipif(not umap_module._HAVE_RINND, reason="requires RINND >= 0.3.1")
@pytest.mark.parametrize("metric", ["euclidean", "cosine"])
def test_real_rinnd_graph_matches_pynndescent_layout(metric):
    data = np.random.RandomState(0).normal(size=(600, 8)).astype(np.float32)

    indices, dists, index = nearest_neighbors(
        data,
        10,
        metric,
        {},
        metric == "cosine",
        42,
        n_jobs=2,
        nearest_neighbor_backend="rinnd",
    )
    graph_indices, _, graph_index = nearest_neighbors(
        data,
        10,
        metric,
        {},
        metric == "cosine",
        42,
        n_jobs=2,
        nearest_neighbor_backend="rinnd",
        graph_only=True,
    )

    np.testing.assert_array_equal(indices[:, 0], np.arange(600))
    np.testing.assert_array_equal(dists[:, 0], 0.0)
    assert np.all(np.diff(dists, axis=1) >= 0)
    assert graph_index is None
    assert graph_indices.shape == indices.shape
    query_indices, _ = index.query(data[:5], 3)
    np.testing.assert_array_equal(query_indices[:, 0], np.arange(5))


@pytest.mark.skipif(not umap_module._HAVE_RINND, reason="requires RINND >= 0.3.1")
def test_real_rinnd_prepares_lazily_and_update_skips_prepare(monkeypatch):
    data = np.random.RandomState(0).normal(size=(300, 8)).astype(np.float32)
    model = UMAP(
        n_neighbors=10,
        n_epochs=10,
        force_approximation_algorithm=True,
        nearest_neighbor_backend="rinnd",
    ).fit(data[:250])
    index = model._knn_search_index

    assert not index._index.is_prepared
    index.prepare()
    assert index._index.is_prepared

    def fail_prepare(self):
        raise AssertionError("RINND indexes are rebuilt, so update must not prepare")

    monkeypatch.setattr(umap_module._RINNDIndex, "prepare", fail_prepare)
    model.update(data[250:])

    assert model.embedding_.shape == (300, 2)
    assert not model._knn_search_index._index.is_prepared


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
