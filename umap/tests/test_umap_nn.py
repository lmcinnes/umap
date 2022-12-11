import numpy as np
import pytest
from numpy.testing import assert_array_almost_equal
from sklearn.neighbors import KDTree
from sklearn.preprocessing import normalize

from umap import distances as dist
from umap.umap_ import (
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


# -------------------------------------------------
#  Utility functions for Nearest Neighbour
# -------------------------------------------------


def knn(indices, nn_data):  # pragma: no cover
    tree = KDTree(nn_data)
    true_indices = tree.query(nn_data, 10, return_distance=False)
    num_correct = 0.0
    for i in range(nn_data.shape[0]):
        num_correct += np.sum(np.in1d(true_indices[i], indices[i]))
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
