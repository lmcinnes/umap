import numpy as np
from nose import SkipTest
from nose.tools import assert_greater_equal, assert_raises
from numpy.testing import assert_array_almost_equal
from scipy import sparse
from sklearn.neighbors import KDTree
from sklearn.preprocessing import normalize

from umap import distances as dist, sparse as spdist
from umap.umap_ import (
    INT32_MAX,
    INT32_MIN,
    nearest_neighbors,
    smooth_knn_dist,
)
from umap.utils import deheap_sort


# ===================================================
#  Nearest Neighbour Test cases
# ===================================================

# nearest_neighbours metric parameter validation
# -----------------------------------------------
def test_nn_bad_metric(nn_data):
    assert_raises(ValueError, nearest_neighbors, nn_data, 10, 42, {}, False, np.random)


def test_nn_bad_metric_sparse_data(sparse_nn_data):
    assert_raises(
        ValueError,
        nearest_neighbors,
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


def knn(indices, nn_data):
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


def test_nn_descent_neighbor_accuracy(nn_data):
    knn_indices, knn_dists, _ = nearest_neighbors(
        nn_data, 10, "euclidean", {}, False, np.random
    )
    percent_correct = knn(knn_indices, nn_data)
    assert_greater_equal(
        percent_correct,
        0.89,
        "NN-descent did not get 89% accuracy on nearest neighbors",
    )


def test_nn_descent_neighbor_accuracy_low_memory(nn_data):
    knn_indices, knn_dists, _ = nearest_neighbors(
        nn_data, 10, "euclidean", {}, False, np.random, low_memory=True
    )
    percent_correct = knn(knn_indices, nn_data)
    assert_greater_equal(
        percent_correct,
        0.89,
        "NN-descent did not get 89% accuracy on nearest neighbors",
    )


def test_angular_nn_descent_neighbor_accuracy(nn_data):
    knn_indices, knn_dists, _ = nearest_neighbors(
        nn_data, 10, "cosine", {}, True, np.random
    )
    angular_data = normalize(nn_data, norm="l2")
    percent_correct = knn(knn_indices, angular_data)
    assert_greater_equal(
        percent_correct,
        0.89,
        "NN-descent did not get 89% accuracy on nearest neighbors",
    )


def test_sparse_nn_descent_neighbor_accuracy(sparse_nn_data):
    knn_indices, knn_dists, _ = nearest_neighbors(
        sparse_nn_data, 20, "euclidean", {}, False, np.random
    )
    percent_correct = knn(knn_indices, sparse_nn_data.todense())
    assert_greater_equal(
        percent_correct,
        0.90,
        "Sparse NN-descent did not get 90% accuracy on nearest neighbors",
    )


def test_sparse_nn_descent_neighbor_accuracy_low_memory(sparse_nn_data):
    knn_indices, knn_dists, _ = nearest_neighbors(
        sparse_nn_data, 20, "euclidean", {}, False, np.random, low_memory=True
    )
    percent_correct = knn(knn_indices, sparse_nn_data.todense())
    assert_greater_equal(
        percent_correct,
        0.90,
        "Sparse NN-descent did not get 90% accuracy on nearest neighbors",
    )


def test_nn_descent_neighbor_accuracy_callable_metric(nn_data):
    knn_indices, knn_dists, _ = nearest_neighbors(
        nn_data, 10, dist.euclidean, {}, False, np.random
    )

    percent_correct = knn(knn_indices, nn_data)
    assert_greater_equal(
        percent_correct,
        0.95,
        "NN-descent did not get 95% "
        "accuracy on nearest neighbors with callable metric",
    )


@SkipTest
def test_sparse_angular_nn_descent_neighbor_accuracy(sparse_nn_data):
    knn_indices, knn_dists, _ = nearest_neighbors(
        sparse_nn_data, 20, "cosine", {}, True, np.random
    )
    angular_data = normalize(sparse_nn_data, norm="l2").toarray()
    percent_correct = knn(knn_indices, angular_data)
    assert_greater_equal(
        percent_correct,
        0.90,
        "Sparse NN-descent did not get 90% accuracy on nearest neighbors",
    )


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

    # sigmas, rhos = smooth_knn_dist(knn_dists, 10, local_connectivity=0.75)
    # shifted_dists = knn_dists - rhos[:, np.newaxis]
    # shifted_dists[shifted_dists < 0.0] = 0.0
    # vals = np.exp(-(shifted_dists / sigmas[:, np.newaxis]))
    # norms = np.sum(vals, axis=1)
    # diff = np.mean(norms) - (1.0 + np.log2(10))
    #
    # assert_almost_equal(diff, 0.0, decimal=1,
    #                     err_msg='Smooth knn-dists does not give expected'
    #                             'norms for local_connectivity=0.75')


# ===================================================
#  Nearest Neighbour Search Test cases
# ===================================================

# ------------------------------
# Utility Function for NN-Search
# ------------------------------
def setup_search_graph(knn_dists, knn_indices, train):
    search_graph = sparse.lil_matrix((train.shape[0], train.shape[0]), dtype=np.int8)
    search_graph.rows[:] = [inds.tolist() for inds in knn_indices]
    search_graph.data[:] = [vals.tolist() for vals in (knn_dists != 0).astype(np.int8)]
    search_graph = search_graph.maximum(search_graph.transpose()).tocsr()
    return search_graph
