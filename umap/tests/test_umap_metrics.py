import numpy as np
from numpy.testing import assert_array_almost_equal
import umap.distances as dist
import umap.sparse as spdist

from sklearn.metrics import pairwise_distances
from sklearn.neighbors import BallTree
from scipy.version import full_version as scipy_full_version_
import pytest


scipy_full_version = tuple(int(n) for n in scipy_full_version_.split("."))


# ===================================================
#  Metrics Test cases
# ===================================================

# ----------------------------------
# Utility functions for Metric tests
# ----------------------------------


def run_test_metric(metric, test_data, dist_matrix, with_grad=False):
    """Core utility function to test target metric on test data"""
    if with_grad:
        dist_function = dist.named_distances_with_gradients[metric]
    else:
        dist_function = dist.named_distances[metric]
    sample_size = test_data.shape[0]
    test_matrix = [
        [dist_function(test_data[i], test_data[j]) for j in range(sample_size)]
        for i in range(sample_size)
    ]
    if with_grad:
        test_matrix = [d for pairs in test_matrix for d, grad in pairs]

    test_matrix = np.array(test_matrix).reshape(sample_size, sample_size)

    assert_array_almost_equal(
        test_matrix,
        dist_matrix,
        err_msg="Distances don't match " "for metric {}".format(metric),
    )


def spatial_check(metric, spatial_data, spatial_distances, with_grad=False):
    # Check that metric is supported for this test, otherwise, fail!
    assert metric in spatial_distances, f"{metric} not valid for spatial data"
    dist_matrix = pairwise_distances(spatial_data, metric=metric)
    # scipy is bad sometimes
    if metric == "braycurtis":
        dist_matrix[np.where(~np.isfinite(dist_matrix))] = 0.0

    if metric in ("cosine", "correlation"):
        dist_matrix[np.where(~np.isfinite(dist_matrix))] = 1.0
        # And because distance between all zero vectors should be zero
        dist_matrix[10, 11] = 0.0
        dist_matrix[11, 10] = 0.0

    run_test_metric(metric, spatial_data, dist_matrix, with_grad=with_grad)


def binary_check(metric, binary_data, binary_distances):
    # Check that metric is supported for this test, otherwise, fail!
    assert metric in binary_distances, f"{metric} not valid for binary data"
    dist_matrix = pairwise_distances(binary_data, metric=metric)

    if metric in ("jaccard", "dice", "sokalsneath", "yule"):
        dist_matrix[np.where(~np.isfinite(dist_matrix))] = 0.0

    if metric in ("kulsinski", "russellrao"):
        dist_matrix[np.where(~np.isfinite(dist_matrix))] = 0.0
        # And because distance between all zero vectors should be zero
        dist_matrix[10, 11] = 0.0
        dist_matrix[11, 10] = 0.0

    run_test_metric(metric, binary_data, dist_matrix)


def run_test_sparse_metric(metric, sparse_test_data, dist_matrix):
    """Core utility function to run test of target metric on sparse data"""
    dist_function = spdist.sparse_named_distances[metric]
    if metric in spdist.sparse_need_n_features:
        test_matrix = np.array(
            [
                [
                    dist_function(
                        sparse_test_data[i].indices,
                        sparse_test_data[i].data,
                        sparse_test_data[j].indices,
                        sparse_test_data[j].data,
                        sparse_test_data.shape[1],
                    )
                    for j in range(sparse_test_data.shape[0])
                ]
                for i in range(sparse_test_data.shape[0])
            ]
        )
    else:
        test_matrix = np.array(
            [
                [
                    dist_function(
                        sparse_test_data[i].indices,
                        sparse_test_data[i].data,
                        sparse_test_data[j].indices,
                        sparse_test_data[j].data,
                    )
                    for j in range(sparse_test_data.shape[0])
                ]
                for i in range(sparse_test_data.shape[0])
            ]
        )
    assert_array_almost_equal(
        test_matrix,
        dist_matrix,
        err_msg="Sparse distances don't match " "for metric {}".format(metric),
    )


def sparse_spatial_check(metric, sparse_spatial_data):
    # Check that metric is supported for this test, otherwise, fail!
    assert (
        metric in spdist.sparse_named_distances
    ), f"{metric} not supported for sparse data"
    dist_matrix = pairwise_distances(np.asarray(sparse_spatial_data.todense()), metric=metric)

    if metric in ("braycurtis", "dice", "sokalsneath", "yule"):
        dist_matrix[np.where(~np.isfinite(dist_matrix))] = 0.0

    if metric in ("cosine", "correlation", "kulsinski", "russellrao"):
        dist_matrix[np.where(~np.isfinite(dist_matrix))] = 1.0
        # And because distance between all zero vectors should be zero
        dist_matrix[10, 11] = 0.0
        dist_matrix[11, 10] = 0.0

    run_test_sparse_metric(metric, sparse_spatial_data, dist_matrix)


def sparse_binary_check(metric, sparse_binary_data):
    # Check that metric is supported for this test, otherwise, fail!
    assert (
        metric in spdist.sparse_named_distances
    ), f"{metric} not supported for sparse data"
    dist_matrix = pairwise_distances(np.asarray(sparse_binary_data.todense()), metric=metric)
    if metric in ("jaccard", "dice", "sokalsneath", "yule"):
        dist_matrix[np.where(~np.isfinite(dist_matrix))] = 0.0

    if metric in ("kulsinski", "russellrao"):
        dist_matrix[np.where(~np.isfinite(dist_matrix))] = 1.0
        # And because distance between all zero vectors should be zero
        dist_matrix[10, 11] = 0.0
        dist_matrix[11, 10] = 0.0

    run_test_sparse_metric(metric, sparse_binary_data, dist_matrix)


# --------------------
# Spatial Metric Tests
# --------------------


def test_euclidean(spatial_data, spatial_distances):
    spatial_check("euclidean", spatial_data, spatial_distances)


def test_manhattan(spatial_data, spatial_distances):
    spatial_check("manhattan", spatial_data, spatial_distances)


def test_chebyshev(spatial_data, spatial_distances):
    spatial_check("chebyshev", spatial_data, spatial_distances)


def test_minkowski(spatial_data, spatial_distances):
    spatial_check("minkowski", spatial_data, spatial_distances)


def test_hamming(spatial_data, spatial_distances):
    spatial_check("hamming", spatial_data, spatial_distances)


def test_canberra(spatial_data, spatial_distances):
    spatial_check("canberra", spatial_data, spatial_distances)


def test_braycurtis(spatial_data, spatial_distances):
    spatial_check("braycurtis", spatial_data, spatial_distances)


def test_cosine(spatial_data, spatial_distances):
    spatial_check("cosine", spatial_data, spatial_distances)


def test_correlation(spatial_data, spatial_distances):
    spatial_check("correlation", spatial_data, spatial_distances)


# --------------------
# Binary Metric Tests
# --------------------


def test_jaccard(binary_data, binary_distances):
    binary_check("jaccard", binary_data, binary_distances)


def test_matching(binary_data, binary_distances):
    binary_check("matching", binary_data, binary_distances)


def test_dice(binary_data, binary_distances):
    binary_check("dice", binary_data, binary_distances)


@pytest.mark.skipif(
    scipy_full_version >= (1, 9), reason="deprecated in SciPy 1.9, removed in 1.11"
)
def test_kulsinski(binary_data, binary_distances):
    binary_check("kulsinski", binary_data, binary_distances)


def test_rogerstanimoto(binary_data, binary_distances):
    binary_check("rogerstanimoto", binary_data, binary_distances)


def test_russellrao(binary_data, binary_distances):
    binary_check("russellrao", binary_data, binary_distances)


def test_sokalmichener(binary_data, binary_distances):
    binary_check("sokalmichener", binary_data, binary_distances)


def test_sokalsneath(binary_data, binary_distances):
    binary_check("sokalsneath", binary_data, binary_distances)


def test_yule(binary_data, binary_distances):
    binary_check("yule", binary_data, binary_distances)


# ---------------------------
# Sparse Spatial Metric Tests
# ---------------------------


def test_sparse_euclidean(sparse_spatial_data):
    sparse_spatial_check("euclidean", sparse_spatial_data)


def test_sparse_manhattan(sparse_spatial_data):
    sparse_spatial_check("manhattan", sparse_spatial_data)


def test_sparse_chebyshev(sparse_spatial_data):
    sparse_spatial_check("chebyshev", sparse_spatial_data)


def test_sparse_minkowski(sparse_spatial_data):
    sparse_spatial_check("minkowski", sparse_spatial_data)


def test_sparse_hamming(sparse_spatial_data):
    sparse_spatial_check("hamming", sparse_spatial_data)


def test_sparse_canberra(sparse_spatial_data):
    sparse_spatial_check("canberra", sparse_spatial_data)


def test_sparse_cosine(sparse_spatial_data):
    sparse_spatial_check("cosine", sparse_spatial_data)


def test_sparse_correlation(sparse_spatial_data):
    sparse_spatial_check("correlation", sparse_spatial_data)


def test_sparse_braycurtis(sparse_spatial_data):
    sparse_spatial_check("braycurtis", sparse_spatial_data)


# ---------------------------
# Sparse Binary Metric Tests
# ---------------------------


def test_sparse_jaccard(sparse_binary_data):
    sparse_binary_check("jaccard", sparse_binary_data)


def test_sparse_matching(sparse_binary_data):
    sparse_binary_check("matching", sparse_binary_data)


def test_sparse_dice(sparse_binary_data):
    sparse_binary_check("dice", sparse_binary_data)


@pytest.mark.skipif(
    scipy_full_version >= (1, 9), reason="deprecated in SciPy 1.9, removed in 1.11"
)
def test_sparse_kulsinski(sparse_binary_data):
    sparse_binary_check("kulsinski", sparse_binary_data)


def test_sparse_rogerstanimoto(sparse_binary_data):
    sparse_binary_check("rogerstanimoto", sparse_binary_data)


def test_sparse_russellrao(sparse_binary_data):
    sparse_binary_check("russellrao", sparse_binary_data)


def test_sparse_sokalmichener(sparse_binary_data):
    sparse_binary_check("sokalmichener", sparse_binary_data)


def test_sparse_sokalsneath(sparse_binary_data):
    sparse_binary_check("sokalsneath", sparse_binary_data)


# --------------------------------
# Standardised/weighted Distances
# --------------------------------
def test_seuclidean(spatial_data):
    v = np.abs(np.random.randn(spatial_data.shape[1]))
    dist_matrix = pairwise_distances(spatial_data, metric="seuclidean", V=v)
    test_matrix = np.array(
        [
            [
                dist.standardised_euclidean(spatial_data[i], spatial_data[j], v)
                for j in range(spatial_data.shape[0])
            ]
            for i in range(spatial_data.shape[0])
        ]
    )
    assert_array_almost_equal(
        test_matrix,
        dist_matrix,
        err_msg="Distances don't match " "for metric seuclidean",
    )

@pytest.mark.skipif(
    scipy_full_version < (1, 8), reason="incorrect function in scipy<1.8"
)
def test_weighted_minkowski(spatial_data):
    v = np.abs(np.random.randn(spatial_data.shape[1]))
    dist_matrix = pairwise_distances(spatial_data, metric="minkowski", w=v, p=3)
    test_matrix = np.array(
        [
            [
                dist.weighted_minkowski(spatial_data[i], spatial_data[j], v, p=3)
                for j in range(spatial_data.shape[0])
            ]
            for i in range(spatial_data.shape[0])
        ]
    )
    assert_array_almost_equal(
        test_matrix,
        dist_matrix,
        err_msg="Distances don't match " "for metric weighted_minkowski",
    )


def test_mahalanobis(spatial_data):
    v = np.cov(np.transpose(spatial_data))
    dist_matrix = pairwise_distances(spatial_data, metric="mahalanobis", VI=v)
    test_matrix = np.array(
        [
            [
                dist.mahalanobis(spatial_data[i], spatial_data[j], v)
                for j in range(spatial_data.shape[0])
            ]
            for i in range(spatial_data.shape[0])
        ]
    )
    assert_array_almost_equal(
        test_matrix,
        dist_matrix,
        err_msg="Distances don't match " "for metric mahalanobis",
    )


def test_haversine(spatial_data):
    tree = BallTree(spatial_data[:, :2], metric="haversine")
    dist_matrix, _ = tree.query(spatial_data[:, :2], k=spatial_data.shape[0])
    test_matrix = np.array(
        [
            [
                dist.haversine(spatial_data[i, :2], spatial_data[j, :2])
                for j in range(spatial_data.shape[0])
            ]
            for i in range(spatial_data.shape[0])
        ]
    )
    test_matrix.sort(axis=1)
    assert_array_almost_equal(
        test_matrix,
        dist_matrix,
        err_msg="Distances don't match " "for metric haversine",
    )


def test_hellinger(spatial_data):
    hellinger_data = np.abs(spatial_data[:-2].copy())
    hellinger_data = hellinger_data / hellinger_data.sum(axis=1)[:, np.newaxis]
    hellinger_data = np.sqrt(hellinger_data)
    dist_matrix = hellinger_data @ hellinger_data.T
    dist_matrix = 1.0 - dist_matrix
    dist_matrix = np.sqrt(dist_matrix)
    # Correct for nan handling
    dist_matrix[np.isnan(dist_matrix)] = 0.0

    test_matrix = dist.pairwise_special_metric(np.abs(spatial_data[:-2]))

    assert_array_almost_equal(
        test_matrix,
        dist_matrix,
        err_msg="Distances don't match " "for metric hellinger",
    )

    # Ensure ll_dirichlet runs
    test_matrix = dist.pairwise_special_metric(
        np.abs(spatial_data[:-2]), metric="ll_dirichlet"
    )
    assert (
        test_matrix is not None
    ), "Pairwise Special Metric with LL Dirichlet metric failed"


def test_sparse_hellinger(sparse_spatial_data):
    dist_matrix = dist.pairwise_special_metric(
        np.abs(sparse_spatial_data[:-2].toarray())
    )
    test_matrix = np.array(
        [
            [
                spdist.sparse_hellinger(
                    np.abs(sparse_spatial_data[i]).indices,
                    np.abs(sparse_spatial_data[i]).data,
                    np.abs(sparse_spatial_data[j]).indices,
                    np.abs(sparse_spatial_data[j]).data,
                )
                for j in range(sparse_spatial_data.shape[0] - 2)
            ]
            for i in range(sparse_spatial_data.shape[0] - 2)
        ]
    )

    assert_array_almost_equal(
        test_matrix,
        dist_matrix,
        err_msg="Sparse distances don't match " "for metric hellinger",
        decimal=3,
    )

    # Ensure ll_dirichlet runs
    test_matrix = np.array(
        [
            [
                spdist.sparse_ll_dirichlet(
                    sparse_spatial_data[i].indices,
                    sparse_spatial_data[i].data,
                    sparse_spatial_data[j].indices,
                    sparse_spatial_data[j].data,
                )
                for j in range(sparse_spatial_data.shape[0])
            ]
            for i in range(sparse_spatial_data.shape[0])
        ]
    )
    assert (
        test_matrix is not None
    ), "Pairwise Special Metric with LL Dirichlet metric failed"


def test_grad_metrics_match_metrics(spatial_data, spatial_distances):
    for metric in dist.named_distances_with_gradients:
        if metric in spatial_distances:
            spatial_check(metric, spatial_data, spatial_distances, with_grad=True)

    # Handle the few special distances separately
    # SEuclidean
    v = np.abs(np.random.randn(spatial_data.shape[1]))
    dist_matrix = pairwise_distances(spatial_data, metric="seuclidean", V=v)
    test_matrix = np.array(
        [
            [
                dist.standardised_euclidean_grad(spatial_data[i], spatial_data[j], v)[0]
                for j in range(spatial_data.shape[0])
            ]
            for i in range(spatial_data.shape[0])
        ]
    )
    assert_array_almost_equal(
        test_matrix,
        dist_matrix,
        err_msg="Distances don't match " "for metric seuclidean",
    )

    if scipy_full_version >= (1, 8):
        # Weighted minkowski
        dist_matrix = pairwise_distances(spatial_data, metric="minkowski", w=v, p=3)
        test_matrix = np.array(
            [
                [
                    dist.weighted_minkowski_grad(spatial_data[i], spatial_data[j], v, p=3)[
                        0
                    ]
                    for j in range(spatial_data.shape[0])
                ]
                for i in range(spatial_data.shape[0])
            ]
        )
        assert_array_almost_equal(
            test_matrix,
            dist_matrix,
            err_msg="Distances don't match " "for metric weighted_minkowski",
        )

    # Mahalanobis
    v = np.abs(np.random.randn(spatial_data.shape[1], spatial_data.shape[1]))
    dist_matrix = pairwise_distances(spatial_data, metric="mahalanobis", VI=v)
    test_matrix = np.array(
        [
            [
                dist.mahalanobis_grad(spatial_data[i], spatial_data[j], v)[0]
                for j in range(spatial_data.shape[0])
            ]
            for i in range(spatial_data.shape[0])
        ]
    )
    assert_array_almost_equal(
        test_matrix,
        dist_matrix,
        decimal=5,
        err_msg="Distances don't match " "for metric mahalanobis",
    )

    # Hellinger
    dist_matrix = dist.pairwise_special_metric(
        np.abs(spatial_data[:-2]), np.abs(spatial_data[:-2])
    )
    test_matrix = np.array(
        [
            [
                dist.hellinger_grad(np.abs(spatial_data[i]), np.abs(spatial_data[j]))[0]
                for j in range(spatial_data.shape[0] - 2)
            ]
            for i in range(spatial_data.shape[0] - 2)
        ]
    )
    assert_array_almost_equal(
        test_matrix,
        dist_matrix,
        err_msg="Distances don't match " "for metric hellinger",
    )
