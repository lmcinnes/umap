"""
Tests for UMAP to ensure things are working as expected.
"""
from nose.tools import assert_less
from nose.tools import assert_greater_equal
import os.path
import numpy as np
from scipy.spatial import distance
from scipy import sparse
from scipy import stats
from sklearn.utils.estimator_checks import check_estimator
from sklearn.utils.testing import (
    assert_equal,
    assert_array_equal,
    assert_almost_equal,
    assert_array_almost_equal,
    assert_raises,
    assert_in,
    assert_not_in,
    assert_no_warnings,
    if_matplotlib,
)
from sklearn.metrics import pairwise_distances, adjusted_rand_score
from sklearn.neighbors import KDTree, BallTree
from sklearn.utils import shuffle
from sklearn.preprocessing import StandardScaler, normalize
from sklearn.manifold.t_sne import trustworthiness
from sklearn.cluster import KMeans
from scipy.stats import mode

from tempfile import mkdtemp
from functools import wraps
from nose import SkipTest

from sklearn import datasets

import umap.distances as dist
import umap.sparse as spdist
import umap.validation as valid
from umap.nndescent import (
    make_initialisations,
    make_initialized_nnd_search,
    initialise_search,
)
from umap.utils import deheap_sort
from umap.umap_ import (
    INT32_MAX,
    INT32_MIN,
    make_forest,
    rptree_leaf_array,
    nearest_neighbors,
    smooth_knn_dist,
    fuzzy_simplicial_set,
    UMAP,
)

np.random.seed(42)
spatial_data = np.random.randn(10, 20)
spatial_data = np.vstack(
    [spatial_data, np.zeros((2, 20))]
)  # Add some all zero data for corner case test
binary_data = np.random.choice(a=[False, True], size=(10, 20), p=[0.66, 1 - 0.66])
binary_data = np.vstack(
    [binary_data, np.zeros((2, 20))]
)  # Add some all zero data for corner case test
sparse_spatial_data = sparse.csr_matrix(spatial_data * binary_data)
sparse_binary_data = sparse.csr_matrix(binary_data)

nn_data = np.random.uniform(0, 1, size=(1000, 5))
nn_data = np.vstack(
    [nn_data, np.zeros((2, 5))]
)  # Add some all zero data for corner case test
binary_nn_data = np.random.choice(a=[False, True], size=(1000, 5), p=[0.66, 1 - 0.66])
binary_nn_data = np.vstack(
    [binary_nn_data, np.zeros((2, 5))]
)  # Add some all zero data for corner case test
sparse_nn_data = sparse.csr_matrix(nn_data * binary_nn_data)

iris = datasets.load_iris()
iris_selection = np.random.choice([True, False], 150, replace=True, p=[0.75, 0.25])

spatial_distances = (
    "euclidean",
    "manhattan",
    "chebyshev",
    "minkowski",
    "hamming",
    "canberra",
    "braycurtis",
    "cosine",
    "correlation",
)

binary_distances = (
    "jaccard",
    "matching",
    "dice",
    "kulsinski",
    "rogerstanimoto",
    "russellrao",
    "sokalmichener",
    "sokalsneath",
    "yule",
)


# Transform isn't stable under batching; hard to opt out of this.
@SkipTest
def test_scikit_learn_compatibility():
    check_estimator(UMAP)


def test_nn_descent_neighbor_accuracy():
    knn_indices, knn_dists, _ = nearest_neighbors(
        nn_data, 10, "euclidean", {}, False, np.random
    )

    tree = KDTree(nn_data)
    true_indices = tree.query(nn_data, 10, return_distance=False)

    num_correct = 0.0
    for i in range(nn_data.shape[0]):
        num_correct += np.sum(np.in1d(true_indices[i], knn_indices[i]))

    percent_correct = num_correct / (spatial_data.shape[0] * 10)
    assert_greater_equal(
        percent_correct,
        0.99,
        "NN-descent did not get 99% " "accuracy on nearest neighbors",
    )


def test_angular_nn_descent_neighbor_accuracy():
    knn_indices, knn_dists, _ = nearest_neighbors(
        nn_data, 10, "cosine", {}, True, np.random
    )

    angular_data = normalize(nn_data, norm="l2")
    tree = KDTree(angular_data)
    true_indices = tree.query(angular_data, 10, return_distance=False)

    num_correct = 0.0
    for i in range(nn_data.shape[0]):
        num_correct += np.sum(np.in1d(true_indices[i], knn_indices[i]))

    percent_correct = num_correct / (spatial_data.shape[0] * 10)
    assert_greater_equal(
        percent_correct,
        0.99,
        "NN-descent did not get 99% " "accuracy on nearest neighbors",
    )


def test_sparse_nn_descent_neighbor_accuracy():
    knn_indices, knn_dists, _ = nearest_neighbors(
        sparse_nn_data, 10, "euclidean", {}, False, np.random
    )

    tree = KDTree(sparse_nn_data.todense())
    true_indices = tree.query(sparse_nn_data.todense(), 10, return_distance=False)

    num_correct = 0.0
    for i in range(nn_data.shape[0]):
        num_correct += np.sum(np.in1d(true_indices[i], knn_indices[i]))

    percent_correct = num_correct / (spatial_data.shape[0] * 10)
    assert_greater_equal(
        percent_correct,
        0.99,
        "Sparse NN-descent did not get " "99% accuracy on nearest " "neighbors",
    )


def test_sparse_angular_nn_descent_neighbor_accuracy():
    knn_indices, knn_dists, _ = nearest_neighbors(
        sparse_nn_data, 10, "cosine", {}, True, np.random
    )

    angular_data = normalize(sparse_nn_data, norm="l2").toarray()
    tree = KDTree(angular_data)
    true_indices = tree.query(angular_data, 10, return_distance=False)

    num_correct = 0.0
    for i in range(nn_data.shape[0]):
        num_correct += np.sum(np.in1d(true_indices[i], knn_indices[i]))

    percent_correct = num_correct / (spatial_data.shape[0] * 10)
    assert_greater_equal(
        percent_correct,
        0.99,
        "NN-descent did not get 99% " "accuracy on nearest neighbors",
    )


def test_smooth_knn_dist_l1norms():
    knn_indices, knn_dists, _ = nearest_neighbors(
        nn_data, 10, "euclidean", {}, False, np.random
    )
    sigmas, rhos = smooth_knn_dist(knn_dists, 10)
    shifted_dists = knn_dists - rhos[:, np.newaxis]
    shifted_dists[shifted_dists < 0.0] = 0.0
    vals = np.exp(-(shifted_dists / sigmas[:, np.newaxis]))
    norms = np.sum(vals, axis=1)

    assert_array_almost_equal(
        norms,
        1.0 + np.log2(10) * np.ones(norms.shape[0]),
        decimal=3,
        err_msg="Smooth knn-dists does not give expected" "norms",
    )


def test_nn_descent_neighbor_accuracy_callable_metric():
    knn_indices, knn_dists, _ = nearest_neighbors(
        nn_data, 10, dist.euclidean, {}, False, np.random
    )

    tree = KDTree(nn_data)
    true_indices = tree.query(nn_data, 10, return_distance=False)

    num_correct = 0.0
    for i in range(nn_data.shape[0]):
        num_correct += np.sum(np.in1d(true_indices[i], knn_indices[i]))

    percent_correct = num_correct / (spatial_data.shape[0] * 10)
    assert_greater_equal(
        percent_correct,
        0.99,
        "NN-descent did not get 99% "
        "accuracy on nearest neighbors with callable metric",
    )


def test_smooth_knn_dist_l1norms_w_connectivity():
    knn_indices, knn_dists, _ = nearest_neighbors(
        nn_data, 10, "euclidean", {}, False, np.random
    )
    sigmas, rhos = smooth_knn_dist(knn_dists, 10, local_connectivity=1.75)
    shifted_dists = knn_dists - rhos[:, np.newaxis]
    shifted_dists[shifted_dists < 0.0] = 0.0
    vals = np.exp(-(shifted_dists / sigmas[:, np.newaxis]))
    norms = np.sum(vals, axis=1)

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


def test_nn_search():
    train = nn_data[100:]
    test = nn_data[:100]
    (knn_indices, knn_dists, rp_forest) = nearest_neighbors(
        train, 10, "euclidean", {}, False, np.random
    )

    graph = fuzzy_simplicial_set(
        nn_data,
        10,
        np.random,
        "euclidean",
        {},
        knn_indices,
        knn_dists,
        False,
        1.0,
        1.0,
        False,
    )

    search_graph = sparse.lil_matrix((train.shape[0], train.shape[0]), dtype=np.int8)
    search_graph.rows = knn_indices
    search_graph.data = (knn_dists != 0).astype(np.int8)
    search_graph = search_graph.maximum(search_graph.transpose()).tocsr()

    random_init, tree_init = make_initialisations(dist.euclidean, ())
    search = make_initialized_nnd_search(dist.euclidean, ())

    rng_state = np.random.randint(INT32_MIN, INT32_MAX, 3).astype(np.int64)
    init = initialise_search(
        rp_forest, train, test, int(10 * 3), random_init, tree_init, rng_state
    )
    result = search(train, search_graph.indptr, search_graph.indices, init, test)

    indices, dists = deheap_sort(result)
    indices = indices[:, :10]

    tree = KDTree(train)
    true_indices = tree.query(test, 10, return_distance=False)

    num_correct = 0.0
    for i in range(test.shape[0]):
        num_correct += np.sum(np.in1d(true_indices[i], indices[i]))

    percent_correct = num_correct / (test.shape[0] * 10)
    assert_greater_equal(
        percent_correct,
        0.99,
        "Sparse NN-descent did not get " "99% accuracy on nearest " "neighbors",
    )


def test_metrics():
    for metric in spatial_distances:
        dist_matrix = pairwise_distances(spatial_data, metric=metric)
        # scipy is bad sometimes
        if metric == "braycurtis":
            dist_matrix[np.where(~np.isfinite(dist_matrix))] = 0.0
        if metric in ("cosine", "correlation"):
            dist_matrix[np.where(~np.isfinite(dist_matrix))] = 1.0
            # And because distance between all zero vectors should be zero
            dist_matrix[10, 11] = 0.0
            dist_matrix[11, 10] = 0.0
        dist_function = dist.named_distances[metric]
        test_matrix = np.array(
            [
                [
                    dist_function(spatial_data[i], spatial_data[j])
                    for j in range(spatial_data.shape[0])
                ]
                for i in range(spatial_data.shape[0])
            ]
        )
        assert_array_almost_equal(
            test_matrix,
            dist_matrix,
            err_msg="Distances don't match " "for metric {}".format(metric),
        )

    for metric in binary_distances:
        dist_matrix = pairwise_distances(binary_data, metric=metric)
        if metric in ("jaccard", "dice", "sokalsneath", "yule"):
            dist_matrix[np.where(~np.isfinite(dist_matrix))] = 0.0
        if metric in ("kulsinski", "russellrao"):
            dist_matrix[np.where(~np.isfinite(dist_matrix))] = 0.0
            # And because distance between all zero vectors should be zero
            dist_matrix[10, 11] = 0.0
            dist_matrix[11, 10] = 0.0
        dist_function = dist.named_distances[metric]
        test_matrix = np.array(
            [
                [
                    dist_function(binary_data[i], binary_data[j])
                    for j in range(binary_data.shape[0])
                ]
                for i in range(binary_data.shape[0])
            ]
        )
        assert_array_almost_equal(
            test_matrix,
            dist_matrix,
            err_msg="Distances don't match " "for metric {}".format(metric),
        )

    # Handle the few special distances separately
    # SEuclidean
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

    # Weighted minkowski
    dist_matrix = pairwise_distances(spatial_data, metric="wminkowski", w=v, p=3)
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
    # Mahalanobis
    v = np.abs(np.random.randn(spatial_data.shape[1], spatial_data.shape[1]))
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
    # Haversine
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


def test_sparse_metrics():
    for metric in spatial_distances:
        if metric in spdist.sparse_named_distances:
            dist_matrix = pairwise_distances(
                sparse_spatial_data.todense(), metric=metric
            )
            if metric in ("braycurtis", "dice", "sokalsneath", "yule"):
                dist_matrix[np.where(~np.isfinite(dist_matrix))] = 0.0
            if metric in ("cosine", "correlation", "kulsinski", "russellrao"):
                dist_matrix[np.where(~np.isfinite(dist_matrix))] = 1.0
                # And because distance between all zero vectors should be zero
                dist_matrix[10, 11] = 0.0
                dist_matrix[11, 10] = 0.0

            dist_function = spdist.sparse_named_distances[metric]
            if metric in spdist.sparse_need_n_features:
                test_matrix = np.array(
                    [
                        [
                            dist_function(
                                sparse_spatial_data[i].indices,
                                sparse_spatial_data[i].data,
                                sparse_spatial_data[j].indices,
                                sparse_spatial_data[j].data,
                                sparse_spatial_data.shape[1],
                            )
                            for j in range(sparse_spatial_data.shape[0])
                        ]
                        for i in range(sparse_spatial_data.shape[0])
                    ]
                )
            else:
                test_matrix = np.array(
                    [
                        [
                            dist_function(
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

            assert_array_almost_equal(
                test_matrix,
                dist_matrix,
                err_msg="Sparse distances don't match " "for metric {}".format(metric),
            )

    for metric in binary_distances:
        if metric in spdist.sparse_named_distances:
            dist_matrix = pairwise_distances(
                sparse_binary_data.todense(), metric=metric
            )
            if metric in ("jaccard", "dice", "sokalsneath", "yule"):
                dist_matrix[np.where(~np.isfinite(dist_matrix))] = 0.0
            if metric in ("kulsinski", "russellrao"):
                dist_matrix[np.where(~np.isfinite(dist_matrix))] = 1.0
                # And because distance between all zero vectors should be zero
                dist_matrix[10, 11] = 0.0
                dist_matrix[11, 10] = 0.0

            dist_function = spdist.sparse_named_distances[metric]
            if metric in spdist.sparse_need_n_features:
                test_matrix = np.array(
                    [
                        [
                            dist_function(
                                sparse_binary_data[i].indices,
                                sparse_binary_data[i].data,
                                sparse_binary_data[j].indices,
                                sparse_binary_data[j].data,
                                sparse_binary_data.shape[1],
                            )
                            for j in range(sparse_binary_data.shape[0])
                        ]
                        for i in range(sparse_binary_data.shape[0])
                    ]
                )
            else:
                test_matrix = np.array(
                    [
                        [
                            dist_function(
                                sparse_binary_data[i].indices,
                                sparse_binary_data[i].data,
                                sparse_binary_data[j].indices,
                                sparse_binary_data[j].data,
                            )
                            for j in range(sparse_binary_data.shape[0])
                        ]
                        for i in range(sparse_binary_data.shape[0])
                    ]
                )

            assert_array_almost_equal(
                test_matrix,
                dist_matrix,
                err_msg="Sparse distances don't match " "for metric {}".format(metric),
            )


def test_umap_sparse_trustworthiness():
    embedding = UMAP(n_neighbors=10).fit_transform(sparse_nn_data[:100])
    trust = trustworthiness(sparse_nn_data[:100].toarray(), embedding, 10)
    assert_greater_equal(
        trust,
        0.92,
        "Insufficiently trustworthy embedding for"
        "sparse test dataset: {}".format(trust),
    )


def test_umap_trustworthiness_on_iris():
    data = iris.data
    embedding = UMAP(n_neighbors=10, min_dist=0.01, random_state=42).fit_transform(data)
    trust = trustworthiness(iris.data, embedding, 10)
    assert_greater_equal(
        trust,
        0.97,
        "Insufficiently trustworthy embedding for" "iris dataset: {}".format(trust),
    )


def test_umap_trustworthiness_on_iris_random_init():
    data = iris.data
    embedding = UMAP(
        n_neighbors=10, min_dist=0.01, random_state=42, init="random"
    ).fit_transform(data)
    trust = trustworthiness(iris.data, embedding, 10)
    assert_greater_equal(
        trust,
        0.95,
        "Insufficiently trustworthy embedding for" "iris dataset: {}".format(trust),
    )


def test_supervised_umap_trustworthiness_on_iris():
    data = iris.data
    embedding = UMAP(n_neighbors=10, min_dist=0.01, random_state=42).fit_transform(
        data, iris.target
    )
    trust = trustworthiness(iris.data, embedding, 10)
    assert_greater_equal(
        trust,
        0.97,
        "Insufficiently trustworthy embedding for" "iris dataset: {}".format(trust),
    )


def test_semisupervised_umap_trustworthiness_on_iris():
    data = iris.data
    target = iris.target.copy()
    target[25:75] = -1
    embedding = UMAP(n_neighbors=10, min_dist=0.01, random_state=42).fit_transform(
        data, target
    )
    trust = trustworthiness(iris.data, embedding, 10)
    assert_greater_equal(
        trust,
        0.97,
        "Insufficiently trustworthy embedding for" "iris dataset: {}".format(trust),
    )


def test_initialized_umap_trustworthiness_on_iris():
    data = iris.data
    embedding = UMAP(
        n_neighbors=10, min_dist=0.01, init=data[:, 2:], random_state=42
    ).fit_transform(data, iris.target)
    trust = trustworthiness(iris.data, embedding, 10)
    assert_greater_equal(
        trust,
        0.97,
        "Insufficiently trustworthy embedding for" "iris dataset: {}".format(trust),
    )


def test_umap_transform_on_iris():
    data = iris.data[iris_selection]
    fitter = UMAP(n_neighbors=10, min_dist=0.01, random_state=42).fit(data)

    new_data = iris.data[~iris_selection]
    embedding = fitter.transform(new_data)

    trust = trustworthiness(new_data, embedding, 10)
    assert_greater_equal(
        trust,
        0.89,
        "Insufficiently trustworthy transform for" "iris dataset: {}".format(trust),
    )


# # This test is currently to expensive to run when turning
# # off numba JITting to detect coverage.
# @SkipTest
# def test_umap_regression_supervision(): # pragma: no cover
#     boston = datasets.load_boston()
#     data = boston.data
#     embedding = UMAP(n_neighbors=10,
#                      min_dist=0.01,
#                      target_metric='euclidean',
#                      random_state=42).fit_transform(data, boston.target)


def test_blobs_cluster():
    data, labels = datasets.make_blobs(n_samples=500, n_features=10, centers=5)
    embedding = UMAP().fit_transform(data)
    assert_equal(adjusted_rand_score(labels, KMeans(5).fit_predict(embedding)), 1.0)


def test_multi_component_layout():
    data, labels = datasets.make_blobs(
        100, 2, centers=5, cluster_std=0.5, center_box=[-20, 20], random_state=42
    )

    true_centroids = np.empty((labels.max() + 1, data.shape[1]), dtype=np.float64)

    for label in range(labels.max() + 1):
        true_centroids[label] = data[labels == label].mean(axis=0)

    true_centroids = normalize(true_centroids, norm="l2")

    embedding = UMAP(n_neighbors=4).fit_transform(data)
    embed_centroids = np.empty((labels.max() + 1, data.shape[1]), dtype=np.float64)
    embed_labels = KMeans(n_clusters=5).fit_predict(embedding)

    for label in range(embed_labels.max() + 1):
        embed_centroids[label] = data[embed_labels == label].mean(axis=0)

    embed_centroids = normalize(embed_centroids, norm="l2")

    error = np.sum((true_centroids - embed_centroids) ** 2)

    assert_less(error, 15.0, msg="Multi component embedding to far astray")


def test_umap_bad_parameters():
    u = UMAP(set_op_mix_ratio=-1.0)
    assert_raises(ValueError, u.fit, nn_data)
    u = UMAP(set_op_mix_ratio=1.5)
    assert_raises(ValueError, u.fit, nn_data)
    u = UMAP(min_dist=2.0)
    assert_raises(ValueError, u.fit, nn_data)
    u = UMAP(min_dist=-1)
    assert_raises(ValueError, u.fit, nn_data)
    u = UMAP(n_components=-1)
    assert_raises(ValueError, u.fit, nn_data)
    u = UMAP(n_components=1.5)
    assert_raises(ValueError, u.fit, nn_data)
    u = UMAP(n_neighbors=0.5)
    assert_raises(ValueError, u.fit, nn_data)
    u = UMAP(n_neighbors=-1)
    assert_raises(ValueError, u.fit, nn_data)
    u = UMAP(metric=45)
    assert_raises(ValueError, u.fit, nn_data)
    u = UMAP(learning_rate=-1.5)
    assert_raises(ValueError, u.fit, nn_data)
    u = UMAP(repulsion_strength=-0.5)
    assert_raises(ValueError, u.fit, nn_data)
    u = UMAP(negative_sample_rate=-1)
    assert_raises(ValueError, u.fit, nn_data)
    u = UMAP(init="foobar")
    assert_raises(ValueError, u.fit, nn_data)
    u = UMAP(init=42)
    assert_raises(ValueError, u.fit, nn_data)
    u = UMAP(init=np.array([[0, 0, 0], [0, 0, 0]]))
    assert_raises(ValueError, u.fit, nn_data)
    u = UMAP(n_epochs=-2)
    assert_raises(ValueError, u.fit, nn_data)
    u = UMAP(target_n_neighbors=1)
    assert_raises(ValueError, u.fit, nn_data)

    u = UMAP(a=1.2, b=1.75, n_neighbors=2000)
    u.fit(nn_data)
    assert_equal(u._a, 1.2)
    assert_equal(u._b, 1.75)
    # assert_raises(ValueError, u.fit, nn_data) we simply warn now

    assert_raises(ValueError, nearest_neighbors, nn_data, 10, 42, {}, False, np.random)
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
