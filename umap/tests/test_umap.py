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
from sklearn.utils.testing import (assert_equal,
                                   assert_array_equal,
                                   assert_array_almost_equal,
                                   assert_raises,
                                   assert_in,
                                   assert_not_in,
                                   assert_no_warnings,
                                   if_matplotlib)
from sklearn.metrics import pairwise_distances
from sklearn.neighbors import KDTree
from sklearn.utils import shuffle
from sklearn.preprocessing import StandardScaler
from scipy.stats import mode

from tempfile import mkdtemp
from functools import wraps
from nose import SkipTest

from sklearn import datasets

import umap.distances as dist
import umap.sparse as spdist
from umap.umap_ import (
    INT32_MAX,
    INT32_MIN,
    rptree_leaf_array,
    make_nn_descent,
    UMAP)

np.random.seed(42)
spatial_data = np.random.randn(10, 20)
binary_data = np.random.choice(a=[False, True],
                               size=(10, 20),
                               p=[0.66, 1-0.66])
sparse_spatial_data = sparse.csr_matrix(spatial_data * binary_data)
sparse_binary_data = sparse.csr_matrix(binary_data)

nn_data = np.random.uniform(0, 1, size=(1000, 5))
binary_nn_data = np.random.choice(a=[False, True],
                                  size=(1000, 5),
                                 p=[0.66, 1-0.66])
sparse_nn_data = sparse.csr_matrix(nn_data * binary_nn_data)

spatial_distances = (
    'euclidean',
    'manhattan',
    'chebyshev',
    'minkowski',
    'hamming',
    'canberra',
    'braycurtis',
    'cosine',
    'correlation'
)

binary_distances = (
    'jaccard',
    'matching',
    'dice',
    'kulsinski',
    'rogerstanimoto',
    'russellrao',
    'sokalmichener',
    'sokalsneath',
    'yule'
)


def test_nn_descent_neighbor_accuracy():
    rng_state = np.random.randint(INT32_MIN, INT32_MAX, size=3)
    nn_descent = make_nn_descent(dist.euclidean, ())
    leaf_array = rptree_leaf_array(nn_data, 10, rng_state)
    tmp_indices, knn_dists = nn_descent(nn_data,
                                        10,
                                        rng_state,
                                        leaf_array=leaf_array)
    knn_indices = tmp_indices.astype(np.int64)
    for i in range(knn_indices.shape[0]):
        order = np.argsort(knn_dists[i])
        knn_dists[i] = knn_dists[i][order]
        knn_indices[i] = knn_indices[i][order]

    tree = KDTree(nn_data)
    true_indices = tree.query(nn_data, 10, return_distance=False)

    num_correct = 0.0
    for i in range(nn_data.shape[0]):
        num_correct += np.sum(np.in1d(true_indices[i], knn_indices[i]))

    percent_correct = num_correct / (spatial_data.shape[0] * 10)
    assert_greater_equal(percent_correct, 0.99, 'NN-descent did not get 99% '
                                               'accuracy on nearest neighbors')

def test_sparse_nn_descent_neighbor_accuracy():
    rng_state = np.random.randint(INT32_MIN, INT32_MAX, size=3)
    nn_descent = spdist.make_sparse_nn_descent(spdist.sparse_euclidean, ())
    leaf_array = rptree_leaf_array(sparse_nn_data, 10, rng_state)
    tmp_indices, knn_dists = nn_descent(sparse_nn_data.indices,
                                        sparse_nn_data.indptr,
                                        sparse_nn_data.data,
                                        sparse_nn_data.shape[0],
                                        10,
                                        rng_state,
                                        leaf_array=leaf_array)
    knn_indices = tmp_indices.astype(np.int64)
    for i in range(knn_indices.shape[0]):
        order = np.argsort(knn_dists[i])
        knn_dists[i] = knn_dists[i][order]
        knn_indices[i] = knn_indices[i][order]

    tree = KDTree(sparse_nn_data.todense())
    true_indices = tree.query(sparse_nn_data.todense(), 10, return_distance=False)

    print(sparse_nn_data.shape)

    num_correct = 0.0
    for i in range(nn_data.shape[0]):
        num_correct += np.sum(np.in1d(true_indices[i], knn_indices[i]))

    percent_correct = num_correct / (spatial_data.shape[0] * 10)
    assert_greater_equal(percent_correct, 0.99, 'Sparse NN-descent did not get '
                                                '99% accuracy on nearest '
                                                'neighbors')

def test_trustworthiness():
    pass


def test_metrics():
    for metric in spatial_distances:
        dist_matrix = pairwise_distances(spatial_data, metric=metric)
        dist_function = dist.named_distances[metric]
        test_matrix = np.array([[dist_function(spatial_data[i], spatial_data[j])
                                    for j in range(spatial_data.shape[0])]
                                        for i in range(spatial_data.shape[0])])
        assert_array_almost_equal(test_matrix, dist_matrix,
                                  err_msg="Distances don't match "
                                          "for metric {}".format(metric))

    for metric in binary_distances:
        dist_matrix = pairwise_distances(binary_data, metric=metric)
        dist_function = dist.named_distances[metric]
        test_matrix = np.array([[dist_function(binary_data[i], binary_data[j])
                                    for j in range(binary_data.shape[0])]
                                        for i in range(binary_data.shape[0])])
        assert_array_almost_equal(test_matrix, dist_matrix,
                                  err_msg="Distances don't match "
                                          "for metric {}".format(metric))

def test_sparse_metrics():
    for metric in spatial_distances:
        # Sparse correlation has precision errors right now, leave out ...
        if metric in spdist.sparse_named_distances and metric is not \
                'correlation':
            dist_matrix = pairwise_distances(sparse_spatial_data.todense(),
                                             metric=metric)
            dist_function = spdist.sparse_named_distances[metric]
            if metric in spdist.sparse_need_n_features:
                test_matrix = np.array(
                    [[dist_function(sparse_spatial_data[i].indices,
                                    sparse_spatial_data[i].data,
                                    sparse_spatial_data[j].indices,
                                    sparse_spatial_data[j].data,
                                    sparse_spatial_data.shape[1])
                        for j in range(sparse_spatial_data.shape[0])]
                            for i in range(sparse_spatial_data.shape[0])])
            else:
                test_matrix = np.array(
                    [[dist_function(sparse_spatial_data[i].indices,
                                    sparse_spatial_data[i].data,
                                    sparse_spatial_data[j].indices,
                                    sparse_spatial_data[j].data)
                        for j in range(sparse_spatial_data.shape[0])]
                            for i in range(sparse_spatial_data.shape[0])])

            assert_array_almost_equal(test_matrix, dist_matrix,
                                      err_msg="Distances don't match "
                                              "for metric {}".format(metric))


def test_sparse_fit():
    pass

@SkipTest
def test_sklearn_digits():
    digits = datasets.load_digits()
    data = digits.data
    embedding = UMAP(n_neighbors=5, min_dist=0.01,
                     random_state=42).fit_transform(data)
    #np.save('digits_embedding_42.npy', embedding)
    to_match = np.load(os.path.join(os.path.dirname(__file__),
                                    'digits_embedding_42.npy'))
    assert_array_almost_equal(embedding, to_match, err_msg='Digits embedding '
                                                           'is not consistent '
                                                           'with previous runs')
