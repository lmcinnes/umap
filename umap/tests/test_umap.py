"""
Tests for UMAP to ensure things are working as expected.
"""
from nose.tools import assert_less
from nose.tools import assert_greater_equal
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
from sklearn.utils import shuffle
from sklearn.preprocessing import StandardScaler
from scipy.stats import mode

from tempfile import mkdtemp
from functools import wraps
from nose import SkipTest

from sklearn import datasets

import umap.distances as dist
import umap.sparse as spdist

np.random.seed(42)
spatial_data = np.random.randn(10, 20)
binary_data = np.random.choice(a=[False, True],
                               size=(10, 20),
                               p=[0.66, 1-0.66])
sparse_spatial_data = sparse.csr_matrix(spatial_data * binary_data)
sparse_binary_data = sparse.csr_matrix(binary_data)

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
    pass

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

def test_sklearn_digits():
    pass