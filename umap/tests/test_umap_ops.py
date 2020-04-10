# ===================================================
#  UMAP Fit and Transform Operations Test cases
#  (not really fitting anywhere else)
# ===================================================

from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
from sklearn.preprocessing import normalize
from nose.tools import assert_equal, assert_less, assert_raises
from numpy.testing import assert_array_equal
from umap import UMAP
import numpy as np

# Transform isn't stable under batching; hard to opt out of this.
# @SkipTest
# def test_scikit_learn_compatibility():
#     check_estimator(UMAP)


# This test is currently to expensive to run when turning
# off numba JITting to detect coverage.
# @SkipTest
# def test_umap_regression_supervision(): # pragma: no cover
#     boston = load_boston()
#     data = boston.data
#     embedding = UMAP(n_neighbors=10,
#                      min_dist=0.01,
#                      target_metric='euclidean',
#                      random_state=42).fit_transform(data, boston.target)
#


# Umap Clusterability
def test_blobs_cluster():
    data, labels = make_blobs(n_samples=500, n_features=10, centers=5)
    embedding = UMAP().fit_transform(data)
    assert_equal(adjusted_rand_score(labels, KMeans(5).fit_predict(embedding)), 1.0)


# Multi-components Layout
def test_multi_component_layout():
    data, labels = make_blobs(
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


# ---------------
# Umap Transform
# --------------


def test_bad_transform_data(nn_data):
    u = UMAP().fit([[1, 1, 1, 1]])
    assert_raises(ValueError, u.transform, [[0, 0, 0, 0]])


# Transform Stability
# -------------------
def test_umap_transform_embedding_stability(iris, iris_selection):
    """Test that transforming data does not alter the learned embeddings

    Issue #217 describes how using transform to embed new data using a
    trained UMAP transformer causes the fitting embedding matrix to change
    in cases when the new data has the same number of rows as the original
    training data.
    """

    data = iris.data[iris_selection]
    fitter = UMAP(n_neighbors=10, min_dist=0.01, random_state=42).fit(data)
    original_embedding = fitter.embedding_.copy()

    # The important point is that the new data has the same number of rows
    # as the original fit data
    new_data = np.random.random(data.shape)
    _ = fitter.transform(new_data)

    assert_array_equal(
        original_embedding,
        fitter.embedding_,
        "Transforming new data changed the original embeddings",
    )

    # Example from issue #217
    a = np.random.random((1000, 10))
    b = np.random.random((1000, 5))

    umap = UMAP()
    u1 = umap.fit_transform(a[:, :5])
    u1_orig = u1.copy()
    assert_array_equal(u1_orig, umap.embedding_)

    _ = umap.transform(b)
    assert_array_equal(u1_orig, umap.embedding_)
