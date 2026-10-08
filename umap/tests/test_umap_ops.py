# ===================================================
#  UMAP Fit and Transform Operations Test cases
#  (not really fitting anywhere else)
# ===================================================

from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, pairwise_distances
from sklearn.preprocessing import normalize
from numpy.testing import assert_array_equal
from umap import UMAP
from umap.umap_ import init_update
from umap.spectral import component_layout
import numpy as np
import scipy.sparse
import pytest
import warnings
from umap.distances import pairwise_special_metric
from umap.utils import disconnected_vertices, fast_knn_indices
from scipy.sparse import csr_matrix

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
def test_fast_knn_indices_matches_argsort():
    rng = np.random.RandomState(0)
    D = rng.rand(200, 200).astype(np.float32)
    D = (D + D.T) / 2
    np.fill_diagonal(D, 0.0)
    k = 15
    expected = np.argsort(D, axis=1, kind="mergesort")[:, :k]
    result = fast_knn_indices(D, k)
    np.testing.assert_array_equal(result, expected)


def test_fast_knn_indices_matches_argsort_with_ties():
    # Duplicate points produce equal distances at the k-th neighbor; which of the
    # tied points is selected changes the kNN graph and therefore the embedding,
    # so ties must be broken by ascending index exactly like a stable argsort.
    rng = np.random.RandomState(1)
    D = rng.randint(0, 5, size=(150, 150)).astype(np.float32)
    D = np.minimum(D, D.T)
    np.fill_diagonal(D, 0.0)
    k = 15
    expected = np.argsort(D, axis=1, kind="mergesort")[:, :k]
    result = fast_knn_indices(D, k)
    np.testing.assert_array_equal(result, expected)


def test_blobs_cluster():
    data, labels = make_blobs(n_samples=500, n_features=10, centers=5)
    embedding = UMAP(n_epochs=100).fit_transform(data)
    assert adjusted_rand_score(labels, KMeans(5).fit_predict(embedding)) == 1.0


# Multi-components Layout
def test_multi_component_layout():
    data, labels = make_blobs(
        100, 2, centers=5, cluster_std=0.5, center_box=(-20, 20), random_state=42
    )

    true_centroids = np.empty((labels.max() + 1, data.shape[1]), dtype=np.float64)

    for label in range(labels.max() + 1):
        true_centroids[label] = data[labels == label].mean(axis=0)

    true_centroids = normalize(true_centroids, norm="l2")

    embedding = UMAP(n_neighbors=4, n_epochs=100).fit_transform(data)
    embed_centroids = np.empty((labels.max() + 1, data.shape[1]), dtype=np.float64)
    embed_labels = KMeans(n_clusters=5).fit_predict(embedding)

    for label in range(embed_labels.max() + 1):
        embed_centroids[label] = data[embed_labels == label].mean(axis=0)

    embed_centroids = normalize(embed_centroids, norm="l2")

    error = np.sum((true_centroids - embed_centroids) ** 2)

    assert error < 15.0, "Multi component embedding to far astray"


# Multi-components Layout
def test_multi_component_layout_precomputed():
    data, labels = make_blobs(
        100, 2, centers=5, cluster_std=0.5, center_box=(-20, 20), random_state=42
    )
    dmat = pairwise_distances(data)

    true_centroids = np.empty((labels.max() + 1, data.shape[1]), dtype=np.float64)

    for label in range(labels.max() + 1):
        true_centroids[label] = data[labels == label].mean(axis=0)

    true_centroids = normalize(true_centroids, norm="l2")

    embedding = UMAP(n_neighbors=4, metric="precomputed", n_epochs=100).fit_transform(
        dmat
    )
    embed_centroids = np.empty((labels.max() + 1, data.shape[1]), dtype=np.float64)
    embed_labels = KMeans(n_clusters=5).fit_predict(embedding)

    for label in range(embed_labels.max() + 1):
        embed_centroids[label] = data[embed_labels == label].mean(axis=0)

    embed_centroids = normalize(embed_centroids, norm="l2")

    error = np.sum((true_centroids - embed_centroids) ** 2)

    assert error < 15.0, "Multi component embedding to far astray"


@pytest.mark.parametrize("num_isolates", [1, 5])
@pytest.mark.parametrize("metric", ["jaccard", "hellinger"])
@pytest.mark.parametrize("force_approximation", [True, False])
def test_disconnected_data(num_isolates, metric, force_approximation):
    options = [False, True]
    disconnected_data = np.random.choice(a=options, size=(10, 30), p=[0.6, 1 - 0.6])
    # Add some disconnected data for the corner case test
    disconnected_data = np.vstack(
        [disconnected_data, np.zeros((num_isolates, 30), dtype="bool")]
    )
    new_columns = np.zeros((num_isolates + 10, num_isolates), dtype="bool")
    for i in range(num_isolates):
        new_columns[10 + i, i] = True
    disconnected_data = np.hstack([disconnected_data, new_columns])

    with warnings.catch_warnings(record=True) as w:
        model = UMAP(
            n_neighbors=3,
            metric=metric,
            force_approximation_algorithm=force_approximation,
        ).fit(disconnected_data)
        assert len(w) >= 1  # at least one warning should be raised here
        # we can't guarantee the order that the warnings will be raised in so check them all.
        flag = 0
        if num_isolates == 1:
            warning_contains = "A few of your vertices"
        elif num_isolates > 1:
            warning_contains = "A large number of your vertices"
        for wn in w:
            flag += warning_contains in str(wn.message)

        isolated_vertices = disconnected_vertices(model)
        assert flag == 1, str(([wn.message for wn in w], isolated_vertices))
        # Check that the first isolate has no edges in our umap.graph_
        assert isolated_vertices[10] == True
        number_of_nan = np.sum(np.isnan(model.embedding_[isolated_vertices]))
        assert number_of_nan >= num_isolates * model.n_components


def test_disconnection_distance_keeps_sigma_finite():
    """A point whose k-nn list mixes kept and pruned neighbours must keep a finite
    sigma, so that its remaining edges are weighted by the usual kernel rather
    than all set to 1.0 (the pruned neighbours used to make the row mean, and
    hence the sigma floor, infinite)."""
    rng = np.random.RandomState(0)
    small = rng.normal(0, 0.3, (8, 5)).astype(np.float32)
    big = (rng.normal(0, 1.0, (300, 5)) + 10.0).astype(np.float32)
    data = np.vstack([small, big])
    for force_approximation in (False, True):
        model = UMAP(
            n_neighbors=15,
            disconnection_distance=5.0,
            random_state=42,
            force_approximation_algorithm=force_approximation,
        ).fit(data)
        assert np.all(np.isfinite(model._sigmas))
        rows = model.graph_.tocsr()[:8].toarray()
        # each small-cluster point keeps its 7 in-cluster edges; the nearest
        # neighbour (and reciprocal nearest neighbours) have strength 1, the
        # others are strictly weaker instead of all being 1.0
        assert np.all((rows > 0.0).sum(axis=1) == 7)
        assert np.all(rows[rows > 0.0] <= 1.0)
        assert np.all((rows == 1.0).sum(axis=1) < (rows > 0.0).sum(axis=1))
        assert np.all([row[row > 0.0].min() < 0.9 for row in rows])


@pytest.mark.parametrize("num_isolates", [1])
@pytest.mark.parametrize("sparse", [True, False])
def test_disconnected_data_precomputed(num_isolates, sparse):
    disconnected_data = np.random.choice(
        a=[False, True], size=(10, 20), p=[0.66, 1 - 0.66]
    )
    # Add some disconnected data for the corner case test
    disconnected_data = np.vstack(
        [disconnected_data, np.zeros((num_isolates, 20), dtype="bool")]
    )
    new_columns = np.zeros((num_isolates + 10, num_isolates), dtype="bool")
    for i in range(num_isolates):
        new_columns[10 + i, i] = True
    disconnected_data = np.hstack([disconnected_data, new_columns])
    dmat = pairwise_special_metric(disconnected_data)
    if sparse:
        dmat = csr_matrix(dmat)
    model = UMAP(n_neighbors=3, metric="precomputed", disconnection_distance=1).fit(
        dmat
    )

    # Check that the first isolate has no edges in our umap.graph_
    isolated_vertices = disconnected_vertices(model)
    assert isolated_vertices[10] == True
    number_of_nan = np.sum(np.isnan(model.embedding_[isolated_vertices]))
    assert number_of_nan >= num_isolates * model.n_components


# ---------------
# Umap Transform
# --------------


def test_bad_transform_data(nn_data):
    u = UMAP().fit([[1, 1, 1, 1]])
    with pytest.raises(ValueError):
        u.transform([[0, 0, 0, 0]])


# Transform Stability
# -------------------
def test_umap_transform_embedding_stability(iris, iris_subset_model, iris_selection):
    """Test that transforming data does not alter the learned embeddings

    Issue #217 describes how using transform to embed new data using a
    trained UMAP transformer causes the fitting embedding matrix to change
    in cases when the new data has the same number of rows as the original
    training data.
    """

    data = iris.data[iris_selection]
    fitter = iris_subset_model
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
    a = np.random.random((100, 10))
    b = np.random.random((100, 5))

    umap = UMAP(n_epochs=100)
    u1 = umap.fit_transform(a[:, :5])
    u1_orig = u1.copy()
    assert_array_equal(u1_orig, umap.embedding_)

    _ = umap.transform(b)
    assert_array_equal(u1_orig, umap.embedding_)


# -----------
# UMAP Update
# -----------
def test_umap_update(iris, iris_subset_model, iris_selection, iris_model):

    new_data = iris.data[~iris_selection]
    new_model = iris_subset_model
    new_model.update(new_data)

    comparison_graph = scipy.sparse.vstack(
        [iris_model.graph_[iris_selection], iris_model.graph_[~iris_selection]]
    )
    comparison_graph = scipy.sparse.hstack(
        [comparison_graph[:, iris_selection], comparison_graph[:, ~iris_selection]]
    )

    error = np.sum(np.abs((new_model.graph_ - comparison_graph).data))

    assert error < 2.10


def test_umap_update_large(
    iris, iris_subset_model_large, iris_selection, iris_model_large
):

    new_data = iris.data[~iris_selection]
    new_model = iris_subset_model_large
    new_model.update(new_data)

    comparison_graph = scipy.sparse.vstack(
        [
            iris_model_large.graph_[iris_selection],
            iris_model_large.graph_[~iris_selection],
        ]
    )
    comparison_graph = scipy.sparse.hstack(
        [comparison_graph[:, iris_selection], comparison_graph[:, ~iris_selection]]
    )

    error = np.sum(np.abs((new_model.graph_ - comparison_graph).data))

    assert error < 3.0  # Higher error tolerance based on approx nearest neighbors


def test_init_update_averages_original_neighbors():
    init = np.zeros((4, 2), dtype=np.float32)
    init[0] = [0.0, 0.0]
    init[1] = [2.0, 4.0]
    # point 2 has two original neighbours, point 3 has none
    indices = np.array([[0, 1], [1, 0], [0, 1], [2, 3]])
    init_update(init, 2, indices)
    np.testing.assert_allclose(init[2], [1.0, 2.0])
    # no original neighbours: fall back to the centre of the original embedding
    np.testing.assert_allclose(init[3], [1.0, 2.0])


def test_umap_update_new_points_without_original_neighbors(iris):
    # https://github.com/lmcinnes/umap/issues/1075
    # some of the new (virginica) points only have new points as neighbours
    model = UMAP(n_epochs=20, random_state=0).fit(iris.data[:100])
    model.update(iris.data[100:])
    assert model.embedding_.shape == (150, 2)
    assert np.all(np.isfinite(model.embedding_))


# -----------------
# UMAP Graph output
# -----------------
def test_umap_graph_layout():
    data, labels = make_blobs(n_samples=500, n_features=10, centers=5)
    model = UMAP(n_epochs=100, transform_mode="graph")
    graph = model.fit_transform(data)
    assert scipy.sparse.issparse(graph)
    nc, cl = scipy.sparse.csgraph.connected_components(graph)
    assert nc == 5

    new_graph = model.transform(data[:10] + np.random.normal(0.0, 0.1, size=(10, 10)))
    assert scipy.sparse.issparse(graph)
    assert new_graph.shape[0] == 10


# ------------------------
# Component layout options
# ------------------------


def test_component_layout_options(nn_data):
    dmat = pairwise_distances(nn_data[:1000])
    n_components = 5
    component_labels = np.repeat(np.arange(5), dmat.shape[0] // 5)
    single = component_layout(
        dmat,
        n_components,
        component_labels,
        2,
        None,
        metric="precomputed",
        metric_kwds={"linkage": "single"},
    )
    average = component_layout(
        dmat,
        n_components,
        component_labels,
        2,
        None,
        metric="precomputed",
        metric_kwds={"linkage": "average"},
    )
    complete = component_layout(
        dmat,
        n_components,
        component_labels,
        2,
        None,
        metric="precomputed",
        metric_kwds={"linkage": "complete"},
    )

    assert single.shape[0] == 5
    assert average.shape[0] == 5
    assert complete.shape[0] == 5

    assert not np.all(single == average)
    assert not np.all(single == complete)
    assert not np.all(average == complete)
