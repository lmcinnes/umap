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
import umap.umap_ as umap_module
from umap.spectral import component_layout
import numpy as np
import scipy.sparse
import pytest
import warnings
from umap.distances import pairwise_special_metric
from umap.utils import disconnected_vertices
from scipy.sparse import csr_matrix


@pytest.mark.parametrize(
    "data",
    [
        np.arange(120, dtype=np.float32).reshape(24, 5),
        csr_matrix(np.arange(120, dtype=np.float32).reshape(24, 5)),
        csr_matrix((24, 5), dtype=np.float32),
    ],
)
def test_input_data_hash_is_parallel_and_content_deterministic(data):
    single_threaded = umap_module._input_data_hash(data, n_jobs=1, chunk_size=17)
    parallel = umap_module._input_data_hash(data, n_jobs=4, chunk_size=17)
    copied = umap_module._input_data_hash(data.copy(), n_jobs=2, chunk_size=17)

    assert single_threaded == parallel == copied

    changed = data.copy()
    if scipy.sparse.issparse(changed) and changed.nnz == 0:
        changed = csr_matrix(
            ([1.0], ([0], [1])), shape=changed.shape, dtype=changed.dtype
        )
    else:
        changed[0, 1] += 1.0
    assert umap_module._input_data_hash(changed, chunk_size=17) != single_threaded

    if not scipy.sparse.issparse(data):
        reshaped = data.reshape(12, 10)
        converted = data.astype(np.float64)
        assert umap_module._input_data_hash(reshaped) != single_threaded
        assert umap_module._input_data_hash(converted) != single_threaded


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_make_epochs_per_sample_matches_numpy(dtype):
    weights = np.array([0.0, 1e-6, 0.125, 0.5, 1.0], dtype=dtype)
    n_epochs = 17
    expected = np.full(weights.shape[0], -1.0, dtype=np.float64)
    n_samples = n_epochs * (weights / weights.max())
    expected[n_samples > 0.0] = float(n_epochs) / np.float64(n_samples[n_samples > 0.0])

    result = umap_module.make_epochs_per_sample(weights, n_epochs)

    assert result.dtype == np.float64
    np.testing.assert_array_equal(result, expected)


def test_make_epochs_per_sample_handles_empty_and_zero_weights():
    empty = umap_module.make_epochs_per_sample(np.empty(0, dtype=np.float32), 17)
    zeros = umap_module.make_epochs_per_sample(np.zeros(4, dtype=np.float32), 17)

    assert empty.dtype == np.float64
    assert empty.shape == (0,)
    np.testing.assert_array_equal(zeros, -np.ones(4, dtype=np.float64))


@pytest.mark.parametrize("mix_ratio", [0.0, 0.3, 1.0])
def test_fuzzy_set_operation_matches_sparse_expression(mix_ratio):
    graph = csr_matrix(
        np.array(
            [
                [0.0, 0.25, 0.0, 0.5],
                [0.75, 0.0, 0.125, 0.0],
                [0.0, 0.0, 0.0, 1.0],
                [0.25, 0.0, 0.0, 0.0],
            ],
            dtype=np.float32,
        )
    )
    transpose = graph.transpose()
    product = graph.multiply(transpose)
    expected = mix_ratio * (graph + transpose - product) + (1.0 - mix_ratio) * product
    expected.eliminate_zeros()

    result = umap_module._fuzzy_set_operation(graph, mix_ratio)

    assert result.dtype == np.float32
    assert result.has_canonical_format
    np.testing.assert_allclose(result.toarray(), expected.toarray(), rtol=1e-7)


@pytest.mark.parametrize("mix_ratio", [0.0, 0.3, 1.0])
def test_fuzzy_simplicial_set_matches_sparse_expression(mix_ratio):
    knn_indices = np.array(
        [
            [0, 1, 1, -1],
            [1, 0, 2, 3],
            [2, 1, 3, 0],
            [3, 2, 0, 1],
        ],
        dtype=np.int32,
    )
    knn_dists = np.array(
        [
            [0.0, 0.5, 0.75, np.inf],
            [0.0, 0.25, 0.5, 1.0],
            [0.0, 0.25, 0.75, 1.0],
            [0.0, 0.5, 0.75, 1.0],
        ],
        dtype=np.float32,
    )
    data = np.zeros((4, 2), dtype=np.float32)
    sigmas, rhos = umap_module.smooth_knn_dist(
        knn_dists, float(knn_indices.shape[1]), local_connectivity=1.0
    )
    rows, cols, vals, dists = umap_module.compute_membership_strengths(
        knn_indices, knn_dists, sigmas, rhos, return_dists=True
    )
    directed_graph = scipy.sparse.coo_matrix((vals, (rows, cols)), shape=(4, 4))
    directed_graph.eliminate_zeros()
    transpose = directed_graph.transpose()
    product = directed_graph.multiply(transpose)
    expected = (
        mix_ratio * (directed_graph + transpose - product) + (1.0 - mix_ratio) * product
    )
    expected.eliminate_zeros()

    expected_dists = scipy.sparse.coo_matrix((dists, (rows, cols)), shape=(4, 4))
    expected_dists = expected_dists.maximum(expected_dists.transpose()).todok()

    result, _, _, result_dists = umap_module.fuzzy_simplicial_set(
        data,
        knn_indices.shape[1],
        np.random.RandomState(42),
        "euclidean",
        knn_indices=knn_indices,
        knn_dists=knn_dists,
        set_op_mix_ratio=mix_ratio,
        return_dists=True,
    )

    np.testing.assert_allclose(result.toarray(), expected.toarray(), rtol=1e-7)
    np.testing.assert_array_equal(result_dists.toarray(), expected_dists.toarray())


def test_transform_cache_uses_equal_input_copy_and_detects_mutation():
    data, _ = make_blobs(
        n_samples=40,
        n_features=4,
        centers=3,
        random_state=42,
    )
    data = data.astype(np.float32)
    model = UMAP(n_neighbors=5, n_epochs=5, random_state=42).fit(data)

    assert model.transform(data.copy()) is model.embedding_

    data[0, 0] += 1.0
    assert model.transform(data) is not model.embedding_


def test_transform_supports_legacy_joblib_input_hash():
    data, _ = make_blobs(
        n_samples=40,
        n_features=4,
        centers=3,
        random_state=42,
    )
    model = UMAP(n_neighbors=5, n_epochs=5, random_state=42).fit(data)
    del model._input_hash_algorithm
    model._input_hash = umap_module.joblib.hash(model._raw_data)

    assert model.transform(model._raw_data.copy()) is model.embedding_


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
    embedding = UMAP(n_epochs=100, compatibility_layout=True).fit_transform(data)
    assert adjusted_rand_score(labels, KMeans(5).fit_predict(embedding)) >= 0.99


# Umap Clusterability
def test_blobs_cluster_adam():
    data, labels = make_blobs(n_samples=500, n_features=10, centers=5)
    embedding = UMAP(n_epochs=100, optimizer="adam").fit_transform(data)
    assert adjusted_rand_score(labels, KMeans(5).fit_predict(embedding)) == 1.0


def test_adam_is_default_optimizer():
    assert UMAP().optimizer == "adam"


def test_blobs_cluster_momentum():
    data, labels = make_blobs(n_samples=500, n_features=10, centers=5, random_state=42)
    embedding = UMAP(n_epochs=100, optimizer="momentum", random_state=42).fit_transform(
        data
    )
    assert adjusted_rand_score(labels, KMeans(5).fit_predict(embedding)) == 1.0


@pytest.mark.parametrize("optimizer", ["momentum", "adam"])
def test_modern_hard_negatives_fit_and_transform(optimizer):
    data, _ = make_blobs(
        n_samples=1050,
        n_features=6,
        centers=5,
        random_state=42,
    )
    model = UMAP(
        n_neighbors=10,
        n_epochs=15,
        init="random",
        optimizer=optimizer,
        negative_selection_range=250,
        negative_sample_scale=0.25,
        negative_sample_scale_adaptation_samples=0,
        exclude_graph_neighbors=True,
        random_state=42,
    ).fit(data)

    transformed = model.transform(data[:31] + 0.01)

    assert model.embedding_.shape == (1050, 2)
    assert transformed.shape == (31, 2)
    assert np.isfinite(model.embedding_).all()
    assert np.isfinite(transformed).all()


@pytest.mark.parametrize(
    "optimizer",
    ["standard", "densmap_standard", "densmap_adam", "densmap_momentum"],
)
def test_removed_optimizer_names_are_rejected(optimizer):
    data, _ = make_blobs(n_samples=30, n_features=4, random_state=42)
    with pytest.raises(ValueError, match="Unknown optimizer"):
        UMAP(optimizer=optimizer).fit(data)


def test_hard_negative_scale_adaptation_default():
    assert UMAP().negative_sample_scale_adaptation_samples == 128
    assert (
        UMAP(
            negative_sample_scale_adaptation_samples=0
        ).negative_sample_scale_adaptation_samples
        == 0
    )


# Umap Clusterability
def test_blobs_cluster_compatibility_mode():
    data, labels = make_blobs(n_samples=500, n_features=10, centers=5)
    embedding = UMAP(n_epochs=100, compatibility_layout=True).fit_transform(data)
    assert adjusted_rand_score(labels, KMeans(5).fit_predict(embedding)) == 1.0


# Umap Clusterability
def test_blobs_cluster_compatibility_optimizer():
    data, labels = make_blobs(n_samples=500, n_features=10, centers=5)
    embedding = UMAP(n_epochs=100, optimizer="compatibility").fit_transform(data)
    assert adjusted_rand_score(labels, KMeans(5).fit_predict(embedding)) == 1.0


@pytest.mark.parametrize("output_metric", ["euclidean", "haversine"])
@pytest.mark.parametrize(
    "compatibility_option",
    [{"compatibility_layout": True}, {"optimizer": "compatibility"}],
)
def test_compatibility_fit_and_transform_dispatch(
    iris, output_metric, compatibility_option
):
    common_parameters = {
        "n_neighbors": 10,
        "n_epochs": 30,
        "init": "spectral",
        "output_metric": output_metric,
        "random_state": 42,
    }
    model = UMAP(
        **common_parameters,
        **compatibility_option,
    ).fit(iris.data)

    assert np.isfinite(model.embedding_).all()
    assert np.isfinite(model.transform(iris.data[:10])).all()


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

    assert error < 1.0
    assert new_model.transform(new_model._raw_data.copy()) is new_model.embedding_


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
