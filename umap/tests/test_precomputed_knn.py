import numpy as np
import pytest
from scipy.sparse import csr_matrix
from sklearn.metrics import pairwise_distances

from umap import UMAP

pytestmark = [
    pytest.mark.filterwarnings("ignore:precomputed_knn.*:UserWarning"),
    pytest.mark.filterwarnings("ignore:using precomputed metric.*:UserWarning"),
    pytest.mark.filterwarnings("ignore:A large number of your vertices.*:UserWarning"),
]


@pytest.fixture(scope="module")
def neighbors():
    data = np.random.RandomState(42).normal(size=(96, 4)).astype(np.float32)
    distances = pairwise_distances(data)
    indices = np.argsort(distances, axis=1, kind="stable")[:, :20]
    knn_distances = np.take_along_axis(distances, indices, axis=1)
    return data, distances, indices, knn_distances


def fit_with_neighbors(data, cache, compatibility_layout, **kwargs):
    return UMAP(
        n_neighbors=5,
        n_epochs=0,
        init="random",
        random_state=42,
        n_jobs=1,
        compatibility_layout=compatibility_layout,
        precomputed_knn=cache,
        **kwargs,
    ).fit(data)


@pytest.mark.parametrize("compatibility_layout", [True, False])
@pytest.mark.parametrize("force_approximation", [False, True])
@pytest.mark.parametrize("sliced", [False, True])
def test_precomputed_knn_trims_larger_cache(
    neighbors, compatibility_layout, force_approximation, sliced
):
    data, _, indices, distances = neighbors
    cache = (indices[:, :10], distances[:, :10])
    if not sliced:
        cache = tuple(array.copy() for array in cache)
    actual = fit_with_neighbors(
        data,
        cache,
        compatibility_layout,
        force_approximation_algorithm=force_approximation,
    )
    expected = fit_with_neighbors(
        data,
        (indices[:, :5].copy(), distances[:, :5].copy()),
        compatibility_layout,
        force_approximation_algorithm=force_approximation,
    )

    np.testing.assert_array_equal(actual._knn_indices, indices[:, :5])
    np.testing.assert_array_equal(actual._knn_dists, distances[:, :5])
    np.testing.assert_array_equal(actual.graph_.toarray(), expected.graph_.toarray())


@pytest.mark.parametrize("compatibility_layout", [True, False])
@pytest.mark.parametrize(
    "input_kind", ["features", "dense_distances", "sparse_distances"]
)
@pytest.mark.parametrize("read_only", [False, True])
@pytest.mark.parametrize("disconnect", [False, True])
@pytest.mark.parametrize("sliced", [False, True])
def test_precomputed_knn_preserves_reusable_cache(
    neighbors, compatibility_layout, input_kind, read_only, disconnect, sliced
):
    data, distance_matrix, indices, distances = neighbors
    metric = "euclidean"
    if input_kind != "features":
        metric = "precomputed"
        data = (
            csr_matrix(distance_matrix)
            if input_kind == "sparse_distances"
            else distance_matrix
        )
    cache = (indices.copy()[:, :5], distances.copy()[:, :5])
    if not sliced:
        cache = tuple(array.copy() for array in cache)
    original = tuple(array.copy() for array in cache)
    for array in cache:
        array.flags.writeable = not read_only
    cutoff = float(np.median(cache[1][:, 1:])) if disconnect else np.inf
    model = fit_with_neighbors(
        data,
        cache,
        compatibility_layout,
        metric=metric,
        disconnection_distance=cutoff,
    )

    for actual, expected in zip(cache, original):
        np.testing.assert_array_equal(actual, expected)
        assert actual.flags.writeable == (not read_only)
    if disconnect:
        assert np.any(model._knn_indices == -1)
        assert np.isinf(model._knn_dists).any()
    else:
        np.testing.assert_array_equal(model._knn_indices, original[0])
        np.testing.assert_array_equal(model._knn_dists, original[1])
        assert np.shares_memory(model._knn_indices, cache[0])
        assert np.shares_memory(model._knn_dists, cache[1])

    reused = fit_with_neighbors(data, cache, compatibility_layout, metric=metric)
    fresh = fit_with_neighbors(data, original, compatibility_layout, metric=metric)
    np.testing.assert_array_equal(reused.graph_.toarray(), fresh.graph_.toarray())


@pytest.mark.parametrize("compatibility_layout", [True, False])
@pytest.mark.parametrize("invalid_shape", ["too_few_neighbors", "wrong_sample_count"])
def test_precomputed_knn_invalid_cache_falls_back(
    neighbors, compatibility_layout, invalid_shape
):
    data, _, indices, distances = neighbors
    cache = (
        (indices[:, :4], distances[:, :4])
        if invalid_shape == "too_few_neighbors"
        else (indices[:-1], distances[:-1])
    )
    with pytest.warns(UserWarning, match="precomputed_knn.*ignored"):
        actual = fit_with_neighbors(data, cache, compatibility_layout)
    expected = fit_with_neighbors(data, (None, None, None), compatibility_layout)

    np.testing.assert_array_equal(actual.graph_.toarray(), expected.graph_.toarray())
