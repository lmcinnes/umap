import numpy as np
import pytest
from sklearn.neighbors import NearestNeighbors

from umap import umato


def _knn(X, k=10):
    return NearestNeighbors(n_neighbors=k).fit(X).kneighbors(X)[1]


@pytest.fixture(scope="module")
def blobs():
    from sklearn.datasets import make_blobs

    X, y = make_blobs(600, 8, centers=4, random_state=0)
    return X.astype(np.float32), y


# --- hub selection and expansion -------------------------------------------


@pytest.mark.parametrize("n_samples, expected", [(5, 2), (30, 10), (3000, 300)])
def test_default_n_hubs(n_samples, expected):
    assert umato.default_n_hubs(n_samples) == expected


def test_select_hubs_partitions_every_point(blobs):
    X, _ = blobs
    knn = _knn(X)
    hubs, groups = umato.select_hubs(knn, 50)
    group_size = int(np.ceil(X.shape[0] / 50))

    assert len(hubs) <= 50
    assert len(np.unique(hubs)) == len(hubs)
    assert np.all(groups >= 0)
    assert np.array_equal(np.unique(groups), np.arange(len(hubs)))
    assert np.bincount(groups).max() <= group_size
    # Each hub is the first point of its own group.
    assert np.array_equal(groups[hubs], np.arange(len(hubs)))


def test_select_hubs_starts_from_most_frequent_neighbor(blobs):
    X, _ = blobs
    knn = _knn(X)
    hubs, _ = umato.select_hubs(knn, 50)
    counts = np.bincount(knn.ravel(), minlength=X.shape[0])
    assert counts[hubs[0]] == counts.max()


def test_select_hubs_ignores_missing_neighbors(blobs):
    X, _ = blobs
    knn = _knn(X)
    knn[:, -3:] = -1
    hubs, groups = umato.select_hubs(knn, 40)
    assert np.all(hubs >= 0)
    assert np.all(groups >= 0)


@pytest.mark.parametrize("n_hubs", [1, 600, 1000])
def test_select_hubs_rejects_invalid_counts(blobs, n_hubs):
    X, _ = blobs
    with pytest.raises(ValueError, match="n_hubs"):
        umato.select_hubs(_knn(X), n_hubs)


def test_expand_hubs_is_closed_under_neighbors(blobs):
    X, _ = blobs
    knn = _knn(X)
    hubs, _ = umato.select_hubs(knn, 30)
    hub_info = umato.expand_hubs(knn, hubs)

    assert np.all(hub_info[hubs] == umato.HUB)
    members = np.flatnonzero(hub_info > umato.OUTLIER)
    assert np.all(hub_info[knn[members]] > umato.OUTLIER)


def test_expand_hubs_leaves_unreachable_points_as_outliers():
    # 0 -> 1 -> 2 is reachable from hub 0; 3 points to 0 but nothing points to 3.
    knn = np.array([[0, 1], [1, 2], [2, 1], [3, 0]])
    hub_info = umato.expand_hubs(knn, np.array([0]))
    assert hub_info.tolist() == [umato.HUB, umato.EXPANDED, umato.EXPANDED, 0]


# --- global hub layout -----------------------------------------------------


def _ab():
    from umap.umap_ import find_ab_params

    return find_ab_params(1.0, 0.1)


def test_hub_distances_match_pairwise_euclidean(blobs):
    from scipy.spatial.distance import cdist
    from umap.distances import named_distances

    X, _ = blobs
    hubs = np.arange(0, 600, 30)
    D = umato.hub_distances(X, hubs, named_distances["euclidean"], {})
    np.testing.assert_allclose(D, cdist(X[hubs], X[hubs]), rtol=1e-4, atol=1e-4)


def test_hub_distances_read_precomputed_matrix(blobs):
    import scipy.sparse
    from scipy.spatial.distance import cdist

    X, _ = blobs
    full = cdist(X, X)
    hubs = np.arange(0, 600, 60)
    D = umato.hub_distances(full, hubs, "precomputed", {})
    np.testing.assert_allclose(D, full[np.ix_(hubs, hubs)])

    sparse = scipy.sparse.csr_matrix(np.where(full < np.median(full), full, 0.0))
    D_sparse = umato.hub_distances(sparse, hubs, "precomputed", {})
    assert np.all(np.isfinite(D_sparse))
    assert np.all(D_sparse[~np.eye(len(hubs), dtype=bool)] > 0)


@pytest.mark.parametrize("init", ["pca", "random", "spectral", "array"])
def test_initial_hub_layout_is_scaled(blobs, init):
    X, _ = blobs
    from scipy.spatial.distance import cdist

    hubs = np.arange(0, 600, 20)
    D = cdist(X[hubs], X[hubs])
    if init == "array":
        init = np.random.RandomState(0).normal(size=(600, 2))
    Z = umato.initial_hub_layout(init, X[hubs], D, hubs, 2, np.random.RandomState(0))
    assert Z.shape == (len(hubs), 2)
    assert np.isclose(np.abs(Z).max(), 1.0)


def test_initial_hub_layout_pads_missing_pca_dimensions():
    X = np.random.RandomState(0).normal(size=(40, 2))
    hubs = np.arange(10)
    Z = umato.initial_hub_layout(
        "pca", X[hubs], None, hubs, 3, np.random.RandomState(0)
    )
    assert Z.shape == (10, 3)
    assert np.all(np.isfinite(Z))


def test_global_hub_layout_preserves_cluster_arrangement():
    from scipy.spatial.distance import cdist
    from scipy.stats import spearmanr

    # Five clusters whose centers lie at very different distances.
    centers = np.array([[0, 0], [1, 0], [5, 0], [5, 6], [-9, 4]], float) * 10
    rng = np.random.RandomState(0)
    X = np.vstack([c + rng.normal(size=(40, 2)) for c in centers]).astype(np.float32)
    X = np.hstack([X, rng.normal(scale=0.1, size=(200, 3))]).astype(np.float32)
    labels = np.repeat(np.arange(5), 40)
    hubs = np.arange(0, 200, 4)

    from umap.distances import named_distances

    a, b = _ab()
    Z = umato.global_hub_layout(
        X,
        hubs,
        2,
        a,
        b,
        named_distances["euclidean"],
        {},
        "pca",
        100,
        np.random.RandomState(0),
    )
    assert Z.shape == (len(hubs), 2)
    assert np.all(np.isfinite(Z))

    hub_labels = labels[hubs]
    low = np.array([Z[hub_labels == k].mean(0) for k in range(5)])
    iu = np.triu_indices(5, 1)
    rho = spearmanr(cdist(centers, centers)[iu], cdist(low, low)[iu])[0]
    assert rho > 0.8


# --- placing the expanded points and the outliers ---------------------------


def test_nearest_hubs_match_brute_force(blobs):
    from scipy.spatial.distance import cdist
    from umap.distances import named_distances

    X, _ = blobs
    hubs = np.arange(0, 600, 15)
    points = np.arange(1, 600, 7)
    nearest = umato.nearest_hubs(
        X, points, hubs, named_distances["euclidean"], {}, 5, chunk_size=16
    )
    expected = np.argsort(cdist(X[points], X[hubs]), axis=1)[:, :5]
    assert np.array_equal(np.sort(nearest, axis=1), np.sort(expected, axis=1))

    precomputed = umato.nearest_hubs(cdist(X, X), points, hubs, "precomputed", {}, 5)
    assert np.array_equal(np.sort(precomputed, axis=1), np.sort(expected, axis=1))


def test_embed_from_hubs_places_points_near_their_hubs(blobs):
    from umap.distances import named_distances

    X, y = blobs
    knn = _knn(X)
    hubs, _ = umato.select_hubs(knn, 120)
    hub_info = umato.expand_hubs(knn, hubs)
    # Put every hub of a cluster at that cluster's corner of a square.
    corners = np.array([[0, 0], [10, 0], [0, 10], [10, 10]], np.float32)
    hub_layout = corners[y[hubs]]

    embedding = umato.embed_from_hubs(
        X,
        hub_layout,
        hubs,
        hub_info,
        named_distances["euclidean"],
        {},
        np.random.RandomState(0),
    )
    np.testing.assert_array_equal(embedding[hubs], hub_layout)
    expanded = np.flatnonzero(hub_info == umato.EXPANDED)
    assert np.all(
        np.linalg.norm(embedding[expanded] - corners[y[expanded]], axis=1) < 1
    )
    assert np.all(embedding[hub_info == umato.OUTLIER] == 0)


def test_place_outliers_uses_embedded_neighbors():
    embedding = np.array([[0, 0], [2, 0], [0, 2], [9, 9], [9, 9]], np.float32)
    hub_info = np.array([2, 1, 1, 0, 0], np.int8)
    # Outlier 3 points to 1 and 2; outlier 4 only points to outlier 3.
    knn = np.array([[0, 1], [1, 0], [2, 0], [1, 2], [3, -1]])
    result = umato.place_outliers(embedding, hub_info, knn, np.random.RandomState(0))
    np.testing.assert_allclose(result[3], [1, 1], atol=0.1)
    np.testing.assert_allclose(result[4], result[3], atol=0.1)


def test_place_outliers_falls_back_to_the_mean():
    embedding = np.array([[0, 0], [2, 0], [5, 5]], np.float32)
    hub_info = np.array([2, 1, 0], np.int8)
    knn = np.array([[0, 1], [1, 0], [2, -1]])
    result = umato.place_outliers(embedding, hub_info, knn, np.random.RandomState(0))
    np.testing.assert_allclose(result[2], [1, 0], atol=0.1)
