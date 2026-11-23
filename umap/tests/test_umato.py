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
