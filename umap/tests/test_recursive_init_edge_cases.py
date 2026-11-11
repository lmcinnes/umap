"""Edge cases and helper-level tests for recursive (label propagation) init."""

import numpy as np
import pytest
import scipy.sparse
from sklearn.datasets import make_blobs

from umap import UMAP
from umap import label_prop
from umap.umap_ import fuzzy_simplicial_set


def _fit(data, **kwargs):
    kwargs.setdefault("n_epochs", 30)
    kwargs.setdefault("random_state", 0)
    return UMAP(compatibility_layout=False, init="recursive", **kwargs).fit_transform(
        data
    )


@pytest.fixture(scope="module")
def multi_level_blobs():
    # Large enough (> base_init_threshold=1024 at two levels) to recurse.
    data, _ = make_blobs(6000, n_features=10, centers=8, random_state=0)
    return data.astype(np.float32)


@pytest.mark.slow
@pytest.mark.parametrize("ratio", [2, 3, 4])
@pytest.mark.parametrize("weights", ["transition", "fuzzy_union"])
def test_multi_level_recursive_init(multi_level_blobs, ratio, weights):
    embedding = _fit(
        multi_level_blobs,
        recursive_coarsening_ratio=ratio,
        coarse_layout_weights=weights,
    )
    assert embedding.shape == (multi_level_blobs.shape[0], 2)
    assert np.all(np.isfinite(embedding))


@pytest.mark.slow
def test_multi_level_recursive_init_is_reproducible(multi_level_blobs):
    graph = fuzzy_simplicial_set(
        multi_level_blobs, 15, 0, "euclidean", compatibility_layout=False
    )[0].tocsr()
    first, second = (
        label_prop.recursive_init(
            graph, multi_level_blobs, 1.6, 0.9, random_state=np.random.RandomState(3)
        )
        for _ in range(2)
    )
    np.testing.assert_array_equal(first, second)


@pytest.mark.slow
def test_multi_level_without_recursive_coarse_init(multi_level_blobs):
    graph = fuzzy_simplicial_set(
        multi_level_blobs, 15, 0, "euclidean", compatibility_layout=False
    )[0].tocsr()
    anchor, mask = label_prop._anchor_reference(
        graph, multi_level_blobs, 2, "sampled_pca", 16384, np.random.RandomState(0)
    )
    result = label_prop.label_propagation_init(
        graph,
        anchor,
        mask,
        1.6,
        0.9,
        random_state=np.random.RandomState(0),
        recursive_init=False,
    )
    assert result.shape == (multi_level_blobs.shape[0], 2)
    assert np.all(np.isfinite(result))


@pytest.mark.parametrize(
    "name, kwargs",
    [
        ("three_components", dict(n_components=3)),
        ("float64", dict()),
        ("tiny", dict(n_neighbors=5)),
    ],
)
def test_recursive_init_variants(name, kwargs):
    data, _ = make_blobs(300, n_features=6, centers=3, random_state=1)
    if name == "tiny":
        data = data[:30]
    if name != "float64":
        data = data.astype(np.float32)
    embedding = _fit(data, **kwargs)
    assert embedding.shape == (data.shape[0], kwargs.get("n_components", 2))
    assert np.all(np.isfinite(embedding))


def test_recursive_init_identical_points():
    embedding = _fit(np.ones((300, 5), dtype=np.float32))
    assert np.all(np.isfinite(embedding))


def test_recursive_init_disconnected_components():
    data, _ = make_blobs(400, n_features=6, centers=2, random_state=2)
    data = np.vstack([data, data + 1000.0]).astype(np.float32)
    embedding = _fit(data)
    assert np.all(np.isfinite(embedding))


# ---------------------------------------------------------------------------
# Helper functions on hand-built graphs
# ---------------------------------------------------------------------------


def _path_graph(n):
    rows = np.concatenate([np.arange(n - 1), np.arange(1, n)])
    cols = np.concatenate([np.arange(1, n), np.arange(n - 1)])
    graph = scipy.sparse.csr_matrix(
        (np.ones(rows.shape[0], dtype=np.float32), (rows, cols)), shape=(n, n)
    )
    graph.sort_indices()
    return graph


def test_fill_unsampled_uses_weighted_sampled_neighbours():
    graph = _path_graph(4)
    graph.data[:] = 1.0
    subset_mask = np.array([True, False, True, False])
    data = np.array([[0.0, 0.0], [4.0, 2.0]], dtype=np.float32)

    result = label_prop._fill_unsampled(graph, data, subset_mask)

    np.testing.assert_allclose(result[1], [2.0, 1.0])  # mean of nodes 0 and 2
    np.testing.assert_allclose(result[3], [4.0, 2.0])  # only node 2 is sampled


def test_fill_unsampled_falls_back_for_zero_weight_neighbours():
    graph = _path_graph(3)
    graph.data[:] = 0.0
    subset_mask = np.array([True, False, True])
    data = np.array([[0.0, 0.0], [2.0, 2.0]], dtype=np.float32)

    result = label_prop._fill_unsampled(graph, data, subset_mask)

    np.testing.assert_allclose(result[1], [1.0, 1.0])
    assert np.all(np.isfinite(result))


def test_remap_labels_makes_labels_contiguous_and_labels_outliers():
    labels = np.array([5, -1, 9, 5, -1], dtype=np.int32)
    result = label_prop.remap_labels(labels.copy())
    np.testing.assert_array_equal(result, [0, 2, 1, 0, 3])


def test_remap_labels_all_unlabelled():
    result = label_prop.remap_labels(np.full(3, -1, dtype=np.int32))
    np.testing.assert_array_equal(result, [0, 1, 2])


def test_label_prop_iteration_spreads_hub_labels():
    graph = _path_graph(5)
    graph.data[:] = 2.0
    labels = np.array([0, -1, -1, -1, 1], dtype=np.int32)
    rng_state = np.array([1, 2, 3], dtype=np.int64)
    for _ in range(4):
        labels = label_prop.label_prop_iteration(
            graph.indptr, graph.indices, graph.data, labels, rng_state
        )
    assert np.all(labels >= 0)
    assert labels[0] == 0 and labels[4] == 1


def test_label_outliers_assigns_reachable_label():
    graph = _path_graph(4)
    labels = np.array([-1, -1, -1, 3], dtype=np.int32)
    rng_state = np.array([1, 2, 3], dtype=np.int64)
    result = label_prop.label_outliers(graph.indptr, graph.indices, labels, rng_state)
    np.testing.assert_array_equal(result, [3, 3, 3, 3])


def test_procrustes_align_handles_collapsed_configuration():
    e1 = np.random.RandomState(0).normal(size=(10, 2)).astype(np.float32)
    rotation = label_prop.procrustes_align(e1, np.zeros_like(e1))
    np.testing.assert_array_equal(rotation, np.eye(2, dtype=np.float32))


def test_procrustes_align_recovers_rotation():
    e1 = np.random.RandomState(0).normal(size=(50, 2))
    theta = 0.7
    rot = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    e2 = e1 @ rot
    aligned = e2 @ label_prop.procrustes_align(e1, e2)
    np.testing.assert_allclose(aligned, e1, atol=1e-8)


def test_coarse_layout_graph_sums_weights_between_parts():
    graph = _path_graph(4)
    labels = np.array([0, 0, 1, 1])
    reduction_map, _ = label_prop._coarsen_graph(graph, labels)
    summed = label_prop._coarse_layout_graph(graph, reduction_map, "transition")
    np.testing.assert_allclose(summed.toarray(), [[2.0, 1.0], [1.0, 2.0]])
    with pytest.raises(ValueError, match="coarse_layout_weights"):
        label_prop._coarse_layout_graph(graph, reduction_map, "bogus")


def test_invalid_hub_selection_rejected():
    graph = _path_graph(2000)
    data = np.random.RandomState(0).normal(size=(2000, 2)).astype(np.float32)
    with pytest.raises(ValueError, match="hub_selection must be"):
        label_prop.label_propagation_init(
            graph,
            data,
            np.ones(2000, dtype=bool),
            1.6,
            0.9,
            random_state=np.random.RandomState(0),
            hub_selection="bogus",
        )


@pytest.mark.parametrize("hub_selection", ["degree", "diverse", "random"])
def test_recursive_init_kwds_forwarded(hub_selection):
    data, _ = make_blobs(2500, n_features=6, centers=4, random_state=0)
    embedding = _fit(
        data.astype(np.float32),
        recursive_init_kwds=dict(
            hub_selection=hub_selection,
            base_init_threshold=512,
            anchor_method="full_pca",
        ),
    )
    assert np.all(np.isfinite(embedding))


def test_recursive_init_kwds_rejects_unknown_keys():
    data, _ = make_blobs(100, n_features=4, random_state=0)
    with pytest.raises(ValueError, match="Unknown recursive_init_kwds"):
        _fit(data, recursive_init_kwds=dict(coarsening_ratio=2))
    with pytest.raises(ValueError, match="must be a dict"):
        _fit(data, recursive_init_kwds=[("hub_selection", "random")])


@pytest.mark.parametrize(
    "n_samples, n_features, n_components",
    [
        (6, 50, 7),  # more components than samples
        (200, 3, 5),  # more components than features
    ],
)
@pytest.mark.parametrize("anchor_method", ["sampled_pca", "projected_pca", "full_pca"])
def test_recursive_init_more_components_than_pca_axes(
    n_samples, n_features, n_components, anchor_method
):
    # PCA can only provide min(n_samples, n_features) axes for the anchor;
    # the embedding must still have the requested number of components.
    data = np.random.RandomState(0).normal(size=(n_samples, n_features))
    embedding = _fit(
        data.astype(np.float32),
        n_components=n_components,
        n_neighbors=min(15, n_samples - 1),
        recursive_init_kwds={"anchor_method": anchor_method},
    )
    assert embedding.shape == (n_samples, n_components)
    assert np.all(np.isfinite(embedding))
