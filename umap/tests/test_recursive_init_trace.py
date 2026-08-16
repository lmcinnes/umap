import numpy as np
import pytest
from scipy.sparse import csr_matrix

import umap.label_prop as label_prop
import umap.umap_ as umap_module
import experiments.recursive_init.run_mnist as run_mnist
from experiments.recursive_init.run_coarsening import (
    _format_optional,
    _paired_difference,
)
from umap import UMAP


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_procrustes_align_preserves_input_precision(dtype):
    left = np.eye(3, 2, dtype=dtype)
    right = np.array([[0, 1], [-1, 0], [1, 1]], dtype=dtype)

    rotation = label_prop.procrustes_align(left, right)

    assert rotation.dtype == dtype
    assert rotation.flags.c_contiguous
    np.testing.assert_allclose(rotation.T @ rotation, np.eye(2), atol=1e-5)


def test_coarsen_graph_can_remove_partition_self_edges():
    graph = csr_matrix(np.ones((6, 6), dtype=np.float32))
    graph.setdiag(0.0)
    labels = np.array([0, 0, 0, 1, 1, 1], dtype=np.int32)

    _, reduced_graph = label_prop._coarsen_graph(graph, labels, remove_diagonal=True)

    assert np.count_nonzero(reduced_graph.diagonal()) == 0
    assert reduced_graph.nnz == 2


def test_diverse_hubs_avoid_adjacent_candidates_when_possible():
    graph = csr_matrix(
        np.array(
            [
                [0, 1, 1, 0, 0, 0],
                [1, 0, 1, 0, 0, 0],
                [1, 1, 0, 1, 0, 0],
                [0, 0, 1, 0, 1, 1],
                [0, 0, 0, 1, 0, 1],
                [0, 0, 0, 1, 1, 0],
            ],
            dtype=np.float32,
        )
    )
    labels = np.full(6, -1, dtype=np.int32)
    degrees = np.asarray(graph.sum(axis=1)).ravel()

    result = label_prop.initialize_labels_from_diverse_hubs(
        labels, 2, degrees, graph.indptr, graph.indices
    )
    hubs = np.flatnonzero(result >= 0)

    assert hubs.shape[0] == 2
    assert graph[hubs[0], hubs[1]] == 0.0


@pytest.mark.parametrize(
    "schedule, expected_initial, expected_next",
    [
        ("original", (8.0, 27.0), (8.0, 27.0)),
        (
            "strong_to_one",
            (np.power(8.0, 0.25), np.power(27.0, 0.25)),
            (np.power(np.power(8.0, 0.25), 0.25), np.power(np.power(27.0, 0.25), 0.25)),
        ),
    ],
)
def test_curve_parameter_schedules(schedule, expected_initial, expected_next):
    initial = label_prop._initial_curve_parameters(8.0, 27.0, schedule)
    following = label_prop._next_curve_parameters(*initial, schedule)

    assert np.allclose(initial, expected_initial)
    assert np.allclose(following, expected_next)


def test_available_good_initialization_requires_finite_nonzero_spread():
    good = np.array([[0.0, 0.0], [1.0, 2.0], [2.0, 4.0]], dtype=np.float32)
    flat = np.ones((3, 2), dtype=np.float32)
    nonfinite = good.copy()
    nonfinite[0, 0] = np.nan

    assert label_prop._good_initialization(good, 8, 4, "available")
    assert not label_prop._good_initialization(flat, 8, 4, "available")
    assert not label_prop._good_initialization(nonfinite, 8, 4, "available")
    assert not label_prop._good_initialization(good, 8, 4, "heuristic")
    assert label_prop._good_initialization(flat, 8, 4, "always")
    assert label_prop._good_initialization(nonfinite, 8, 4, "always")
    assert label_prop._good_initialization(None, 8, 4, "always")
    assert not label_prop._good_initialization(good, 8, 4, "never")
    assert not label_prop._good_initialization(None, 8, 4, "never")

    with pytest.raises(ValueError, match="good_initialization_policy"):
        label_prop._good_initialization(good, 8, 4, "unknown")


def test_stratified_subset_is_deterministic_and_balanced():
    data = np.arange(120, dtype=np.float32).reshape(60, 2)
    target = np.repeat(np.arange(3), 20)

    first = run_mnist._stratified_subset(data, target, 30, seed=17)
    second = run_mnist._stratified_subset(data, target, 30, seed=17)

    assert np.array_equal(first[2], second[2])
    assert np.array_equal(np.bincount(first[1]), [10, 10, 10])
    assert first[0].flags.c_contiguous


def test_covtype_loader_standardizes_selected_sample(monkeypatch, tmp_path):
    class Dataset:
        data = np.column_stack(
            [np.arange(60, dtype=np.float32), np.arange(60, dtype=np.float32) ** 2]
        )
        target = np.repeat(np.arange(1, 4), 20)

    monkeypatch.setattr(run_mnist, "fetch_covtype", lambda data_home: Dataset())

    data, target, indices = run_mnist.load_dataset("covtype", 30, 17, tmp_path)

    assert data.shape == (30, 2)
    assert target.shape == (30,)
    assert indices.shape == (30,)
    assert data.dtype == np.float32
    assert data.flags.c_contiguous
    assert np.allclose(data.mean(axis=0), 0.0, atol=1e-6)
    assert np.allclose(data.std(axis=0), 1.0, atol=1e-6)


def test_optional_report_metric_formats_single_seed_as_unavailable():
    assert _format_optional(None) == "n/a"
    assert _format_optional(0.123456) == "0.1235"


def test_paired_difference_matches_runs_by_seed():
    baseline = [{"seed": 0, "value": 1.0}, {"seed": 1, "value": 2.0}]
    candidate = [{"seed": 1, "value": 2.5}, {"seed": 0, "value": 1.25}]

    mean, std = _paired_difference(baseline, candidate, lambda run: run["value"])

    assert mean == pytest.approx(0.375)
    assert std == pytest.approx(np.std([0.25, 0.5], ddof=1))


@pytest.mark.parametrize("ratio", [2, 3, 4])
def test_umap_routes_recursive_coarsening_ratio(monkeypatch, ratio):
    captured = {}

    def fake_recursive_init(graph, data, a, b, **kwargs):
        captured["ratio"] = kwargs["coarsening_ratio"]
        return (
            np.random.RandomState(42)
            .normal(size=(graph.shape[0], 2))
            .astype(np.float32)
        )

    monkeypatch.setattr(umap_module, "recursive_init", fake_recursive_init)
    data = np.random.RandomState(42).normal(size=(40, 4)).astype(np.float32)

    UMAP(
        init="recursive",
        recursive_coarsening_ratio=ratio,
        n_neighbors=5,
        n_epochs=5,
        random_state=42,
    ).fit(data)

    assert captured["ratio"] == ratio


def test_pca_anchor_methods_have_expected_membership():
    data = np.random.RandomState(8).normal(size=(40, 6)).astype(np.float32)
    graph = csr_matrix(np.eye(40, dtype=np.float32))

    sampled, sampled_mask = label_prop._anchor_reference(
        graph,
        data,
        2,
        "sampled_pca",
        12,
        np.random.RandomState(9),
    )
    full, full_mask = label_prop._anchor_reference(
        graph,
        data,
        2,
        "projected_pca",
        12,
        np.random.RandomState(9),
    )

    assert sampled.shape == (12, 2)
    assert sampled_mask.sum() == 12
    assert full.shape == (40, 2)
    assert full_mask.all()
    assert np.all(np.isfinite(sampled))
    assert np.all(np.isfinite(full))
    assert np.allclose(sampled.min(axis=0), 0.0)
    assert np.allclose(full.max(axis=0), 10.0)


def test_projected_pca_anchor_supports_sparse_projection():
    data = csr_matrix(np.random.RandomState(10).normal(size=(30, 5)).astype(np.float32))
    anchor, mask = label_prop._anchor_reference(
        csr_matrix(np.eye(30, dtype=np.float32)),
        data,
        2,
        "projected_pca",
        10,
        np.random.RandomState(11),
    )

    assert anchor.shape == (30, 2)
    assert mask.all()
    assert np.all(np.isfinite(anchor))


def test_unknown_anchor_method_is_rejected():
    with pytest.raises(ValueError, match="anchor_method"):
        label_prop._anchor_reference(
            csr_matrix(np.eye(5, dtype=np.float32)),
            np.ones((5, 3), dtype=np.float32),
            2,
            "unknown",
            5,
            np.random.RandomState(13),
        )


def test_resolve_negative_selection_range_modes():
    assert label_prop._resolve_negative_selection_range(123, "coarse_n") == 123
    assert label_prop._resolve_negative_selection_range(123, "fixed_200k") == 200_000
    assert (
        label_prop._resolve_negative_selection_range(123, "scaled_coarse", scale=0.25)
        == 31
    )
    assert (
        label_prop._resolve_negative_selection_range(123, "scaled_coarse", scale=2.0)
        == 123
    )


def test_resolve_negative_selection_range_rejects_unknown_mode():
    with pytest.raises(ValueError, match="recursive_negative_selection_range_mode"):
        label_prop._resolve_negative_selection_range(16, "bad_mode")


def test_resolve_depth_schedule_supports_scalars_and_default_dicts():
    assert label_prop._resolve_depth_schedule(4.0, 3) == 4.0
    assert label_prop._resolve_depth_schedule({1: 8.0, "default": 4.0}, 1) == 8.0
    assert label_prop._resolve_depth_schedule({1: 8.0, "default": 4.0}, 3) == 4.0
    assert label_prop._resolve_depth_schedule({"2": 0.25, "default": 0.5}, 2) == 0.25


def test_resolve_depth_schedule_requires_depth_or_default():
    with pytest.raises(ValueError, match="depth schedule dict"):
        label_prop._resolve_depth_schedule({1: 2.0}, 3)
