"""Focused checks for the eligible 2D Adam radial lookup path."""

import numpy as np
import pytest

from umap import UMAP, layouts


@pytest.fixture
def traced_layout(monkeypatch):
    calls, keys = [], []
    epoch, make = layouts._l9_single_epoch_adam, layouts._l9_make_table

    def traced_epoch(*args, **kwargs):
        calls.append(int(args[14]))
        return epoch(*args, **kwargs)

    def traced_make(*args, **kwargs):
        keys.append(tuple(args))
        return make(*args, **kwargs)

    monkeypatch.setattr(layouts, "_l9_single_epoch_adam", traced_epoch)
    monkeypatch.setattr(layouts, "_l9_make_table", traced_make)
    return calls, keys


def test_l9_lookup_support_and_accuracy():
    a, b, gamma, scale = 1.5, 0.9, 1.0, 0.75
    table = layouts._l9_make_table(a, b, gamma, scale, 9)
    assert table.shape == (60 * 513,)
    assert np.isfinite(table).all()
    for value in (np.float32(0), np.float32(2.0**-41), np.float32(2.0**20), np.float32(np.inf)):
        assert layouts._l9_lookup(value, table, 9) == -1.0
    for value in np.geomspace(2.0**-39, 2.0**19, 500).astype(np.float32):
        radius = np.sqrt(float(value))
        c = scale * 2 * gamma * b / ((0.001 + value) * (1 + a * float(value)**b))
        expected = min(gamma, 16.0) * np.tanh(c * radius / gamma) / radius
        assert layouts._l9_lookup(value, table, 9) == pytest.approx(expected, rel=1e-5)


def test_l9_final_fit_rebuild_and_transform(traced_layout):
    calls, keys = traced_layout
    x = np.random.RandomState(17).normal(size=(1300, 4)).astype(np.float32)
    model = UMAP(compatibility_layout=False, optimizer="adam", init="random", n_neighbors=10,
                 n_epochs=50, n_jobs=1, random_state=42, negative_selection_range=200)
    y = model.fit_transform(x)
    assert y.shape == (1300, 2) and np.isfinite(y).all()
    assert calls == list(range(50))
    assert len(keys) > 1 and len(set(keys)) == len(keys)
    before = len(calls)
    assert np.isfinite(model.transform(x[:5])).all()  # small batch: native kernel
    assert len(calls) == before
    assert np.isfinite(model.transform(x + 0.01)).all()
    assert len(calls) > before


@pytest.mark.parametrize("overrides,size", [
    ({}, 500),
    ({"a": 1.5, "b": 2.0}, 1300),
    ({"repulsion_strength": 0.0}, 1300),
    ({"densmap": True}, 1300),
])
def test_l9_ineligible_fit_uses_native(traced_layout, overrides, size):
    calls, keys = traced_layout
    x = np.random.RandomState(31).normal(size=(size, 4)).astype(np.float32)
    params = dict(compatibility_layout=False, optimizer="adam", init="random", n_neighbors=10,
                  n_epochs=12, n_jobs=1, random_state=42)
    y = UMAP(**(params | overrides)).fit_transform(x)
    assert np.isfinite(y).all()
    assert calls == [] and keys == []


@pytest.mark.parametrize("overrides", [
    {"n_components": 3},
    {"exclude_graph_neighbors": True},
    {"optimizer": "momentum"},
])
def test_l9_extended_fit_uses_table(traced_layout, overrides):
    calls, keys = traced_layout
    x = np.random.RandomState(31).normal(size=(1300, 4)).astype(np.float32)
    params = dict(compatibility_layout=False, optimizer="adam", init="random", n_neighbors=10,
                  n_epochs=12, n_jobs=1, random_state=42)
    y = UMAP(**(params | overrides)).fit_transform(x)
    assert np.isfinite(y).all()
    assert keys
    if overrides.get("optimizer") != "momentum":
        assert calls == list(range(12))


def _two_epoch_layouts(optimizer, n_components, transform):
    rng = np.random.RandomState(5)
    n_tail = 1500
    n_head = 1200 if transform else n_tail
    tail = rng.normal(scale=3.0, size=(n_tail, n_components)).astype(np.float32)
    head = (
        rng.normal(scale=3.0, size=(n_head, n_components)).astype(np.float32)
        if transform else tail
    )
    k = 8
    indices = np.sort(
        np.stack([rng.choice(n_tail, k, replace=False) for _ in range(n_head)]), axis=1
    ).astype(np.int32).ravel()
    indptr = np.arange(0, n_head * k + 1, k, dtype=np.int32)
    data = rng.uniform(0.1, 1.0, n_head * k).astype(np.float32)
    epochs_per_sample = np.ones(n_head * k, dtype=np.float64)
    results = []
    make_table = layouts._l9_make_table
    for use_table in (False, True):
        h = head.copy()
        t = h if not transform else tail.copy()
        enable = use_table
        if optimizer == "adam" and not use_table:
            # Same kernel and negatives, but every lookup misses (-1), so each
            # repulsion is computed directly.
            enable = True
            layouts._l9_make_table = lambda *args: np.full_like(make_table(*args), -1.0)
        try:
            out = layouts.optimize_layout_euclidean(
                h, t, None, None, 2, n_tail, epochs_per_sample, 1.577, 0.895,
                np.zeros(3, dtype=np.int64), 1.0, 1.0, 5.0,
                tqdm_kwds={"disable": True}, csr_indptr=indptr, csr_indices=indices,
                csr_data=data, optimizer=optimizer,
                random_state=np.random.RandomState(0), _enable_l9=enable,
            )
        finally:
            layouts._l9_make_table = make_table
        results.append((head, out.copy()))
    return results


@pytest.mark.parametrize("optimizer", ["adam", "momentum"])
@pytest.mark.parametrize("n_components,transform", [(2, False), (3, False), (2, True)])
def test_l9_table_matches_direct_repulsion(traced_layout, optimizer, n_components, transform):
    calls, keys = traced_layout
    (start, direct), (_, tabled) = _two_epoch_layouts(optimizer, n_components, transform)
    assert keys  # the table path ran for the _enable_l9=True layout
    assert bool(calls) == (optimizer == "adam")
    step_direct = direct - start
    step_tabled = tabled - start
    assert np.abs(step_direct).max() > 0
    np.testing.assert_allclose(step_tabled, step_direct, rtol=1e-3, atol=1e-5)


@pytest.mark.parametrize("hash_key", [-1, 99])
@pytest.mark.parametrize("dim", [2, 3])
def test_l9_single_block_result_ignores_source_order(dim, hash_key):
    # The speedups visit sources in memory order, schedule them dynamically
    # across workers and skip the node shuffle; all rely on a single-block
    # epoch being independent of source order and of the worker split.
    rng = np.random.RandomState(3)
    n, k = 1500, 8
    embedding = rng.normal(scale=3.0, size=(n, dim)).astype(np.float32)
    indices = np.sort(
        np.stack([rng.choice(n, k, replace=False) for _ in range(n)]), axis=1
    ).astype(np.int32).ravel()
    indptr = np.arange(0, n * k + 1, k, dtype=np.int32)
    epochs_per_sample = rng.uniform(1.0, 3.0, n * k)
    epochs_per_negative = epochs_per_sample / 5.0 * 1.5
    table = layouts._l9_make_table(1.577, 0.895, 1.0, 1.0, 9)
    identity = np.arange(n, dtype=np.int32)
    results = []
    schedules = ((identity, 1), (rng.permutation(n).astype(np.int32), 4))
    for order, n_workers in schedules:
        e = embedding.copy()
        next_negative = epochs_per_negative.copy()
        next_sample = epochs_per_sample.copy()
        updates = np.zeros((n, dim), dtype=np.float32)
        adam_m = np.zeros((n, dim), dtype=np.float32)
        adam_v = np.zeros((n, dim), dtype=np.float32)
        for epoch in range(4):
            layouts._l9_single_epoch_adam(
                e, e, indptr, indices, n, epochs_per_sample, 1.577, 0.895, 1.0,
                dim, 1.0, epochs_per_negative, next_negative, next_sample, epoch,
                updates, adam_m, adam_v, 0.9, 0.99, order, identity, identity,
                n, n, 1.0, False, table, n_workers, hash_key,
            )
            updates[:] = 0.0
        results.append(e)
    assert not np.array_equal(results[0], embedding)
    np.testing.assert_array_equal(results[0], results[1])
