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


def test_l9_final_fit_rebuild_and_transform_fallback(traced_layout):
    calls, keys = traced_layout
    x = np.random.RandomState(17).normal(size=(1300, 4)).astype(np.float32)
    model = UMAP(compatibility_layout=False, optimizer="adam", init="random", n_neighbors=10,
                 n_epochs=50, n_jobs=1, random_state=42, negative_selection_range=200)
    y = model.fit_transform(x)
    assert y.shape == (1300, 2) and np.isfinite(y).all()
    assert calls == list(range(50))
    assert len(keys) > 1 and len(set(keys)) == len(keys)
    before = len(calls)
    assert np.isfinite(model.transform(x[:5])).all()
    assert len(calls) == before


@pytest.mark.parametrize("overrides,size", [
    ({}, 500),
    ({"n_components": 3}, 1300),
    ({"a": 1.5, "b": 2.0}, 1300),
    ({"repulsion_strength": 0.0}, 1300),
    ({"exclude_graph_neighbors": True}, 1300),
    ({"densmap": True}, 1300),
    ({"optimizer": "momentum"}, 1300),
])
def test_l9_ineligible_fit_uses_native(traced_layout, overrides, size):
    calls, keys = traced_layout
    x = np.random.RandomState(31).normal(size=(size, 4)).astype(np.float32)
    params = dict(compatibility_layout=False, optimizer="adam", init="random", n_neighbors=10,
                  n_epochs=12, n_jobs=1, random_state=42)
    y = UMAP(**(params | overrides)).fit_transform(x)
    assert np.isfinite(y).all()
    assert calls == [] and keys == []
