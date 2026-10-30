"""Tests for the 0.6 transition-release layout defaults and their resolution."""

import pickle
import warnings

import numpy as np
import pytest
from sklearn.base import clone
from sklearn.datasets import make_blobs

from umap import UMAP
from umap.umap_ import _resolve_layout_options


@pytest.fixture(scope="module")
def blobs():
    data, _ = make_blobs(120, n_features=5, centers=3, random_state=0)
    return data.astype(np.float32)


def _fit(data, **kwargs):
    kwargs.setdefault("n_neighbors", 10)
    kwargs.setdefault("n_epochs", 20)
    return UMAP(**kwargs).fit(data)


def test_default_parameters_are_unresolved_sentinels():
    params = UMAP().get_params()
    assert params["compatibility_layout"] is None
    assert params["optimizer"] is None
    assert params["init"] == "auto"


def test_default_uses_legacy_pipeline_with_future_warning(blobs):
    with pytest.warns(FutureWarning, match="compatibility_layout"):
        model = _fit(blobs, random_state=1)
    assert model._compatibility_layout is True
    assert model._optimizer == "compatibility"
    assert model._init == "spectral"
    # Public parameters are never rewritten (scikit-learn estimator contract).
    assert model.compatibility_layout is None
    assert model.optimizer is None
    assert model.init == "auto"


@pytest.mark.parametrize(
    "kwargs, expected",
    [
        (dict(compatibility_layout=True), (True, "compatibility", "spectral")),
        (dict(compatibility_layout=False), (False, "adam", "recursive")),
        (dict(optimizer="adam"), (False, "adam", "recursive")),
        (dict(optimizer="momentum"), (False, "momentum", "recursive")),
        (
            dict(compatibility_layout=False, optimizer="compatibility"),
            (False, "compatibility", "recursive"),
        ),
        (dict(compatibility_layout=False, init="pca"), (False, "adam", "pca")),
        (
            dict(compatibility_layout=True, init="random"),
            (True, "compatibility", "random"),
        ),
    ],
)
def test_explicit_choices_resolve_without_future_warning(blobs, kwargs, expected):
    with warnings.catch_warnings():
        warnings.simplefilter("error", FutureWarning)
        model = _fit(blobs, random_state=1, **kwargs)
    assert (model._compatibility_layout, model._optimizer, model._init) == expected


@pytest.mark.parametrize("optimizer", ["adam", "momentum"])
def test_modern_optimizer_conflicts_with_compatibility_layout(blobs, optimizer):
    with pytest.raises(ValueError, match="requires compatibility_layout=False"):
        _fit(blobs, compatibility_layout=True, optimizer=optimizer)


@pytest.mark.parametrize("value", ["yes", 1, 0.0])
def test_compatibility_layout_must_be_bool_or_none(blobs, value):
    with pytest.raises(ValueError, match="compatibility_layout must be"):
        _fit(blobs, compatibility_layout=value)


def test_unknown_optimizer_rejected(blobs):
    with pytest.raises(ValueError, match="Unknown optimizer"):
        _fit(blobs, optimizer="standard", compatibility_layout=False)


def test_clone_preserves_unresolved_parameters(blobs):
    model = UMAP(n_neighbors=10, n_epochs=20, random_state=1)
    with pytest.warns(FutureWarning):
        model.fit(blobs)
    cloned = clone(model)
    assert cloned.get_params() == model.get_params()
    assert cloned.init == "auto"


def test_function_level_resolution():
    assert _resolve_layout_options(None, None) == (True, "compatibility")
    assert _resolve_layout_options(None, "adam") == (False, "adam")
    assert _resolve_layout_options(False, None) == (False, "adam")
    assert _resolve_layout_options(False, "compatibility") == (False, "compatibility")
    assert _resolve_layout_options(True, None) == (True, "compatibility")
    with pytest.raises(ValueError):
        _resolve_layout_options(True, "momentum")


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(compatibility_layout=True),
        dict(compatibility_layout=False, optimizer="compatibility"),
        dict(compatibility_layout=False, optimizer="adam"),
        dict(compatibility_layout=False, optimizer="momentum"),
        dict(compatibility_layout=True, densmap=True),
        dict(compatibility_layout=False, optimizer="adam", densmap=True),
        dict(compatibility_layout=False, optimizer="momentum", densmap=True),
    ],
)
def test_seeded_fits_are_reproducible(blobs, kwargs):
    first = _fit(blobs, random_state=7, **kwargs).embedding_
    second = _fit(blobs, random_state=7, **kwargs).embedding_
    np.testing.assert_array_equal(first, second)


NEW_IN_0_6 = (
    "compatibility_layout",
    "optimizer",
    "negative_selection_range",
    "negative_sample_scale",
    "exclude_graph_neighbors",
    "negative_sample_scale_adaptation_samples",
    "recursive_coarsening_ratio",
    "nearest_neighbor_backend",
    "coarse_layout_weights",
    "coarsen_on_layout_weights",
    "init_rescale",
    "init_scale_factor",
    "_compatibility_layout",
    "_optimizer",
    "_init",
    "_data_was_uniqued",
    "_input_hash_algorithm",
)


def _as_legacy_pickle(model):
    state = pickle.loads(pickle.dumps(model)).__getstate__()
    for name in NEW_IN_0_6:
        state.pop(name, None)
    state["init"] = "spectral"
    legacy = UMAP.__new__(UMAP)
    legacy.__setstate__(state)
    return legacy


def test_models_pickled_before_0_6_still_transform_and_update(blobs):
    model = _fit(blobs[:100], random_state=3, compatibility_layout=True)
    expected = model.transform(blobs[100:])

    legacy = _as_legacy_pickle(model)

    assert legacy._compatibility_layout is True
    assert legacy._optimizer == "compatibility"
    assert legacy.compatibility_layout is None
    assert legacy.nearest_neighbor_backend == "auto"
    np.testing.assert_array_equal(legacy.transform(blobs[100:]), expected)
    legacy.inverse_transform(legacy.embedding_[:2])
    legacy.update(blobs[100:])
    assert legacy.embedding_.shape == (blobs.shape[0], 2)
    repr(legacy)


def test_unfitted_legacy_pickle_resolves_on_fit(blobs):
    state = UMAP(n_neighbors=10, n_epochs=20).__getstate__()
    for name in NEW_IN_0_6:
        state.pop(name, None)
    legacy = UMAP.__new__(UMAP)
    legacy.__setstate__(state)
    assert not hasattr(legacy, "_compatibility_layout")
    with pytest.warns(FutureWarning):
        legacy.fit(blobs)
    assert legacy._compatibility_layout is True


def test_update_forwards_layout_options(blobs, monkeypatch):
    import umap.umap_ as umap_module

    captured = []
    original = umap_module.simplicial_set_embedding

    def capture(*args, **kwargs):
        captured.append(kwargs)
        return original(*args, **kwargs)

    model = _fit(
        blobs[:100],
        random_state=3,
        optimizer="momentum",
        negative_selection_range=500,
        init_rescale="per_axis",
        init_scale_factor=0.5,
    )
    monkeypatch.setattr(umap_module, "simplicial_set_embedding", capture)
    model.update(blobs[100:])

    assert captured, "update() did not re-embed"
    kwargs = captured[-1]
    assert kwargs["compatibility_layout"] is False
    assert kwargs["optimizer"] == "momentum"
    assert kwargs["negative_selection_range"] == 500
    assert kwargs["init_rescale"] == "per_axis"
    assert kwargs["init_scale_factor"] == 0.5
