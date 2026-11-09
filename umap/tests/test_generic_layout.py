import numpy as np
import pytest
from sklearn.datasets import make_blobs

from umap import UMAP
from umap.layouts import _generic_negative_search_candidates

MODERN_OPTIMIZERS = ("adam", "momentum")


@pytest.mark.parametrize(
    ("n_vertices", "negative_selection_range", "expected"),
    [
        (1000, 1000, 1),
        (1000, 501, 1),
        (1000, 500, 2),
        (1000, 250, 4),
        (1000, 1, 4),
        (1000, 2000, 1),
        (1000, 200_000, 1),
    ],
)
def test_generic_negative_search_candidate_count(
    n_vertices, negative_selection_range, expected
):
    assert (
        _generic_negative_search_candidates(n_vertices, negative_selection_range)
        == expected
    )


def generic_model(optimizer, **kwargs):
    parameters = dict(
        n_neighbors=10,
        n_epochs=20,
        init="random",
        output_metric="haversine",
        optimizer=optimizer,
        random_state=42,
    )
    parameters.update(kwargs)
    return UMAP(**parameters)


@pytest.mark.parametrize("optimizer", MODERN_OPTIMIZERS)
def test_modern_generic_fit_and_asymmetric_transform(optimizer, capsys):
    data, _ = make_blobs(
        n_samples=150,
        n_features=6,
        centers=4,
        random_state=42,
    )
    model = generic_model(optimizer).fit(data[:120])

    transformed = model.transform(data[120:])

    assert model.embedding_.shape == (120, 2)
    assert transformed.shape == (30, 2)
    assert np.isfinite(model.embedding_).all()
    assert np.isfinite(transformed).all()
    assert capsys.readouterr().out == ""


@pytest.mark.parametrize("optimizer", MODERN_OPTIMIZERS)
def test_modern_generic_epoch_snapshots(optimizer):
    data, _ = make_blobs(
        n_samples=80,
        n_features=5,
        centers=4,
        random_state=42,
    )
    model = generic_model(optimizer, n_epochs=[5, 10]).fit(data)

    assert len(model.embedding_list_) == 2
    assert all(
        embedding.shape == (data.shape[0], 2) for embedding in model.embedding_list_
    )
    assert all(np.isfinite(embedding).all() for embedding in model.embedding_list_)


@pytest.mark.parametrize("optimizer", MODERN_OPTIMIZERS)
def test_modern_generic_force_ranked_negatives_are_finite(optimizer):
    data, _ = make_blobs(
        n_samples=100,
        n_features=5,
        centers=4,
        random_state=42,
    )

    embedding = generic_model(
        optimizer, negative_selection_range=data.shape[0] // 4
    ).fit_transform(data)

    assert embedding.shape == (data.shape[0], 2)
    assert np.isfinite(embedding).all()
