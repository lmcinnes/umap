import numpy as np
import pytest
from numpy.testing import assert_allclose
from sklearn.datasets import make_blobs

from umap import UMAP

OPTIMIZERS = ("adam", "momentum")


def small_model(optimizer, exclude_graph_neighbors=False, **kwargs):
    parameters = dict(
        n_neighbors=10,
        n_epochs=20,
        init="random",
        optimizer=optimizer,
        random_state=42,
        exclude_graph_neighbors=exclude_graph_neighbors,
        negative_sample_scale_adaptation_samples=0,
    )
    parameters.update(kwargs)
    return UMAP(**parameters)


@pytest.mark.parametrize("optimizer", OPTIMIZERS)
@pytest.mark.parametrize("exclude_graph_neighbors", [False, True])
def test_small_kernel_fit_and_asymmetric_transform(optimizer, exclude_graph_neighbors):
    data, _ = make_blobs(
        n_samples=120,
        n_features=6,
        centers=4,
        random_state=42,
    )
    model = small_model(
        optimizer,
        exclude_graph_neighbors=exclude_graph_neighbors,
    ).fit(data[:90])

    transformed = model.transform(data[90:])

    assert model.embedding_.shape == (90, 2)
    assert transformed.shape == (30, 2)
    assert np.isfinite(model.embedding_).all()
    assert np.isfinite(transformed).all()


@pytest.mark.parametrize("optimizer", OPTIMIZERS)
def test_small_kernel_is_reproducible(optimizer):
    data, _ = make_blobs(
        n_samples=100,
        n_features=5,
        centers=4,
        random_state=42,
    )

    first = small_model(optimizer).fit_transform(data)
    second = small_model(optimizer).fit_transform(data)

    assert_allclose(first, second, rtol=0.0, atol=0.0)


@pytest.mark.parametrize("optimizer", OPTIMIZERS)
def test_small_kernel_exclusion_terminates_for_complete_graph(optimizer):
    data, _ = make_blobs(
        n_samples=12,
        n_features=4,
        centers=1,
        random_state=42,
    )
    embedding = small_model(
        optimizer,
        exclude_graph_neighbors=True,
        n_neighbors=data.shape[0] - 1,
        n_epochs=5,
    ).fit_transform(data)

    assert embedding.shape == (data.shape[0], 2)
    assert np.isfinite(embedding).all()


def test_graph_neighbor_exclusion_defaults_off():
    assert UMAP().exclude_graph_neighbors is False
