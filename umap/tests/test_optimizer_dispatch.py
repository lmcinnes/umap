"""Cross-cutting verification for optimizer dispatch boundaries.

These tests fill combinations not exercised by the focused kernel suites. Some
matrix dimensions are intentionally not parametrized: inverse transform has a
dedicated optimizer-independent objective, aligned UMAP has one established
optimizer, and DensMAP is only implemented for Euclidean output embeddings.
"""

import numpy as np
import pytest
import scipy.sparse
from sklearn.datasets import make_blobs

from umap import UMAP
import umap.umap_ as umap_module

MODERN_OPTIMIZERS = ("adam", "momentum")


def sparse_blobs(n_samples):
    data, _ = make_blobs(
        n_samples=n_samples,
        n_features=24,
        centers=6,
        random_state=42,
    )
    data[np.abs(data) < 4.0] = 0.0
    return scipy.sparse.csr_matrix(data.astype(np.float32))


@pytest.mark.parametrize("optimizer", MODERN_OPTIMIZERS)
def test_modern_densmap_memberships_follow_csr_order(monkeypatch, optimizer):
    captured = {}

    def capture_layout(head_embedding, *args, **kwargs):
        captured["memberships"] = kwargs["densmap_kwds"]["mu"].copy()
        captured["csr_data"] = kwargs["csr_data"].copy()
        return head_embedding

    monkeypatch.setattr(umap_module, "optimize_layout_euclidean", capture_layout)
    data, _ = make_blobs(
        n_samples=80,
        n_features=6,
        centers=4,
        random_state=42,
    )

    UMAP(
        n_neighbors=10,
        n_epochs=20,
        init="random",
        optimizer=optimizer,
        densmap=True,
        random_state=42,
    ).fit(data)

    np.testing.assert_array_equal(captured["memberships"], captured["csr_data"])


@pytest.mark.parametrize("optimizer", MODERN_OPTIMIZERS)
def test_large_sparse_hard_negative_fit_and_transform(optimizer):
    data = sparse_blobs(1050)
    model = UMAP(
        n_neighbors=10,
        n_epochs=10,
        init="random",
        optimizer=optimizer,
        negative_selection_range=250,
        negative_sample_scale_adaptation_samples=0,
        exclude_graph_neighbors=True,
        random_state=42,
    ).fit(data)

    transformed = model.transform(data[:25])

    assert model.embedding_.shape == (1050, 2)
    assert transformed.shape == (25, 2)
    assert np.isfinite(model.embedding_).all()
    assert np.isfinite(transformed).all()


@pytest.mark.parametrize("optimizer", MODERN_OPTIMIZERS)
def test_large_generic_fit_and_transform(optimizer):
    data, _ = make_blobs(
        n_samples=1060,
        n_features=8,
        centers=6,
        random_state=42,
    )
    model = UMAP(
        n_neighbors=10,
        n_epochs=10,
        init="random",
        output_metric="haversine",
        optimizer=optimizer,
        negative_selection_range=265,
        random_state=42,
    ).fit(data[:1030])

    transformed = model.transform(data[1030:])

    assert model.embedding_.shape == (1030, 2)
    assert transformed.shape == (30, 2)
    assert np.isfinite(model.embedding_).all()
    assert np.isfinite(transformed).all()


@pytest.mark.parametrize("optimizer", MODERN_OPTIMIZERS)
def test_large_densmap_fit_produces_finite_density_radii(optimizer):
    data, _ = make_blobs(
        n_samples=1030,
        n_features=8,
        centers=6,
        cluster_std=[0.2, 0.4, 0.8, 1.2, 1.8, 2.5],
        random_state=42,
    )
    model = UMAP(
        n_neighbors=10,
        n_epochs=10,
        init="random",
        optimizer=optimizer,
        densmap=True,
        output_dens=True,
        random_state=42,
    ).fit(data)

    assert model.embedding_.shape == (1030, 2)
    assert np.isfinite(model.embedding_).all()
    assert np.isfinite(model.rad_orig_).all()
    assert np.isfinite(model.rad_emb_).all()


@pytest.mark.parametrize("optimizer", MODERN_OPTIMIZERS)
def test_unseeded_parallel_fit_is_finite(optimizer):
    data, _ = make_blobs(
        n_samples=180,
        n_features=6,
        centers=4,
        random_state=42,
    )
    embedding = UMAP(
        n_neighbors=10,
        n_epochs=10,
        init="random",
        optimizer=optimizer,
        random_state=None,
        n_jobs=2,
    ).fit_transform(data)

    assert embedding.shape == (180, 2)
    assert np.isfinite(embedding).all()


def test_sparse_compatibility_fit_and_transform():
    data = sparse_blobs(180)
    model = UMAP(
        n_neighbors=10,
        n_epochs=15,
        init="random",
        optimizer="compatibility",
        random_state=42,
    ).fit(data[:150])

    transformed = model.transform(data[150:])

    assert model.embedding_.shape == (150, 2)
    assert transformed.shape == (30, 2)
    assert np.isfinite(model.embedding_).all()
    assert np.isfinite(transformed).all()
