import numpy as np
import pytest
from sklearn.datasets import make_blobs

from umap import UMAP


@pytest.fixture(scope="module")
def inverse_model():
    data, _ = make_blobs(
        n_samples=120,
        n_features=4,
        centers=4,
        random_state=42,
    )
    return UMAP(n_epochs=30, random_state=42, transform_seed=42).fit(data)


def test_inverse_transform_is_reproducible(inverse_model):
    query = np.mean(inverse_model.embedding_[:5], axis=0, keepdims=True)

    first = inverse_model.inverse_transform(query)
    second = inverse_model.inverse_transform(query)

    np.testing.assert_allclose(first, second)
    assert np.isfinite(first).all()


def test_inverse_transform_outside_hull_uses_finite_extrapolation(inverse_model):
    embedding = inverse_model.embedding_
    query = np.max(embedding, axis=0) + 10.0 * np.ptp(embedding, axis=0)

    with pytest.warns(UserWarning, match="outside the embedding convex hull"):
        result = inverse_model.inverse_transform(query.reshape(1, -1))

    assert result.shape == (1, inverse_model._raw_data.shape[1])
    assert np.isfinite(result).all()


def test_inverse_transform_rejects_wrong_embedding_dimension(inverse_model):
    query = np.zeros((1, inverse_model.embedding_.shape[1] + 1), dtype=np.float32)

    with pytest.raises(ValueError, match="same number of dimensions"):
        inverse_model.inverse_transform(query)
