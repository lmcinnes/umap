import numpy as np
import pytest
from scipy import sparse

from umap import UMAP


@pytest.mark.parametrize("as_sparse", [False, True])
@pytest.mark.parametrize("n_components", [2, 3])
@pytest.mark.parametrize("n_samples", [1, 6])
def test_fit_transform_single_unique_sample(as_sparse, n_components, n_samples):
    data = np.tile(np.array([[1.0, 2.0, 3.0]], dtype=np.float32), (n_samples, 1))
    if as_sparse:
        data = sparse.csr_matrix(data)

    model = UMAP(unique=True, n_components=n_components, n_jobs=1, random_state=42)
    embedding = model.fit_transform(data)

    assert embedding.shape == (n_samples, n_components)
    np.testing.assert_array_equal(embedding, np.zeros((n_samples, n_components)))
    np.testing.assert_array_equal(model.embedding_, embedding)
    np.testing.assert_array_equal(
        model._unique_inverse_, np.zeros(n_samples, dtype=int)
    )


@pytest.mark.parametrize("as_sparse", [False, True])
def test_fit_transform_single_sample_without_unique(as_sparse):
    data = np.array([[1.0, 2.0, 3.0]], dtype=np.float32)
    if as_sparse:
        data = sparse.csr_matrix(data)

    embedding = UMAP(unique=False, n_jobs=1, random_state=42).fit_transform(data)

    np.testing.assert_array_equal(embedding, np.zeros((1, 2)))
