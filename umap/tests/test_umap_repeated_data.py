import numpy as np
from scipy import sparse
from umap import UMAP
from umap.utils import csr_unique


# ===================================================
#  Spatial Data Test cases
# ===================================================
#  Use force_approximation_algorithm in order to test
#  the region of the code that is called for n>4096
# ---------------------------------------------------


def test_repeated_points_large_sparse_spatial(sparse_spatial_data_repeats):
    model = UMAP(
        n_neighbors=3,
        unique=True,
        force_approximation_algorithm=True,
        n_epochs=20,
        verbose=True,
    ).fit(sparse_spatial_data_repeats)
    assert np.unique(model.embedding_[0:2], axis=0).shape[0] == 1


def test_repeated_points_small_sparse_spatial(sparse_spatial_data_repeats):
    model = UMAP(n_neighbors=3, unique=True, n_epochs=20).fit(
        sparse_spatial_data_repeats
    )
    assert np.unique(model.embedding_[0:2], axis=0).shape[0] == 1


# Use force_approximation_algorithm in order to test the region
# of the code that is called for n>4096
def test_repeated_points_large_dense_spatial(spatial_repeats):
    model = UMAP(
        n_neighbors=3, unique=True, force_approximation_algorithm=True, n_epochs=50
    ).fit(spatial_repeats)
    assert np.unique(model.embedding_[0:2], axis=0).shape[0] == 1


def test_repeated_points_small_dense_spatial(spatial_repeats):
    model = UMAP(n_neighbors=3, unique=True, n_epochs=20).fit(spatial_repeats)
    assert np.unique(model.embedding_[0:2], axis=0).shape[0] == 1


# ===================================================
#  Binary Data Test cases
# ===================================================
# Use force_approximation_algorithm in order to test
# the region of the code that is called for n>4096
# ---------------------------------------------------


def test_repeated_points_large_sparse_binary(sparse_binary_data_repeats):
    model = UMAP(
        n_neighbors=3, unique=True, force_approximation_algorithm=True, n_epochs=50
    ).fit(sparse_binary_data_repeats)
    assert np.unique(model.embedding_[0:2], axis=0).shape[0] == 1


def test_repeated_points_small_sparse_binary(sparse_binary_data_repeats):
    model = UMAP(n_neighbors=3, unique=True, n_epochs=20).fit(
        sparse_binary_data_repeats
    )
    assert np.unique(model.embedding_[0:2], axis=0).shape[0] == 1


# Every row has the same number of non-zeros, so csr_unique must
# still compare whole rows rather than individual values.
def test_repeated_points_sparse_equal_nnz_rows():
    data = np.eye(12)[np.random.RandomState(0).randint(0, 12, 60)]
    index, inverse, counts = csr_unique(sparse.csr_matrix(data))
    assert index.shape[0] == np.unique(data, axis=0).shape[0]
    assert np.array_equal(data[index][inverse], data)
    assert counts.sum() == data.shape[0]

    model = UMAP(n_neighbors=3, unique=True, n_epochs=20).fit(sparse.csr_matrix(data))
    assert model.embedding_.shape == (60, 2)
    assert np.array_equal(model.embedding_[index][inverse], model.embedding_)


# Use force_approximation_algorithm in order to test
# the region of the code that is called for n>4096
def test_repeated_points_large_dense_binary(binary_repeats):
    model = UMAP(
        n_neighbors=3, unique=True, force_approximation_algorithm=True, n_epochs=20
    ).fit(binary_repeats)
    assert np.unique(model.embedding_[0:2], axis=0).shape[0] == 1


def test_repeated_points_small_dense_binary(binary_repeats):
    model = UMAP(n_neighbors=3, unique=True, n_epochs=20).fit(binary_repeats)
    assert np.unique(binary_repeats[0:2], axis=0).shape[0] == 1
    assert np.unique(model.embedding_[0:2], axis=0).shape[0] == 1


# ===================================================
#  Repeated Data Test cases
# ===================================================


# ----------------------------------------------------
# This should test whether the n_neighbours are being
# reduced properly when your n_neighbours is larger
# than the unique data set size
# ----------------------------------------------------
def test_repeated_points_large_n(repetition_dense):
    model = UMAP(n_neighbors=5, unique=True, n_epochs=20).fit(repetition_dense)
    assert model._n_neighbors == 3
