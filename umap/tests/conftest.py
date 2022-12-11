# ===========================
#  Testing (session) Fixture
# ==========================

import pytest
import numpy as np
from scipy import sparse
from sklearn.datasets import load_iris
from umap import UMAP, AlignedUMAP

# Globals, used for all the tests
SEED = 189212  # 0b101110001100011100
np.random.seed(SEED)


# Spatial and Binary Data
# -----------------------
@pytest.fixture(scope="session")
def spatial_data():
    # - Spatial Data
    spatial_data = np.random.randn(10, 20)
    # Add some all zero data for corner case test
    return np.vstack([spatial_data, np.zeros((2, 20))])


@pytest.fixture(scope="session")
def binary_data():
    binary_data = np.random.choice(a=[False, True], size=(10, 20), p=[0.66, 1 - 0.66])
    # Add some all zero data for corner case test
    binary_data = np.vstack([binary_data, np.zeros((2, 20), dtype="bool")])
    return binary_data


# Sparse Spatial and Binary Data
# ------------------------------
@pytest.fixture(scope="session")
def sparse_spatial_data(spatial_data, binary_data):
    return sparse.csr_matrix(spatial_data * binary_data)


@pytest.fixture(scope="session")
def sparse_binary_data(binary_data):
    return sparse.csr_matrix(binary_data)


# Nearest Neighbour Data
# -----------------------
@pytest.fixture(scope="session")
def nn_data():
    nn_data = np.random.uniform(0, 1, size=(1000, 5))
    nn_data = np.vstack(
        [nn_data, np.zeros((2, 5))]
    )  # Add some all zero data for corner case test
    return nn_data


@pytest.fixture(scope="session")
def binary_nn_data():
    binary_nn_data = np.random.choice(
        a=[False, True], size=(1000, 5), p=[0.66, 1 - 0.66]
    )
    binary_nn_data = np.vstack(
        [binary_nn_data, np.zeros((2, 5), dtype="bool")]
    )  # Add some all zero data for corner case test
    return binary_nn_data


@pytest.fixture(scope="session")
def sparse_nn_data():
    return sparse.random(1000, 50, density=0.5, format="csr")


# Data With Repetitions
# ---------------------


@pytest.fixture(scope="session")
def repetition_dense():
    # Dense data for testing small n
    return np.array(
        [
            [5, 6, 7, 8],
            [5, 6, 7, 8],
            [5, 6, 7, 8],
            [5, 6, 7, 8],
            [5, 6, 7, 8],
            [5, 6, 7, 8],
            [1, 1, 1, 1],
            [1, 2, 3, 4],
            [1, 1, 2, 1],
        ]
    )


@pytest.fixture(scope="session")
def spatial_repeats(spatial_data):
    # spatial data repeats
    spatial_repeats = np.vstack(
        [np.repeat(spatial_data[0:2], [2, 0], axis=0), spatial_data, np.zeros((2, 20))]
    )
    # Add some all zero data for corner case test.  Make the first three rows identical
    # binary Data Repeat
    return spatial_repeats


@pytest.fixture(scope="session")
def binary_repeats(binary_data):
    binary_repeats = np.vstack(
        [
            np.repeat(binary_data[0:2], [2, 0], axis=0),
            binary_data,
            np.zeros((2, 20), dtype="bool"),
        ]
    )
    # Add some all zero data for corner case test.  Make the first three rows identical
    return binary_repeats


@pytest.fixture(scope="session")
def sparse_spatial_data_repeats(spatial_repeats, binary_repeats):
    return sparse.csr_matrix(spatial_repeats * binary_repeats)


@pytest.fixture(scope="session")
def sparse_binary_data_repeats(binary_repeats):
    return sparse.csr_matrix(binary_repeats)


@pytest.fixture(scope="session")
def sparse_test_data(nn_data, binary_nn_data):
    return sparse.csr_matrix(nn_data * binary_nn_data)


@pytest.fixture(scope="session")
def iris():
    return load_iris()


@pytest.fixture(scope="session")
def iris_selection():
    return np.random.choice([True, False], 150, replace=True, p=[0.75, 0.25])


@pytest.fixture(scope="session")
def aligned_iris(iris):
    slices = [iris.data[i : i + 50] for i in range(0, 125, 25)]
    target = [iris.target[i : i + 50] for i in range(0, 125, 25)]
    return slices, target


@pytest.fixture(scope="session")
def aligned_iris_relations():
    return [{a: a + 25 for a in range(25)} for i in range(4)]


@pytest.fixture(scope="session")
def iris_model(iris):
    return UMAP(n_neighbors=10, min_dist=0.01, random_state=42).fit(iris.data)


@pytest.fixture(scope="session")
def iris_model_large(iris):
    return UMAP(
        n_neighbors=10,
        min_dist=0.01,
        random_state=42,
        force_approximation_algorithm=True,
    ).fit(iris.data)


@pytest.fixture(scope="session")
def iris_subset_model(iris, iris_selection):
    return UMAP(n_neighbors=10, min_dist=0.01, random_state=42).fit(
        iris.data[iris_selection]
    )


@pytest.fixture(scope="session")
def iris_subset_model_large(iris, iris_selection):
    return UMAP(
        n_neighbors=10,
        min_dist=0.01,
        random_state=42,
        force_approximation_algorithm=True,
    ).fit(iris.data[iris_selection])


@pytest.fixture(scope="session")
def supervised_iris_model(iris):
    return UMAP(n_neighbors=10, min_dist=0.01, n_epochs=200, random_state=42).fit(
        iris.data, iris.target
    )


@pytest.fixture(scope="session")
def aligned_iris_model(aligned_iris, aligned_iris_relations):
    data, target = aligned_iris
    model = AlignedUMAP()
    model.fit(data, relations=aligned_iris_relations)
    return model


# UMAP Distance Metrics
# ---------------------
@pytest.fixture(scope="session")
def spatial_distances():
    return (
        "euclidean",
        "manhattan",
        "chebyshev",
        "minkowski",
        "hamming",
        "canberra",
        "braycurtis",
        "cosine",
        "correlation",
    )


@pytest.fixture(scope="session")
def binary_distances():
    return (
        "jaccard",
        "matching",
        "dice",
        "kulsinski",
        "rogerstanimoto",
        "russellrao",
        "sokalmichener",
        "sokalsneath",
        "yule",
    )
