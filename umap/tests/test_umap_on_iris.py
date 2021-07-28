from umap import UMAP
from scipy import sparse
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
from sklearn.neighbors import KDTree
from scipy.spatial.distance import cdist, pdist, squareform

try:
    # works for sklearn>=0.22
    from sklearn.manifold import trustworthiness
except ImportError:
    # this is to comply with requirements (scikit-learn>=0.20)
    # More recent versions of sklearn have exposed trustworthiness
    # in top level module API
    # see: https://github.com/scikit-learn/scikit-learn/pull/15337
    from sklearn.manifold.t_sne import trustworthiness

# ===================================================
#  UMAP Test cases on IRIS Dataset
# ===================================================

# UMAP Trustworthiness on iris
# ----------------------------
def test_umap_trustworthiness_on_iris(iris, iris_model):
    embedding = iris_model.embedding_
    trust = trustworthiness(iris.data, embedding, 10)
    assert (
        trust >= 0.97
    ), "Insufficiently trustworthy embedding for" "iris dataset: {}".format(trust)


def test_initialized_umap_trustworthiness_on_iris(iris):
    data = iris.data
    embedding = UMAP(
        n_neighbors=10,
        min_dist=0.01,
        init=data[:, 2:],
        n_epochs=200,
        random_state=42,
    ).fit_transform(data)
    trust = trustworthiness(iris.data, embedding, 10)
    assert (
        trust >= 0.97
    ), "Insufficiently trustworthy embedding for" "iris dataset: {}".format(trust)


def test_umap_trustworthiness_on_sphere_iris(
    iris,
):
    data = iris.data
    embedding = UMAP(
        n_neighbors=10,
        min_dist=0.01,
        n_epochs=200,
        random_state=42,
        output_metric="haversine",
    ).fit_transform(data)
    # Since trustworthiness doesn't support haversine, project onto
    # a 3D embedding of the sphere and use cosine distance
    r = 3
    projected_embedding = np.vstack(
        [
            r * np.sin(embedding[:, 0]) * np.cos(embedding[:, 1]),
            r * np.sin(embedding[:, 0]) * np.sin(embedding[:, 1]),
            r * np.cos(embedding[:, 0]),
        ]
    ).T
    trust = trustworthiness(iris.data, projected_embedding, 10, metric="cosine")
    assert (
        trust >= 0.80
    ), "Insufficiently trustworthy spherical embedding for iris dataset: {}".format(
        trust
    )


# UMAP Transform on iris
# ----------------------
def test_umap_transform_on_iris(iris, iris_subset_model, iris_selection):
    fitter = iris_subset_model

    new_data = iris.data[~iris_selection]
    embedding = fitter.transform(new_data)

    trust = trustworthiness(new_data, embedding, 10)
    assert (
        trust >= 0.85
    ), "Insufficiently trustworthy transform for" "iris dataset: {}".format(trust)


def test_umap_transform_on_iris_w_pynndescent(iris, iris_selection):
    data = iris.data[iris_selection]
    fitter = UMAP(
        n_neighbors=10,
        min_dist=0.01,
        n_epochs=100,
        random_state=42,
        force_approximation_algorithm=True,
    ).fit(data)

    new_data = iris.data[~iris_selection]
    embedding = fitter.transform(new_data)

    trust = trustworthiness(new_data, embedding, 10)
    assert (
        trust >= 0.85
    ), "Insufficiently trustworthy transform for" "iris dataset: {}".format(trust)


def test_umap_transform_on_iris_modified_dtype(iris, iris_subset_model, iris_selection):
    fitter = iris_subset_model
    fitter.embedding_ = fitter.embedding_.astype(np.float64)

    new_data = iris.data[~iris_selection]
    embedding = fitter.transform(new_data)

    trust = trustworthiness(new_data, embedding, 10)
    assert (
        trust >= 0.8
    ), "Insufficiently trustworthy transform for iris dataset: {}".format(trust)


def test_umap_sparse_transform_on_iris(iris, iris_selection):
    data = sparse.csr_matrix(iris.data[iris_selection])
    assert sparse.issparse(data)
    fitter = UMAP(
        n_neighbors=10,
        min_dist=0.01,
        random_state=42,
        n_epochs=100,
        # force_approximation_algorithm=True,
    ).fit(data)

    new_data = sparse.csr_matrix(iris.data[~iris_selection])
    assert sparse.issparse(new_data)
    embedding = fitter.transform(new_data)

    trust = trustworthiness(new_data, embedding, 10)
    assert (
        trust >= 0.80
    ), "Insufficiently trustworthy transform for" "iris dataset: {}".format(trust)


# UMAP precomputed metric transform on iris
# ----------------------
def test_precomputed_transform_on_iris(iris, iris_selection):
    data = iris.data[iris_selection]
    distance_matrix = squareform(pdist(data))

    fitter = UMAP(
        n_neighbors=10,
        min_dist=0.01,
        random_state=42,
        n_epochs=100,
        metric="precomputed",
    ).fit(distance_matrix)

    new_data = iris.data[~iris_selection]
    new_distance_matrix = cdist(new_data, data)
    embedding = fitter.transform(new_distance_matrix)

    trust = trustworthiness(new_data, embedding, 10)
    assert (
        trust >= 0.85
    ), "Insufficiently trustworthy transform for" "iris dataset: {}".format(trust)


# UMAP precomputed metric transform on iris with sparse distances
# ----------------------
def test_precomputed_sparse_transform_on_iris(iris, iris_selection):
    data = iris.data[iris_selection]
    distance_matrix = sparse.csr_matrix(squareform(pdist(data)))

    fitter = UMAP(
        n_neighbors=10,
        min_dist=0.01,
        random_state=42,
        n_epochs=100,
        metric="precomputed",
    ).fit(distance_matrix)

    new_data = iris.data[~iris_selection]
    new_distance_matrix = sparse.csr_matrix(cdist(new_data, data))
    embedding = fitter.transform(new_distance_matrix)

    trust = trustworthiness(new_data, embedding, 10)
    assert (
        trust >= 0.85
    ), "Insufficiently trustworthy transform for" "iris dataset: {}".format(trust)


# UMAP Clusterability on Iris
# ---------------------------
def test_umap_clusterability_on_supervised_iris(supervised_iris_model, iris):
    embedding = supervised_iris_model.embedding_
    clusters = KMeans(3).fit_predict(embedding)
    assert adjusted_rand_score(clusters, iris.target) >= 0.95


# UMAP Inverse transform on Iris
# ------------------------------
def test_umap_inverse_transform_on_iris(iris, iris_model):
    highd_tree = KDTree(iris.data)
    fitter = iris_model
    lowd_tree = KDTree(fitter.embedding_)
    for i in range(1, 150, 20):
        query_point = fitter.embedding_[i]
        near_points = lowd_tree.query([query_point], k=5, return_distance=False)
        centroid = np.mean(np.squeeze(fitter.embedding_[near_points]), axis=0)
        highd_centroid = fitter.inverse_transform([centroid])
        highd_near_points = highd_tree.query(
            highd_centroid, k=10, return_distance=False
        )
        assert np.intersect1d(near_points, highd_near_points[0]).shape[0] >= 3
