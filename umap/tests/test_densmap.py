import numpy as np
from umap import UMAP
import pytest

try:
    # works for sklearn>=0.22
    from sklearn.manifold import trustworthiness
except ImportError:
    # this is to comply with requirements (scikit-learn>=0.20)
    # More recent versions of sklearn have exposed trustworthiness
    # in top level module API
    # see: https://github.com/scikit-learn/scikit-learn/pull/15337
    from sklearn.manifold.t_sne import trustworthiness


def test_densmap_trustworthiness(nn_data):
    data = nn_data[:50]
    embedding, rad_h, rad_l = UMAP(
        n_neighbors=10,
        min_dist=0.01,
        random_state=42,
        n_epochs=200,
        densmap=True,
        output_dens=True,
    ).fit_transform(data)
    trust = trustworthiness(data, embedding, n_neighbors=10)
    assert (
        trust >= 0.7
    ), "Insufficiently trustworthy embedding for" "nn dataset: {}".format(trust)


def test_densmap_trustworthiness_adam(nn_data):
    data = nn_data[:50]
    embedding, rad_h, rad_l = UMAP(
        n_neighbors=10,
        min_dist=0.01,
        random_state=42,
        n_epochs=200,
        densmap=True,
        output_dens=True,
        optimizer="adam",
    ).fit_transform(data)
    trust = trustworthiness(data, embedding, n_neighbors=10)
    assert (
        trust >= 0.7
    ), "Insufficiently trustworthy embedding for" "nn dataset: {}".format(trust)


def test_densmap_trustworthiness_momentum(nn_data):
    data = nn_data[:50]
    embedding, rad_h, rad_l = UMAP(
        n_neighbors=10,
        min_dist=0.01,
        random_state=42,
        n_epochs=200,
        densmap=True,
        output_dens=True,
        optimizer="momentum",
    ).fit_transform(data)
    trust = trustworthiness(data, embedding, n_neighbors=10)
    assert (
        trust >= 0.7
    ), "Insufficiently trustworthy embedding for nn dataset: {}".format(trust)


@pytest.mark.parametrize(
    "compatibility_option",
    [{"compatibility_layout": True}, {"optimizer": "compatibility"}],
)
def test_densmap_trustworthiness_compatibility(nn_data, compatibility_option):
    data = nn_data[:50]
    embedding, rad_h, rad_l = UMAP(
        n_neighbors=10,
        min_dist=0.01,
        random_state=42,
        n_epochs=200,
        densmap=True,
        output_dens=True,
        **compatibility_option,
    ).fit_transform(data)
    trust = trustworthiness(data, embedding, n_neighbors=10)
    assert (
        trust >= 0.72
    ), "Insufficiently trustworthy embedding for" "nn dataset: {}".format(trust)


@pytest.mark.skip()
def test_densmap_trustworthiness_random_init(nn_data):  # pragma: no cover
    data = nn_data[:50]
    embedding = UMAP(
        n_neighbors=10,
        min_dist=0.01,
        random_state=42,
        init="random",
        densmap=True,
    ).fit_transform(data)
    trust = trustworthiness(data, embedding, n_neighbors=10)
    assert (
        trust >= 0.75
    ), "Insufficiently trustworthy embedding for" "nn dataset: {}".format(trust)


def test_densmap_trustworthiness_on_iris(iris):
    densmap_iris_model = UMAP(
        n_neighbors=10,
        min_dist=0.01,
        random_state=42,
        densmap=True,
        verbose=True,
    ).fit(iris.data)
    embedding = densmap_iris_model.embedding_
    trust = trustworthiness(iris.data, embedding, n_neighbors=10)
    assert (
        trust >= 0.97
    ), "Insufficiently trustworthy embedding for" "iris dataset: {}".format(trust)

    with pytest.raises(NotImplementedError):
        densmap_iris_model.transform(iris.data[:10])

    with pytest.raises(ValueError):
        densmap_iris_model.inverse_transform(embedding[:10])


def test_densmap_trustworthiness_on_iris_supervised(iris):
    densmap_iris_model = UMAP(
        n_neighbors=10,
        min_dist=0.01,
        random_state=42,
        densmap=True,
        verbose=True,
    ).fit(iris.data, y=iris.target)
    embedding = densmap_iris_model.embedding_
    trust = trustworthiness(iris.data, embedding, n_neighbors=10)
    assert (
        trust >= 0.97
    ), "Insufficiently trustworthy embedding for" "iris dataset: {}".format(trust)


def _log_local_radius(embedding, n_neighbors):
    from sklearn.neighbors import NearestNeighbors

    distances, _ = (
        NearestNeighbors(n_neighbors=n_neighbors).fit(embedding).kneighbors(embedding)
    )
    return np.log(np.mean(distances[:, 1:] ** 2, axis=1) + 1e-8)


@pytest.mark.parametrize("compatibility_layout", [True, False])
def test_output_dens_embedding_radii_match_embedding(iris, compatibility_layout):
    model = UMAP(
        densmap=True,
        output_dens=True,
        n_epochs=100,
        random_state=42,
        compatibility_layout=compatibility_layout,
    ).fit(iris.data)

    assert model.rad_orig_.shape == model.rad_emb_.shape == (iris.data.shape[0],)
    assert np.all(np.isfinite(model.rad_emb_))
    # rad_emb_ must describe local density in the returned embedding.
    independent = _log_local_radius(model.embedding_, 30)
    assert np.corrcoef(model.rad_emb_, independent)[0, 1] > 0.8
    # densMAP preserves the original local radii in the embedding.
    assert np.corrcoef(model.rad_orig_, model.rad_emb_)[0, 1] > 0.5
