from umap import UMAP
from sklearn.datasets import make_blobs
from nose.tools import assert_greater_equal, assert_raises
from nose import SkipTest
import pytest
import numpy as np

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
        n_epochs=100,
        densmap=True,
        output_dens=True,
    ).fit_transform(data)
    trust = trustworthiness(data, embedding, 10)
    assert_greater_equal(
        trust,
        0.75,
        "Insufficiently trustworthy embedding for" "nn dataset: {}".format(trust),
    )


@SkipTest
def test_densmap_trustworthiness_random_init(nn_data):  # pragma: no cover
    data = nn_data[:50]
    embedding = UMAP(
        n_neighbors=10, min_dist=0.01, random_state=42, init="random", densmap=True,
    ).fit_transform(data)
    trust = trustworthiness(data, embedding, 10)
    assert_greater_equal(
        trust,
        0.75,
        "Insufficiently trustworthy embedding for" "nn dataset: {}".format(trust),
    )


def test_densmap_trustworthiness_on_iris(iris):
    densmap_iris_model = UMAP(
        n_neighbors=10, min_dist=0.01, random_state=42, densmap=True, verbose=True,
    ).fit(iris.data)
    embedding = densmap_iris_model.embedding_
    trust = trustworthiness(iris.data, embedding, 10)
    assert_greater_equal(
        trust,
        0.97,
        "Insufficiently trustworthy embedding for" "iris dataset: {}".format(trust),
    )

    assert_raises(NotImplementedError, densmap_iris_model.transform, iris.data[:10])
    assert_raises(ValueError, densmap_iris_model.inverse_transform, embedding[:10])

    with pytest.raises(NotImplementedError):
        supervised_densmap_iris_model = UMAP(
            n_neighbors=10, min_dist=0.01, random_state=42, densmap=True, verbose=True,
        ).fit(iris.data, y=iris.target)
