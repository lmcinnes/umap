import numpy as np
from numpy.testing import assert_allclose, assert_array_equal
import pytest

from umap import UMAP
import umap.distances as dist


def _data(dtype=np.float64):
    return np.array([[0, 2, 5], [6, 1, 0], [2, 4, 3], [1, 0, 8]], dtype=dtype)


def _expected(X, Y, metric, value):
    if metric == "minkowski":
        return np.sum(np.abs(X[:, None] - Y) ** value, axis=-1) ** (1 / value)
    p = (X + value) / (X + value).sum(axis=1, keepdims=True)
    q = (Y + value) / (Y + value).sum(axis=1, keepdims=True)
    return np.sum((p[:, None] - q) * np.log(p[:, None] / q), axis=-1) / 2


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("rectangular", [False, True])
@pytest.mark.parametrize(
    "metric,kwds,value",
    [("minkowski", {"p": 1.0}, 1.0), ("symmetric_kl", {"z": 0.5}, 0.5)],
)
def test_named_metric_kwds(dtype, rectangular, metric, kwds, value):
    X = _data(dtype)
    Y = X[:2].copy() if rectangular else None
    before = X.copy()
    expected = _expected(X, Y if rectangular else X, metric, value)

    result = dist.pairwise_special_metric(X, Y, metric=metric, kwds=kwds)

    assert_allclose(result, expected, rtol=2e-6, atol=2e-6)
    assert_array_equal(X, before)
    if rectangular:
        assert_array_equal(Y, before[:2])


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("rectangular", [False, True])
def test_named_metric_multiple_kwds(dtype, rectangular):
    X = _data(dtype)
    Y = X[:2].copy() if rectangular else None
    w = np.array([0.25, 2.0, 1.0], dtype=dtype)
    differences = np.abs(X[:, None] - (Y if rectangular else X))
    expected = np.sum(w * differences**3, axis=-1) ** (1 / 3)

    result = dist.pairwise_special_metric(
        X, Y, metric="wminkowski", kwds={"w": w, "p": 3.0}
    )

    assert_allclose(result, expected, rtol=2e-6, atol=2e-6)


@pytest.mark.parametrize("rectangular", [False, True])
@pytest.mark.parametrize("kwds", [None, {}])
@pytest.mark.parametrize("metric,value", [("minkowski", 2.0), ("symmetric_kl", 1e-11)])
def test_named_metric_defaults_unchanged(rectangular, kwds, metric, value):
    X = _data()
    Y = X[:2].copy() if rectangular else None

    result = dist.pairwise_special_metric(X, Y, metric=metric, kwds=kwds)

    assert_allclose(
        result, _expected(X, Y if rectangular else X, metric, value), atol=2e-6
    )


@pytest.mark.parametrize("rectangular", [False, True])
@pytest.mark.parametrize(
    "metric,kwds,value",
    [("minkowski", {"p": 1.0}, 1.0), ("symmetric_kl", {"z": 0.5}, 0.5)],
)
def test_callable_metric_kwds_unchanged(rectangular, metric, kwds, value):
    X = _data()
    Y = X[:2].copy() if rectangular else None

    result = dist.pairwise_special_metric(
        X, Y, metric=dist.named_distances[metric], kwds=kwds
    )

    assert_allclose(
        result, _expected(X, Y if rectangular else X, metric, value), atol=2e-6
    )


@pytest.mark.parametrize("smoothing", [0.5, 2.0])
def test_supervised_target_metric_kwds(smoothing):
    X = np.array(
        [[0, 1], [1, 2], [2, 1], [3, 5], [4, 3], [5, 4], [6, 5], [7, 3]],
        dtype=np.float32,
    )
    y = np.vstack([_data(), [[4, 3, 0], [1, 2, 6], [5, 0, 2], [2, 7, 1]]]).astype(
        np.float32
    )
    options = dict(
        n_neighbors=3,
        n_epochs=20,
        init="random",
        random_state=42,
        n_jobs=1,
        target_metric_kwds={"z": smoothing},
    )

    named = UMAP(target_metric="symmetric_kl", **options).fit(X, y=y)
    callable_model = UMAP(target_metric=dist.symmetric_kl, **options).fit(X, y=y)

    assert_allclose(named.graph_.toarray(), callable_model.graph_.toarray(), atol=1e-6)
