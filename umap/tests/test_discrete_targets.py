import numpy as np
import pytest

from umap import UMAP
from umap.distances import get_discrete_params, levenshtein, ordinal_distance


@pytest.mark.parametrize("dtype", [np.int32, np.float32, np.float64])
def test_constant_ordinal_target_distance(dtype):
    target = np.full(20, 7, dtype=dtype)
    params = get_discrete_params(target, "ordinal")
    assert ordinal_distance(target[0], target[-1], **params) == 0.0


@pytest.mark.parametrize("dtype", [np.int32, np.float32, np.float64])
def test_nonconstant_ordinal_target_scale(dtype):
    target = np.asarray([0, 1], dtype=dtype)
    params = get_discrete_params(target, "ordinal")
    assert params == {"support_size": 0.5}
    assert ordinal_distance(target[0], target[-1], **params) == 2.0


@pytest.mark.parametrize(
    "target_metric,dtype,value",
    [
        ("ordinal", np.int32, 7),
        ("ordinal", np.float32, 7),
        ("ordinal", np.float64, 7),
        ("string", str, ""),
    ],
)
def test_constant_discrete_targets_match_categories(target_metric, dtype, value):
    data = np.random.default_rng(472).normal(size=(20, 3)).astype(np.float32)
    target = np.full(len(data), value, dtype=dtype)
    options = dict(
        n_neighbors=5,
        n_epochs=10,
        init="random",
        random_state=42,
        n_jobs=1,
        transform_mode="graph",
    )
    categorical = UMAP(target_metric="categorical", **options).fit(
        data, np.zeros(len(data))
    )
    model = UMAP(target_metric=target_metric, **options).fit(data, target)
    assert np.isfinite(model.graph_.data).all()
    np.testing.assert_array_equal(model.graph_.toarray(), categorical.graph_.toarray())


def test_empty_string_target_distance():
    params = get_discrete_params(np.array(["", ""]), "string")
    assert levenshtein("", "", *params.values()) == 0.0


def test_nonempty_string_target_scale():
    params = get_discrete_params(np.array(["abc", "ab"]), "string")
    assert params == {"normalisation": 1.0, "max_dist": 1.0}
    assert levenshtein("abc", "ab", *params.values()) == 1.0
