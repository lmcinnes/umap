# ===============================
#  UMAP fit Parameters Validation
# ===============================

import warnings
import numpy as np
from sklearn.metrics import pairwise_distances
import pytest
import numba
from umap import UMAP

# verify that we can import this; potentially for later use
import umap.validation

warnings.filterwarnings("ignore", category=UserWarning)


def test_umap_negative_op(nn_data):
    u = UMAP(set_op_mix_ratio=-1.0)
    with pytest.raises(ValueError):
        u.fit(nn_data)


def test_umap_too_large_op(nn_data):
    u = UMAP(set_op_mix_ratio=1.5)
    with pytest.raises(ValueError):
        u.fit(nn_data)


def test_umap_bad_too_large_min_dist(nn_data):
    u = UMAP(min_dist=2.0)
    # a RuntimeWarning about division by zero in a,b curve fitting is expected
    # caught and ignored for this test
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        with pytest.raises(ValueError):
            u.fit(nn_data)


def test_umap_negative_min_dist(nn_data):
    u = UMAP(min_dist=-1)
    with pytest.raises(ValueError):
        u.fit(nn_data)


def test_umap_negative_n_components(nn_data):
    u = UMAP(n_components=-1)
    with pytest.raises(ValueError):
        u.fit(nn_data)


def test_umap_non_integer_n_components(nn_data):
    u = UMAP(n_components=1.5)
    with pytest.raises(ValueError):
        u.fit(nn_data)


def test_umap_too_small_n_neighbours(nn_data):
    u = UMAP(n_neighbors=0.5)
    with pytest.raises(ValueError):
        u.fit(nn_data)


def test_umap_negative_n_neighbours(nn_data):
    u = UMAP(n_neighbors=-1)
    with pytest.raises(ValueError):
        u.fit(nn_data)


def test_umap_bad_metric(nn_data):
    u = UMAP(metric=45)
    with pytest.raises(ValueError):
        u.fit(nn_data)


def test_umap_negative_learning_rate(nn_data):
    u = UMAP(learning_rate=-1.5)
    with pytest.raises(ValueError):
        u.fit(nn_data)


def test_umap_negative_repulsion(nn_data):
    u = UMAP(repulsion_strength=-0.5)
    with pytest.raises(ValueError):
        u.fit(nn_data)


def test_umap_negative_sample_rate(nn_data):
    u = UMAP(negative_sample_rate=-1)
    with pytest.raises(ValueError):
        u.fit(nn_data)


def test_umap_bad_init(nn_data):
    u = UMAP(init="foobar")
    with pytest.raises(ValueError):
        u.fit(nn_data)


def test_umap_bad_numeric_init(nn_data):
    u = UMAP(init=42)
    with pytest.raises(ValueError):
        u.fit(nn_data)


def test_umap_bad_matrix_init(nn_data):
    u = UMAP(init=np.array([[0, 0, 0], [0, 0, 0]]))
    with pytest.raises(ValueError):
        u.fit(nn_data)


def test_umap_negative_n_epochs(nn_data):
    u = UMAP(n_epochs=-2)
    with pytest.raises(ValueError):
        u.fit(nn_data)


def test_umap_negative_target_n_neighbours(nn_data):
    u = UMAP(target_n_neighbors=1)
    with pytest.raises(ValueError):
        u.fit(nn_data)


def test_umap_bad_output_metric(nn_data):
    u = UMAP(output_metric="foobar")
    with pytest.raises(ValueError):
        u.fit(nn_data)
    u = UMAP(output_metric="precomputed")
    with pytest.raises(ValueError):
        u.fit(nn_data)
    u = UMAP(output_metric="hamming")
    with pytest.raises(ValueError):
        u.fit(nn_data)


def test_haversine_on_highd(nn_data):
    u = UMAP(metric="haversine")
    with pytest.raises(ValueError):
        u.fit(nn_data)


def test_umap_haversine_embed_to_highd(nn_data):
    u = UMAP(n_components=3, output_metric="haversine")
    with pytest.raises(ValueError):
        u.fit(nn_data)


def test_umap_too_many_neighbors_warns(nn_data):
    u = UMAP(a=1.2, b=1.75, n_neighbors=2000, n_epochs=11, init="random")
    u.fit(
        nn_data[
            :100,
        ]
    )
    assert u._a == 1.2
    assert u._b == 1.75


def test_densmap_lambda(nn_data):
    u = UMAP(densmap=True, dens_lambda=-1.0)
    with pytest.raises(ValueError):
        u.fit(nn_data)


def test_densmap_var_shift(nn_data):
    u = UMAP(densmap=True, dens_var_shift=-1.0)
    with pytest.raises(ValueError):
        u.fit(nn_data)


def test_densmap_frac(nn_data):
    u = UMAP(densmap=True, dens_frac=-1.0)
    with pytest.raises(ValueError):
        u.fit(nn_data)
    u = UMAP(densmap=True, dens_frac=2.0)
    with pytest.raises(ValueError):
        u.fit(nn_data)


def test_umap_unique_and_precomputed(nn_data):
    u = UMAP(metric="precomputed", unique=True)
    with pytest.raises(ValueError):
        u.fit(nn_data)


def test_densmap_bad_output_metric(nn_data):
    u = UMAP(densmap=True, output_metric="haversine")
    with pytest.raises(ValueError):
        u.fit(nn_data)


def test_umap_bad_n_components(nn_data):
    u = UMAP(n_components=2.3)
    with pytest.raises(ValueError):
        u.fit(nn_data)
    u = UMAP(n_components="23")
    with pytest.raises(ValueError):
        u.fit(nn_data)
    u = UMAP(n_components=np.float64(2.3))
    with pytest.raises(ValueError):
        u.fit(nn_data)


def test_umap_bad_metrics(nn_data):
    u = UMAP(metric="foobar")
    with pytest.raises(ValueError):
        u.fit(nn_data)
    u = UMAP(metric=2.75)
    with pytest.raises(ValueError):
        u.fit(nn_data)
    u = UMAP(output_metric="foobar")
    with pytest.raises(ValueError):
        u.fit(nn_data)
    u = UMAP(output_metric=2.75)
    with pytest.raises(ValueError):
        u.fit(nn_data)
    # u = UMAP(target_metric="foobar")
    # assert_raises(ValueError, u.fit, nn_data)
    # u = UMAP(target_metric=2.75)
    # assert_raises(ValueError, u.fit, nn_data)


def test_umap_bad_n_jobs(nn_data):
    u = UMAP(n_jobs=-2)
    with pytest.raises(ValueError):
        u.fit(nn_data)
    u = UMAP(n_jobs=0)
    with pytest.raises(ValueError):
        u.fit(nn_data)


def test_umap_custom_distance_w_grad(nn_data):
    @numba.njit()
    def dist1(x, y):
        return np.sum(np.abs(x - y))

    @numba.njit()
    def dist2(x, y):
        return np.sum(np.abs(x - y)), (x - y)

    u = UMAP(metric=dist1, n_epochs=11)
    with pytest.warns(UserWarning) as warnings:
        u.fit(nn_data[:10])
    assert len(warnings) >= 1

    u = UMAP(metric=dist2, n_epochs=11)
    with pytest.warns(UserWarning) as warnings:
        u.fit(nn_data[:10])
    assert len(warnings) <= 1


def test_umap_bad_output_metric_no_grad(nn_data):
    @numba.njit()
    def dist1(x, y):
        return np.sum(np.abs(x - y))

    u = UMAP(output_metric=dist1)
    with pytest.raises(ValueError):
        u.fit(nn_data)


def test_umap_bad_hellinger_data(nn_data):
    u = UMAP(metric="hellinger")
    with pytest.raises(ValueError):
        u.fit(-nn_data)


def test_umap_update_bad_params(nn_data):
    dmat = pairwise_distances(nn_data[:100])
    u = UMAP(metric="precomputed", n_epochs=11)
    u.fit(dmat)
    with pytest.raises(ValueError):
        u.update(dmat)

    u = UMAP(n_epochs=11)
    u.fit(nn_data[:100], y=np.repeat(np.arange(5), 20))
    with pytest.raises(ValueError):
        u.update(nn_data[100:200])


def test_umap_fit_data_and_targets_compliant():
    # x and y are required to be the same length
    u = UMAP()
    x = np.random.uniform(0, 1, (256, 10))
    y = np.random.randint(10, size=(257,))
    with pytest.raises(ValueError):
        u.fit(x, y)

    u = UMAP()
    x = np.random.uniform(0, 1, (256, 10))
    y = np.random.randint(10, size=(255,))
    with pytest.raises(ValueError):
        u.fit(x, y)

    u = UMAP()
    x = np.random.uniform(0, 1, (256, 10))
    with pytest.raises(ValueError):
        u.fit(x, [])


def test_umap_fit_instance_returned():
    # Test that fit returns a new UMAP instance

    # Passing both data and targets
    u = UMAP()
    x = np.random.uniform(0, 1, (256, 10))
    y = np.random.randint(10, size=(256,))
    res = u.fit(x, y)
    assert isinstance(res, UMAP)

    # Passing only data
    u = UMAP()
    x = np.random.uniform(0, 1, (256, 10))
    res = u.fit(x)
    assert isinstance(res, UMAP)


def test_umap_inverse_transform_fails_expectedly(sparse_spatial_data, nn_data):
    u = UMAP(n_epochs=11)
    u.fit(sparse_spatial_data[:100])
    with pytest.raises(ValueError):
        u.inverse_transform(u.embedding_[:10])
    u = UMAP(metric="dice", n_epochs=11)
    u.fit(nn_data[:100])
    with pytest.raises(ValueError):
        u.inverse_transform(u.embedding_[:10])
