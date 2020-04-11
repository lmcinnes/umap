# ===============================
#  UMAP fit Parameters Validation
# ===============================

import warnings
import numpy as np
from nose.tools import assert_equal, assert_raises
from umap import UMAP

warnings.filterwarnings("ignore", category=UserWarning)


def test_umap_negative_op(nn_data):
    u = UMAP(set_op_mix_ratio=-1.0)
    assert_raises(ValueError, u.fit, nn_data)


def test_umap_too_large_op(nn_data):
    u = UMAP(set_op_mix_ratio=1.5)
    assert_raises(ValueError, u.fit, nn_data)


def test_umap_bad_too_large_min_dist(nn_data):
    u = UMAP(min_dist=2.0)
    # a RuntimeWarning about division by zero in a,b curve fitting is expected
    # caught and ignored for this test
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        assert_raises(ValueError, u.fit, nn_data)


def test_umap_negative_min_dist(nn_data):
    u = UMAP(min_dist=-1)
    assert_raises(ValueError, u.fit, nn_data)


def test_umap_negative_n_components(nn_data):
    u = UMAP(n_components=-1)
    assert_raises(ValueError, u.fit, nn_data)


def test_umap_non_integer_n_components(nn_data):
    u = UMAP(n_components=1.5)
    assert_raises(ValueError, u.fit, nn_data)


def test_umap_too_small_n_neighbours(nn_data):
    u = UMAP(n_neighbors=0.5)
    assert_raises(ValueError, u.fit, nn_data)


def test_umap_negative_n_neighbours(nn_data):
    u = UMAP(n_neighbors=-1)
    assert_raises(ValueError, u.fit, nn_data)


def test_umap_bad_metric(nn_data):
    u = UMAP(metric=45)
    assert_raises(ValueError, u.fit, nn_data)


def test_umap_negative_learning_rate(nn_data):
    u = UMAP(learning_rate=-1.5)
    assert_raises(ValueError, u.fit, nn_data)


def test_umap_negative_repulsion(nn_data):
    u = UMAP(repulsion_strength=-0.5)
    assert_raises(ValueError, u.fit, nn_data)


def test_umap_negative_sample_rate(nn_data):
    u = UMAP(negative_sample_rate=-1)
    assert_raises(ValueError, u.fit, nn_data)


def test_umap_bad_init(nn_data):
    u = UMAP(init="foobar")
    assert_raises(ValueError, u.fit, nn_data)


def test_umap_bad_numeric_init(nn_data):
    u = UMAP(init=42)
    assert_raises(ValueError, u.fit, nn_data)


def test_umap_bad_matrix_init(nn_data):
    u = UMAP(init=np.array([[0, 0, 0], [0, 0, 0]]))
    assert_raises(ValueError, u.fit, nn_data)


def test_umap_negative_n_epochs(nn_data):
    u = UMAP(n_epochs=-2)
    assert_raises(ValueError, u.fit, nn_data)


def test_umap_negative_target_n_neighbours(nn_data):
    u = UMAP(target_n_neighbors=1)
    assert_raises(ValueError, u.fit, nn_data)


def test_umap_bad_output_metric(nn_data):
    u = UMAP(output_metric="foobar")
    assert_raises(ValueError, u.fit, nn_data)
    u = UMAP(output_metric="precomputed")
    assert_raises(ValueError, u.fit, nn_data)
    u = UMAP(output_metric="hamming")
    assert_raises(ValueError, u.fit, nn_data)


def test_haversine_on_highd(nn_data):
    u = UMAP(metric="haversine")
    assert_raises(ValueError, u.fit, nn_data)


def test_umap_haversine_embed_to_highd(nn_data):
    u = UMAP(n_components=3, output_metric="haversine")
    assert_raises(ValueError, u.fit, nn_data)


def test_umap_too_many_neighbors_warns(nn_data):
    u = UMAP(a=1.2, b=1.75, n_neighbors=2000, n_epochs=11, init="random")
    u.fit(
        nn_data[:100,]
    )
    assert_equal(u._a, 1.2)
    assert_equal(u._b, 1.75)


def test_umap_fit_data_and_targets_compliant():
    # x and y are required to be the same length
    u = UMAP()
    x = np.random.uniform(0, 1, (256, 10))
    y = np.random.randint(10, size=(257,))
    assert_raises(ValueError, u.fit, x, y)

    u = UMAP()
    x = np.random.uniform(0, 1, (256, 10))
    y = np.random.randint(10, size=(255,))
    assert_raises(ValueError, u.fit, x, y)

    u = UMAP()
    x = np.random.uniform(0, 1, (256, 10))
    assert_raises(ValueError, u.fit, x, [])


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
