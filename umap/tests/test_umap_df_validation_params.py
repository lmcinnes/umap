# =========================================
#  DataFrameUMAP fit Parameters Validation
# ========================================

import numpy as np
from nose.tools import assert_raises

from umap.umap_ import DataFrameUMAP


def test_dfumap_negative_op(nn_data):
    u = DataFrameUMAP(
        metrics=[("e", "euclidean", [0, 1, 2, 3, 4])], set_op_mix_ratio=-1.0
    )
    assert_raises(ValueError, u.fit, nn_data)


def test_dfumap_too_large_op(nn_data):
    u = DataFrameUMAP(
        metrics=[("e", "euclidean", [0, 1, 2, 3, 4])], set_op_mix_ratio=1.5
    )
    assert_raises(ValueError, u.fit, nn_data)


def test_dfumap_too_large_min_dist(nn_data):
    u = DataFrameUMAP(metrics=[("e", "euclidean", [0, 1, 2, 3, 4])], min_dist=2.0)
    assert_raises(ValueError, u.fit, nn_data)


def test_dfumap_negative_min_dist(nn_data):
    u = DataFrameUMAP(metrics=[("e", "euclidean", [0, 1, 2, 3, 4])], min_dist=-1)
    assert_raises(ValueError, u.fit, nn_data)


def test_dfumap_negative_n_components(nn_data):
    u = DataFrameUMAP(metrics=[("e", "euclidean", [0, 1, 2, 3, 4])], n_components=-1)
    assert_raises(ValueError, u.fit, nn_data)


def test_dfumap_non_integer_n_components(nn_data):
    u = DataFrameUMAP(metrics=[("e", "euclidean", [0, 1, 2, 3, 4])], n_components=1.5)
    assert_raises(ValueError, u.fit, nn_data)


def test_dfumap_too_small_n_neighbours(nn_data):
    u = DataFrameUMAP(metrics=[("e", "euclidean", [0, 1, 2, 3, 4])], n_neighbors=0.5)
    assert_raises(ValueError, u.fit, nn_data)


def test_dfumap_negative_n_neighbours(nn_data):
    u = DataFrameUMAP(metrics=[("e", "euclidean", [0, 1, 2, 3, 4])], n_neighbors=-1)
    assert_raises(ValueError, u.fit, nn_data)


def test_dfumap_non_existing_metric_name(nn_data):
    u = DataFrameUMAP(metrics=[("e", "foobar", [0, 1, 2, 3, 4])])
    assert_raises(AssertionError, u.fit, nn_data)


def test_dfumap_negative_learning_rate(nn_data):
    u = DataFrameUMAP(metrics=[("e", "euclidean", [0, 1, 2, 3, 4])], learning_rate=-1.5)
    assert_raises(ValueError, u.fit, nn_data)


def test_dfumap_negative_repulsion(nn_data):
    u = DataFrameUMAP(
        metrics=[("e", "euclidean", [0, 1, 2, 3, 4])], repulsion_strength=-0.5
    )
    assert_raises(ValueError, u.fit, nn_data)


def test_dfumap_negative_sample_rate(nn_data):
    u = DataFrameUMAP(
        metrics=[("e", "euclidean", [0, 1, 2, 3, 4])], negative_sample_rate=-1
    )
    assert_raises(ValueError, u.fit, nn_data)


def test_dfumap_bad_init(nn_data):
    u = DataFrameUMAP(metrics=[("e", "euclidean", [0, 1, 2, 3, 4])], init="foobar")
    assert_raises(ValueError, u.fit, nn_data)


def test_dfumap_bad_numeric_init(nn_data):
    u = DataFrameUMAP(metrics=[("e", "euclidean", [0, 1, 2, 3, 4])], init=42)
    assert_raises(ValueError, u.fit, nn_data)


def test_dfumap_bad_matrix_init(nn_data):
    u = DataFrameUMAP(
        metrics=[("e", "euclidean", [0, 1, 2, 3, 4])],
        init=np.array([[0, 0, 0], [0, 0, 0]]),
    )
    assert_raises(ValueError, u.fit, nn_data)


def test_dfumap_negative_n_epochs(nn_data):
    u = DataFrameUMAP(metrics=[("e", "euclidean", [0, 1, 2, 3, 4])], n_epochs=-2)
    assert_raises(ValueError, u.fit, nn_data)


def test_dfumap_negative_target_n_neighbours(nn_data):
    u = DataFrameUMAP(
        metrics=[("e", "euclidean", [0, 1, 2, 3, 4])], target_n_neighbors=1
    )
    assert_raises(ValueError, u.fit, nn_data)


def test_dfumap_metrics_bad_df_column(nn_data):
    u = DataFrameUMAP(metrics=[("e", "euclidean", "bad_columns")])
    assert_raises(AssertionError, u.fit, nn_data)


def test_dfumap_metrics_bad_column_positions_as_floats(nn_data):
    u = DataFrameUMAP(metrics=[("e", "euclidean", [0.1, 0.2, 0.75])])
    assert_raises(AssertionError, u.fit, nn_data)
