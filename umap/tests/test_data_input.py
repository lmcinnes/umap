import numpy as np
import pytest as pytest

# Check Data Input
# -----------------------
from numba import njit

from umap import UMAP


@pytest.fixture(scope="session")
def all_finite_data():
    d = np.arange(100.0).reshape(25, 4)
    return d


@pytest.fixture(scope="session")
def inverse_data():
    d = np.arange(50).reshape(25, 2)
    return d


@njit
def nan_dist(a: np.ndarray, b: np.ndarray):
    a[0] = np.nan
    a[1] = np.inf
    return 0, a


def test_check_input_data(all_finite_data, inverse_data):
    inf_data = all_finite_data.copy()
    inf_data[0] = np.inf

    nan_data = all_finite_data.copy()
    nan_data[0] = np.nan

    u = UMAP(metric=nan_dist)

    u.fit_transform(all_finite_data)
    u.fit(all_finite_data)
    u.transform(all_finite_data)
    u.update(all_finite_data)
    u.inverse_transform(inverse_data)

    u.fit_transform(nan_data, force_all_finite='allow-nan')
    u.fit(nan_data, force_all_finite='allow-nan')
    u.transform(nan_data, force_all_finite='allow-nan')
    u.update(nan_data, force_all_finite='allow-nan')
    u.inverse_transform(inverse_data)

    u.fit_transform(inf_data, force_all_finite=False)
    u.fit(inf_data, force_all_finite=False)
    u.transform(inf_data, force_all_finite=False)
    u.update(inf_data, force_all_finite=False)
    u.inverse_transform(inverse_data)
