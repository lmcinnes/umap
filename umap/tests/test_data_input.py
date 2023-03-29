import numpy as np
import pytest as pytest
from numba import njit
from umap import UMAP


@pytest.fixture(scope="session")
def all_finite_data():
    return np.arange(100.0).reshape(25, 4)


@pytest.fixture(scope="session")
def inverse_data():
    return np.arange(50).reshape(25, 2)


@njit
def nan_dist(a: np.ndarray, b: np.ndarray):
    a[0] = np.nan
    a[1] = np.inf
    return 0, a


def test_check_input_data(all_finite_data, inverse_data):
    """
    Data input to UMAP gets checked for liability.
    This tests checks the if data input is dismissed/accepted
    according to the "force_all_finite" keyword as used by
    sklearn.

    Parameters
    ----------
    all_finite_data
    inverse_data
    -------

    """
    inf_data = all_finite_data.copy()
    inf_data[0] = np.inf
    nan_data = all_finite_data.copy()
    nan_data[0] = np.nan
    inf_nan_data = all_finite_data.copy()
    inf_nan_data[0] = np.nan
    inf_nan_data[1] = np.inf

    # wrapper to call each data handling function of UMAP in a convenient way
    def call_umap_functions(data, force_all_finite):
        u = UMAP(metric=nan_dist)
        if force_all_finite is None:
            u.fit_transform(data)
            u.fit(data)
            u.transform(data)
            u.update(data)
            u.inverse_transform(inverse_data)
        else:
            u.fit_transform(data, force_all_finite=force_all_finite)
            u.fit(data, force_all_finite=force_all_finite)
            u.transform(data, force_all_finite=force_all_finite)
            u.update(data, force_all_finite=force_all_finite)
            u.inverse_transform(inverse_data)

    # Check whether correct data input is accepted
    call_umap_functions(all_finite_data, None)
    call_umap_functions(all_finite_data, True)

    call_umap_functions(nan_data, 'allow-nan')
    call_umap_functions(all_finite_data, 'allow-nan')

    call_umap_functions(inf_data, False)
    call_umap_functions(inf_nan_data, False)
    call_umap_functions(nan_data, False)
    call_umap_functions(all_finite_data, False)

    # Check whether illegal data raises a ValueError
    with pytest.raises(ValueError):
        call_umap_functions(nan_data, None)
        call_umap_functions(inf_data, None)
        call_umap_functions(inf_nan_data, None)

        call_umap_functions(nan_data, True)
        call_umap_functions(inf_data, True)
        call_umap_functions(inf_nan_data, True)

        call_umap_functions(inf_data, 'allow-nan')
        call_umap_functions(inf_nan_data, 'allow-nan')
