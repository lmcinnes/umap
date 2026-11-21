import inspect

import numba
import numpy as np
import pytest

from umap import UMAP, layouts


@pytest.mark.parametrize(
    "optimizer,kernel,float64_params",
    [
        (
            "adam",
            layouts.optimize_small_layout_euclidean_single_epoch_adam,
            ("a", "b", "gamma", "beta1", "beta2"),
        ),
        (
            "momentum",
            layouts.optimize_small_layout_euclidean_single_epoch_momentum,
            ("a", "b", "gamma"),
        ),
    ],
)
def test_layout_kernel_scalars_stay_float64(optimizer, kernel, float64_params):
    data = np.random.default_rng(0).normal(size=(300, 8)).astype(np.float32)
    UMAP(
        n_neighbors=10,
        n_epochs=5,
        optimizer=optimizer,
        a=np.float32(1.5),
        b=np.float32(0.9),
        random_state=0,
    ).fit(data)
    param_names = list(inspect.signature(kernel.py_func).parameters)
    assert kernel.signatures
    for signature in kernel.signatures:
        for name in float64_params:
            assert signature[param_names.index(name)] == numba.float64
