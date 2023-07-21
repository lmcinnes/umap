from umap.spectral import spectral_layout, tswspectral_layout

import numpy as np
import pytest
from warnings import catch_warnings


def test_tsw_spectral_init(iris):
    # create an arbitrary (dense) random affinity matrix
    seed = 42
    rng = np.random.default_rng(seed=seed)
    # matrix must be of sufficient size of lobpcg will refuse to work on it
    n = 20
    graph = rng.standard_normal(n * n).reshape((n, n)) ** 2
    graph = graph.T * graph

    spec = spectral_layout(None, graph, 2, random_state=seed)
    tsw_spec = tswspectral_layout(None, graph, 2, random_state=seed, tol=1e-8)

    # make sure the two methods produce matrices that are close in values
    rmsd = np.sqrt(np.mean(np.sum((np.abs(spec) - np.abs(tsw_spec)) ** 2, axis=1)))
    assert (
        rmsd < 1e-6
    ), "tsvd-warmed spectral init insufficiently close to standard spectral init"


def test_ensure_fallback_to_random_on_spectral_failure():
    dim = 1000
    k = 10
    assert k >= 10
    assert dim // 10 > k
    y = np.eye(dim, k=1)
    u = np.random.random((dim, dim // 10))
    graph = y + y.T + u @ u.T
    with pytest.warns(
        UserWarning,
        match="Spectral initialisation failed!"
    ):
        tswspectral_layout(u, graph, k, random_state=42, maxiter=2)
