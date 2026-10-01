from umap.spectral import spectral_layout, tswspectral_layout

import numpy as np
import scipy.sparse
import pytest
import re
from scipy.version import full_version as scipy_full_version_
from warnings import catch_warnings

scipy_full_version = tuple(
    int(n)
    for n in re.findall(r"[0-9]+\.[0-9]+\.?[0-9]*", scipy_full_version_)[0].split(".")
)


@pytest.mark.skipif(
    scipy_full_version < (1, 10) or scipy_full_version >= (1, 15),
    reason="SciPy installing with Python 3.7 does not converge under same circumstances",
)
def test_tsw_spectral_init(iris):
    # create an arbitrary (dense) random affinity matrix
    seed = 42
    rng = np.random.default_rng(seed=seed)
    # matrix must be of sufficient size of lobpcg will refuse to work on it
    n = 20
    graph = rng.standard_normal(n * n).reshape((n, n)) ** 2
    graph = graph.T * graph

    spec = spectral_layout(None, graph, 2, random_state=seed**2)
    tsw_spec = tswspectral_layout(None, graph, 2, random_state=seed**2, tol=1e-8)

    # Make sure the two methods produce similar embeddings.
    rmsd = np.mean(np.sum((spec - tsw_spec) ** 2, axis=1))
    assert (
        rmsd < 1e-6
    ), "tsvd-warmed spectral init insufficiently close to standard spectral init"


@pytest.mark.skipif(
    scipy_full_version < (1, 10),
    reason="SciPy installing with Py 3.7 does not warn reliably on convergence failure",
)
def test_ensure_fallback_to_random_on_spectral_failure():
    dim = 1000
    k = 10
    assert k >= 10
    assert dim // 10 > k
    y = np.eye(dim, k=1)
    u = np.random.random((dim, dim // 10))
    graph = y + y.T + u @ u.T
    with pytest.warns(UserWarning, match="Spectral initialisation failed!"):
        tswspectral_layout(u, graph, k, random_state=42, maxiter=2, method="lobpcg")


def test_spectral_layout_regular_graph_is_deterministic():
    # lmcinnes/umap#1277: on a regular graph (here twelve exact duplicates of
    # one point, whose fuzzy graph is complete with every weight 1) the
    # all-ones start vector is an exact eigenvector of the normalised
    # Laplacian, and ARPACK restarts from its own internal random generator,
    # which random_state does not control.
    n = 12
    graph = scipy.sparse.csr_matrix(np.ones((n, n)) - np.eye(n))
    layouts = [spectral_layout(None, graph, 5, random_state=42) for _ in range(3)]
    assert np.array_equal(layouts[0], layouts[1])
    assert np.array_equal(layouts[1], layouts[2])
