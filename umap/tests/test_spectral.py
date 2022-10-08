from umap.spectral import spectral_layout, tswspectral_layout

import numpy as np


def test_tsw_spectral_init(iris):
    # create an arbitrary (dense) random affinity matrix
    seed = 42
    rng = np.random.default_rng(seed=seed)
    # matrix must be of sufficient size of lobpcg will refuse to work on it
    n = 20
    graph = rng.standard_normal(n * n).reshape((n, n)) ** 2
    graph = graph.T * graph

    spec = spectral_layout(None, graph, 2, random_state=seed)
    tsw_spec = tswspectral_layout(None, graph, 2, random_state=seed)

    # make sure the two methods produce matrices that are close in values
    rmsd = np.sqrt(np.mean(np.sum((np.abs(spec) - np.abs(tsw_spec)) ** 2, axis=1)))
    assert (
        rmsd < 1e-6
    ), "tsvd-warmed spectral init insufficiently close to standard spectral init"
