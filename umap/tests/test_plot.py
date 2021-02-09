import numpy as np
import pytest
import umap

# Globals, used for all the tests
SEED = 189212  # 0b101110001100011100
np.random.seed(SEED)

try:
    from umap import plot

    IMPORT_PLOT = True
except ImportError:
    IMPORT_PLOT = False

plot_only = pytest.mark.skipif(not IMPORT_PLOT, reason="umap plot not found.")


@pytest.fixture(scope="session")
def mapper(iris):
    return umap.UMAP(n_epochs=100).fit(iris.data)


# These tests requires revision: Refactoring is
# needed as there is no assertion nor
# property verification.
@plot_only
def test_plot_runs_at_all(mapper, iris, iris_selection):
    from umap import plot as umap_plot

    umap_plot.points(mapper)
    umap_plot.points(mapper, labels=iris.target)
    umap_plot.points(mapper, values=iris.data[:, 0])
    umap_plot.points(mapper, labels=iris.target, subset_points=iris_selection)
    umap_plot.points(mapper, values=iris.data[:, 0], subset_points=iris_selection)
    umap_plot.points(mapper, theme="fire")
    umap_plot.diagnostic(mapper, diagnostic_type="all")
    umap_plot.diagnostic(mapper, diagnostic_type="neighborhood")
    umap_plot.connectivity(mapper)
    umap_plot.connectivity(mapper, theme="fire")
    umap_plot.connectivity(mapper, edge_bundling="hammer")
    umap_plot.interactive(mapper)
    umap_plot.interactive(mapper, labels=iris.target)
    umap_plot.interactive(mapper, values=iris.data[:, 0])
    umap_plot.interactive(mapper, labels=iris.target, subset_points=iris_selection)
    umap_plot.interactive(mapper, values=iris.data[:, 0], subset_points=iris_selection)
    umap_plot.interactive(mapper, theme="fire")
    umap_plot._datashade_points(mapper.embedding_)
    umap_plot._datashade_points(mapper.embedding_, labels=iris.target)
    umap_plot._datashade_points(mapper.embedding_, values=iris.data[:, 0])
