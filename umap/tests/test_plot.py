from nose.tools import assert_less
from nose.tools import assert_greater_equal
import os.path
import numpy as np

from nose import SkipTest

from sklearn import datasets

import umap
import umap.plot

np.random.seed(42)
iris = datasets.load_iris()

mapper = umap.UMAP().fit(iris.data)


def test_plot_runs_at_all():
    umap.plot.points(mapper, labels=iris.target)
    umap.plot.points(mapper, values=iris.data[:, 0])
    umap.plot.diagnostic(mapper, diagnostic_type="all")
    umap.plot.connectivity(mapper)
