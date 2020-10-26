from warnings import warn, catch_warnings, simplefilter
from .umap_ import UMAP

try:
    with catch_warnings():
        simplefilter("ignore")
        from .parametric_umap import ParametricUMAP
except ImportError:
    warn("Tensorflow not installed; ParametricUMAP will be unavailable")
from .aligned_umap import AlignedUMAP

# Workaround: https://github.com/numba/numba/issues/3341
import numba

import pkg_resources

try:
    __version__ = pkg_resources.get_distribution("umap-learn").version
except pkg_resources.DistributionNotFound:
    __version__ = "0.5-dev"
