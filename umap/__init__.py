from .umap_ import UMAP
from .parametric_umap import ParametricUMAP
from .aligned_umap import AlignedUMAP

# Workaround: https://github.com/numba/numba/issues/3341
import numba

import pkg_resources

try:
    __version__ = pkg_resources.get_distribution("umap-learn").version
except pkg_resources.DistributionNotFound:
    __version__ = "0.5-dev"
