from .umap_ import UMAP

# Workaround: https://github.com/numba/numba/issues/3341
import numba

import pkg_resources

__version__ = pkg_resources.get_distribution("umap-learn").version
