from warnings import warn, catch_warnings, simplefilter
from .umap_ import UMAP

try:
    with catch_warnings():
        simplefilter("ignore")
        from .parametric_umap import ParametricUMAP, load_ParametricUMAP
except ImportError:
    warn(
        "Tensorflow not installed; ParametricUMAP will be unavailable",
        category=ImportWarning,
    )

    # Add a dummy class to raise an error
    class ParametricUMAP(object):
        """A placeholder class for ParametricUMAP when TensorFlow is not installed.

        This class provides a graceful error message when users attempt to instantiate
        ParametricUMAP without having TensorFlow installed. The real implementation
        requires TensorFlow >= 2.0 and is loaded from umap.parametric_umap.

        Examples
        --------
        >>> try:
        ...     from umap import ParametricUMAP
        ... except ImportError:
        ...     pass  # TensorFlow not available

        See Also
        --------
        load_ParametricUMAP : Load a saved ParametricUMAP model
        """
        def __init__(self, **kwds):
            warn(
                """The umap.parametric_umap package requires Tensorflow > 2.0 to be installed.
            You can install Tensorflow at https://www.tensorflow.org/install

            or you can install the CPU version of Tensorflow using 

            pip install umap-learn[parametric_umap]

            """
            )
            raise ImportError(
                "umap.parametric_umap requires Tensorflow >= 2.0"
            ) from None


from .aligned_umap import AlignedUMAP

# Workaround: https://github.com/numba/numba/issues/3341
import numba

from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version("umap-learn")
except PackageNotFoundError:
    __version__ = "0.5-dev"
