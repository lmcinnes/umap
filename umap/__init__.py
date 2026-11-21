from warnings import warn, catch_warnings, simplefilter
from .umap_ import UMAP


class _ParametricUMAPUnavailable(object):
    """Stand-in for ParametricUMAP when Tensorflow cannot be imported."""

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


_ParametricUMAPUnavailable.__name__ = "ParametricUMAP"
_ParametricUMAPUnavailable.__qualname__ = "ParametricUMAP"
_PARAMETRIC_NAMES = ("ParametricUMAP", "load_ParametricUMAP")


def __getattr__(name):
    # ParametricUMAP imports Tensorflow (and sklearn.neighbors), which costs
    # seconds; load it on first use rather than on every ``import umap``.
    if name not in _PARAMETRIC_NAMES:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    try:
        with catch_warnings():
            simplefilter("ignore")
            from .parametric_umap import ParametricUMAP, load_ParametricUMAP
    except ImportError:
        warn(
            "Tensorflow not installed; ParametricUMAP will be unavailable",
            category=ImportWarning,
        )
        globals()["ParametricUMAP"] = _ParametricUMAPUnavailable
        if name == "load_ParametricUMAP":
            raise AttributeError(
                f"module {__name__!r} has no attribute {name!r} "
                "(umap.parametric_umap requires Tensorflow >= 2.0)"
            ) from None
        return _ParametricUMAPUnavailable
    globals().update(
        ParametricUMAP=ParametricUMAP, load_ParametricUMAP=load_ParametricUMAP
    )
    return globals()[name]


def __dir__():
    return sorted(set(globals()) | set(_PARAMETRIC_NAMES))


from .aligned_umap import AlignedUMAP

# Workaround: https://github.com/numba/numba/issues/3341
import numba

from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version("umap-learn")
except PackageNotFoundError:
    __version__ = "0.5-dev"
