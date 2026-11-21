"""``import umap`` should not pull in modules only some code paths need."""

import subprocess
import sys


def _run(code):
    result = subprocess.run(
        [sys.executable, "-W", "ignore", "-c", code],
        capture_output=True,
        text=True,
        check=True,
    )
    return result.stdout.strip()


def test_import_umap_defers_optional_heavy_modules():
    loaded = _run(
        "import sys, umap; print(','.join(m for m in ("
        "'umap.parametric_umap', 'sklearn.manifold', 'sklearn.neighbors', "
        "'tensorflow', 'keras') if m in sys.modules))"
    )
    assert loaded == ""


def test_parametric_umap_resolves_lazily():
    out = _run(
        "import umap\n"
        "try:\n"
        "    import tensorflow  # noqa: F401\n"
        "    have_tf = True\n"
        "except ImportError:\n"
        "    have_tf = False\n"
        "P = umap.ParametricUMAP\n"
        "from umap import ParametricUMAP\n"
        "assert P is ParametricUMAP and P.__name__ == 'ParametricUMAP'\n"
        "assert 'ParametricUMAP' in dir(umap)\n"
        "if not have_tf:\n"
        "    try:\n"
        "        P()\n"
        "        raise AssertionError('stub should raise')\n"
        "    except ImportError:\n"
        "        pass\n"
        "try:\n"
        "    umap.not_an_attribute\n"
        "    raise AssertionError('expected AttributeError')\n"
        "except AttributeError:\n"
        "    pass\n"
        "print('ok')"
    )
    assert out == "ok"
