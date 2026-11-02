"""Regenerate the umap-learn 0.5.x compatibility fixtures.

The reference is the final 0.5.x code line: the ``master`` branch at the
commit merged into 0.6 (a2e06aa), which includes bug fixes made after the
0.5.12 release. Export that tree and pass its root (the directory containing
the ``umap`` package)::

    git archive a2e06aa umap | tar -x -C /tmp/umap-0.5-final
    python make_compat_fixtures.py /tmp/umap-0.5-final [output.npz]

Without an output path the committed fixture next to this script is replaced.
CI regenerates fixtures in its own environment and points the regression test
at them with the ``UMAP_COMPAT_FIXTURES`` environment variable.

The output records the platform and the versions of NumPy (and its BLAS),
SciPy, scikit-learn, Numba and PyNNDescent, since bit-exact results are only
expected when all of them match.
"""

import importlib.util
import os
import sys
import warnings

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))


def main(legacy_root, output=None):
    # An editable install of the development tree registers an import hook that
    # would shadow the legacy checkout, so drop it before importing umap.
    sys.meta_path[:] = [
        finder
        for finder in sys.meta_path
        if not type(finder).__module__.startswith("__editable__")
    ]
    sys.path.insert(0, os.path.abspath(legacy_root))
    import umap

    if not umap.__file__.startswith(os.path.abspath(legacy_root)):
        raise RuntimeError(f"Imported umap from {umap.__file__}, not {legacy_root}")

    spec = importlib.util.spec_from_file_location(
        "_compat_cases", os.path.join(HERE, os.pardir, "_compat_cases.py")
    )
    cases_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cases_module)

    arrays = {"environment": np.array(cases_module.environment_fingerprint())}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for name, case in cases_module.build_cases().items():
            results = cases_module.run_case(umap, *case, extra_kwargs={})
            for key, value in results.items():
                arrays[f"{name}/{key}"] = np.asarray(value)
            print("generated", name)

        for key, value in cases_module.run_aligned_case(umap, {}).items():
            arrays[f"aligned/{key}"] = np.asarray(value)
        print("generated aligned")

    if output is None:
        output = os.path.join(HERE, cases_module.FIXTURE_FILE)
    np.savez_compressed(output, **arrays)
    print("wrote", output)


if __name__ == "__main__":
    main(*sys.argv[1:3])
