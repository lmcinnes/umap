"""Regenerate the umap-learn 0.5.x compatibility fixtures.

The reference is the final 0.5.x code line: the ``master`` branch at the
commit merged into 0.6 (a2e06aa), which includes bug fixes made after the
0.5.12 release. Export that tree and pass its root (the directory containing
the ``umap`` package)::

    git archive a2e06aa umap | tar -x -C /tmp/umap-0.5-final
    python make_compat_fixtures.py /tmp/umap-0.5-final

The output is written next to this script and records the platform used, since
bit-exact numba results are only expected on a matching platform.
"""

import importlib.util
import os
import platform
import sys
import warnings

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))


def main(legacy_root):
    sys.path.insert(0, legacy_root)
    import umap

    if not umap.__file__.startswith(os.path.abspath(legacy_root)):
        raise RuntimeError(f"Imported umap from {umap.__file__}, not {legacy_root}")

    spec = importlib.util.spec_from_file_location(
        "_compat_cases", os.path.join(HERE, os.pardir, "_compat_cases.py")
    )
    cases_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cases_module)

    arrays = {
        "platform": np.array(f"{platform.system()}-{platform.machine()}"),
        "numba_version": np.array(__import__("numba").__version__),
    }
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

    output = os.path.join(HERE, cases_module.FIXTURE_FILE)
    np.savez_compressed(output, **arrays)
    print("wrote", output)


if __name__ == "__main__":
    main(sys.argv[1])
