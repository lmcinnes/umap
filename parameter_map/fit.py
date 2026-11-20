"""Fit UMAP for each setting of the design, saving every embedding.

Runs share the dataset's exact k-nearest-neighbor graph, which UMAP prunes
to each run's n_neighbors, so that a run costs only graph construction and
layout. Each embedding is saved by run key, and a run already made is never
repeated, so an interrupted batch simply carries on when restarted.
"""

import json
import subprocess
import time
import warnings

import numpy as np

import design
from datasets import REPO_ROOT


def embedding_path(dataset, key):
    return dataset.dir / "runs" / f"{key}.npy"


def load_embedding(dataset, key):
    return np.load(embedding_path(dataset, key))


def _git_commit():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], cwd=REPO_ROOT, text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def run_one(dataset, params, data, knn, seed=design.RUN_SEED):
    """Fit UMAP with ``params`` (a setting of the design); return seconds
    taken, or None if the run already exists.

    The embedding is saved as float32 and the run logged in ``runs.jsonl``.
    """
    import umap

    key = design.run_key(params, seed)
    path = embedding_path(dataset, key)
    if path.exists():
        return None
    path.parent.mkdir(parents=True, exist_ok=True)
    used = {name: design.as_used(name, value, len(data)) for name, value in params.items()}
    start = time.perf_counter()
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="precomputed_knn")
        embedding = umap.UMAP(
            **design.BASE_PARAMS, metric=dataset.metric, **used,
            random_state=seed, precomputed_knn=knn,
        ).fit_transform(data)  # fmt: skip
    seconds = time.perf_counter() - start

    # Write then rename, so an interrupted run never leaves a partial file.
    partial = path.with_suffix(".partial")
    with open(partial, "wb") as f:
        np.save(f, embedding.astype(np.float32))
    partial.replace(path)
    record = {
        "key": key,
        "params": params,
        "seed": seed,
        "seconds": round(seconds, 3),
        "finite": bool(np.isfinite(embedding).all()),
        "commit": _git_commit(),
    }
    with open(dataset.dir / "runs.jsonl", "a") as f:
        f.write(json.dumps(record, sort_keys=True) + "\n")
    return seconds


def run_all(dataset, settings):
    """Make every run in ``settings`` that does not exist yet."""
    missing = [p for p in settings if not embedding_path(dataset, design.run_key(p)).exists()]
    if not missing:
        return
    print(f"{dataset.name}: fitting {len(missing)} of {len(settings)} settings", flush=True)
    data, _ = dataset.load()
    knn = dataset.knn()
    import umap

    # Compile the Numba kernels first, so the first timings are fair.
    umap.UMAP(compatibility_layout=False, random_state=0, n_epochs=20).fit(data[:5000])
    start = time.perf_counter()
    for i, params in enumerate(missing, 1):
        run_one(dataset, params, data, knn)
        if i % 50 == 0 or i == len(missing):
            minutes = (time.perf_counter() - start) / 60
            print(f"  {i}/{len(missing)} fitted, {minutes:.1f} min", flush=True)
