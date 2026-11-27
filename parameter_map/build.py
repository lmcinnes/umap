"""Build a dataset's parameter map, from UMAP fits to the viewer's files.

  1. fit UMAP for each of the dataset's settings of the design (``fit.py``);
  2. compute each run's features and thumbnail (``features.py``);
  3. lay the runs out on a grid and build the zoom levels (``layout.py``);
  4. write ``meta.json`` and the thumbnail bundles for the viewer.

Every step keeps what it has made, so a build can be stopped and started
again, and a change to one step only redoes what depends on it. The
fitting dominates: about 3.5 seconds a run for Fashion-MNIST.

The viewer goes in ``<data root>/<dataset>/viewer``; ``make_site.py`` gathers
the viewers into one site. Thumbnails are packed into bundles, each holding
a ``BUNDLE`` x ``BUNDLE`` block of cells at one zoom level, so that a view
loads a few files rather than one per tile, and a hosted copy stays well
under a host's limit on files. A bundle is ``tiles/<level>/<bx>-<by>.bin``:
a little-endian uint32 count, each thumbnail's byte length as uint32, then
the thumbnails, for the block's cells in row-major order.

Usage: python build.py <dataset> [...]   (default: every dataset)
"""

import io
import json
import shutil
import struct
import sys
from pathlib import Path

import numpy as np

import design
import features
import fit
import layout
from datasets import DATASETS

BUNDLE = 4
THUMB_FORMAT = "webp"
WEBP_QUALITY = 80
VIEWER_SOURCE = Path(__file__).parent / "viewer"


def cell_record(x, y, representative, members, run_levels, n_levels):
    """A cell: the run it shows, and how many of its runs use each level of
    each parameter (in the order of ``meta['param_names']``)."""
    record = {"x": x, "y": y, "run": representative, "count": len(members)}
    if len(members) > 1:
        record["hist"] = [
            np.bincount(run_levels[members, p], minlength=n).tolist()
            for p, n in enumerate(n_levels)
        ]
    return record


def write_bundles(dataset, out, cells, keys):
    """Write each level's thumbnails in bundles; return (files, bytes)."""
    encoded = {}

    def image(key):
        if key not in encoded:
            buffer = io.BytesIO()
            features.thumbnail(dataset, key).save(buffer, THUMB_FORMAT, quality=WEBP_QUALITY)
            encoded[key] = buffer.getvalue()
        return encoded[key]

    files = total = 0
    for level, records in cells.items():
        n = 1 << int(level)
        block = min(BUNDLE, n)
        folder = out / "tiles" / level
        folder.mkdir(parents=True)
        for by in range(n // block):
            for bx in range(n // block):
                parts = [
                    image(keys[records[y * n + x]["run"]])
                    for y in range(by * block, (by + 1) * block)
                    for x in range(bx * block, (bx + 1) * block)
                ]
                data = struct.pack(f"<{len(parts) + 1}I", len(parts), *map(len, parts))
                data += b"".join(parts)
                (folder / f"{bx}-{by}.bin").write_bytes(data)
                files, total = files + 1, total + len(data)
    return files, total


def build(dataset):
    size = dataset.size
    side = round(np.sqrt(size))
    assert side * side == size and side & (side - 1) == 0, "size must be a power of 4"
    settings = design.settings(size)
    keys = [design.run_key(p) for p in settings]
    params = [design.full_params(p) for p in settings]
    fit.run_all(dataset, settings)
    features.ensure(dataset, keys)

    print(f"{dataset.name}: laying out {size} runs", flush=True)
    distance = features.distance(dataset, keys)
    meta_layout, cells = layout.placement(distance, keys, dataset.dir / "layout.npz")
    scores = layout.scores(cells, distance)
    print("  grid:", "  ".join(f"{name} {value:.3f}" for name, value in scores.items()))
    levels_of = design.levels()
    run_levels = np.array([[levels_of[n].index(p[n]) for n in levels_of] for p in params])
    n_points = len(dataset.load()[1])
    as_used = lambda name, value: design.as_used(name, value, n_points)

    meta = {
        "side": side,
        "levels": int(np.log2(side)),
        "top_level": layout.TOP_LEVEL,
        "thumb_format": THUMB_FORMAT,
        "thumb_bundle": BUNDLE,
        "dataset": dataset.title,
        # Arguments, beyond the varied ones, needed to reproduce a plot.
        "base_params": {
            **({} if dataset.metric == "euclidean" else {"metric": dataset.metric}),
            "compatibility_layout": design.BASE_PARAMS["compatibility_layout"],
            "random_state": design.RUN_SEED,
        },
        "param_names": list(levels_of),
        # Values as passed to UMAP for this dataset.
        "param_levels": {n: [as_used(n, v) for v in levels] for n, levels in levels_of.items()},
        "defaults": {n: as_used(n, default) for n, default in design.defaults().items()},
        # Per run: its key, its level of each parameter, and its position in
        # the meta-UMAP, scaled into [0, 1] at equal aspect.
        "runs": [
            {"id": key, "levels": row.tolist(), "xy": [round(float(v), 4) for v in xy]}
            for key, row, xy in zip(keys, run_levels, layout.unit_square(meta_layout))
        ],
        # Each level's cells in row-major order: index = y * 2^level + x.
        "cells": {
            str(level): [
                cell_record(x, y, *grid[(x, y)], run_levels, [len(v) for v in levels_of.values()])
                for y in range(2**level)
                for x in range(2**level)
            ]
            for level, grid in sorted(layout.pyramid(cells, distance).items())
        },
    }

    out = dataset.dir / "viewer"
    if out.exists():
        shutil.rmtree(out)
    out.mkdir()
    files, total = write_bundles(dataset, out, meta["cells"], keys)
    with open(out / "meta.json", "w") as f:
        json.dump(meta, f, separators=(",", ":"))
    for path in VIEWER_SOURCE.iterdir():  # the page, for viewing this dataset alone
        if path.is_file():
            shutil.copy(path, out / path.name)
    print(f"  {files} bundles, {total / 1e6:.1f} MB; viewer in {out}", flush=True)


if __name__ == "__main__":
    for name in sys.argv[1:] or DATASETS:
        build(DATASETS[name])
