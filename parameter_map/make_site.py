"""Gather the datasets' viewers into one site, with a dataset drop-down.

The site folder holds the viewer page, ``datasets.json`` listing the
datasets in drop-down order, and a folder per dataset with its ``meta.json``
and thumbnail bundles. By default it is ``<data root>/site``, and the
dataset folders are symbolic links to each dataset's viewer, for local use
(``python -m http.server --directory <that folder>``).

``--space`` writes a self-contained copy to ``<data root>/space`` instead,
with the README a Hugging Face static Space needs, ready to upload.

Usage: python make_site.py [--space]
"""

import json
import shutil
import sys
from pathlib import Path

from datasets import DATA_ROOT, DATASETS

SPACE_README = """---
title: UMAP Parameter Map
emoji: 🗺️
colorFrom: blue
colorTo: gray
sdk: static
pinned: false
short_description: Explore how UMAP's parameters change its embeddings
---

An interactive map of UMAP embeddings under many parameter settings, on
three datasets. Zoom towards the kind of embedding you want, then click a
tile for the parameters and code that produce it. See the
[UMAP documentation](https://umap-learn.readthedocs.io/en/latest/parameter_map.html)
for how the map was made.
"""
VIEWER_SOURCE = Path(__file__).parent / "viewer"


def main():
    copy = "--space" in sys.argv[1:]
    out = DATA_ROOT / ("space" if copy else "site")
    out.mkdir(exist_ok=True)
    for path in VIEWER_SOURCE.iterdir():
        if path.is_file():
            shutil.copy(path, out / path.name)
    if copy:
        (out / "README.md").write_text(SPACE_README)
    listed = []
    for dataset in DATASETS.values():
        source, target = dataset.dir / "viewer", out / dataset.name
        if not (source / "meta.json").exists():
            print(f"skipping {dataset.name}: no viewer built in {source}")
            continue
        if target.is_symlink() or target.is_file():
            target.unlink()
        elif target.exists():
            shutil.rmtree(target)
        if copy:
            target.mkdir()
            shutil.copy(source / "meta.json", target)
            shutil.copytree(source / "tiles", target / "tiles")
        else:
            target.symlink_to(source, target_is_directory=True)
        listed.append({"id": dataset.name, "label": dataset.label})
    with open(out / "datasets.json", "w") as f:
        json.dump(listed, f, indent=1)
    print(f"site in {out} with {', '.join(d['id'] for d in listed)}")


if __name__ == "__main__":
    main()
