# UMAP parameter map

This folder holds the code that builds the interactive parameter map in the
UMAP documentation (`doc/parameter_map.rst`). The map shows UMAP embeddings
of a dataset under many parameter settings, with settings that give similar
looking embeddings placed near each other.

## Building

The code is a [uv](https://docs.astral.sh/uv/) project that uses the UMAP in
this repository. Outputs go in `experiments/parameter_map/` at the top of the
repository, which git ignores. To put them somewhere else, set
`UMAP_PARAM_MAP_DIR`.

```bash
uv run python build.py fashion_mnist   # or 20newsgroups, celegans; none for all
uv run python make_site.py             # all built datasets in one local site
python -m http.server --directory ../experiments/parameter_map/site
```

`build.py` fits UMAP for every setting, computes each run's features and
thumbnail, lays the runs out, and writes the viewer. Each step keeps what it
makes, so an interrupted build carries on where it stopped. Fitting takes
almost all of the time: about 15 hours for the 16,384 Fashion-MNIST runs,
and 1 to 6 hours for each of the other datasets.

`make_site.py --space` writes a self-contained copy of the site that can be
uploaded to a Hugging Face static Space.

## Code

| File | What it does |
| --- | --- |
| `datasets.py` | The datasets: loading, class colours, the shared k-NN graph |
| `design.py` | The parameters, their levels, and which settings are run |
| `fit.py` | Running UMAP for each setting and saving the embeddings |
| `render.py` | Drawing the thumbnails |
| `features.py` | Describing each embedding, and the distance between them |
| `layout.py` | The meta-UMAP, the grid, and the zoom levels |
| `build.py` | All of the above, for one dataset, through to the viewer's files |
| `make_site.py` | Combining the datasets' viewers into one site |
| `viewer/index.html` | The viewer |

## Making changes

- **Parameters and levels:** edit `GRID`, `VARIED` and the sampling rules
  in `design.py`. Runs are cached by their parameters, so a changed design
  only fits the settings that are new.
- **A new dataset:** write a loader and a palette function in `datasets.py`
  and add a `Dataset` entry to `DATASETS`. The loader returns the data,
  integer labels and class names. `size` is the number of settings to run,
  which must be a power of four. Sizes below the full design use the
  first settings of the design, which are a representative sample of it.
- **How embeddings are compared:** `features.py`. Cached features are kept
  in `<dataset>/features/`; delete that folder after changing how features
  are computed.
- **How thumbnails look:** `render.py`. Delete `<dataset>/thumbs/`, or call
  `features.refresh_thumbnails`, and rebuild.
- **The layout:** `layout.py`. The layout is cached in
  `<dataset>/layout.npz`; delete that file after changing it.
