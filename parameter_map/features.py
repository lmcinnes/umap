"""How alike two embeddings look: per-run features and the distance between runs.

Two embeddings can look much the same while rotated, reflected or rearranged
relative to each other, so runs are compared through features that describe
what a plot looks like rather than where its points are. There are three
groups, each made of blocks of features:

  texture:     band-pass energy of the plot's ink at several spatial scales,
               overall and per class (grain and speckle against solid blobs)
  morphology:  area, outline length, pieces and holes of the inked region as
               the ink threshold rises, overall and per class
  descriptors: neighbor distances at several scales, distances between random
               pairs of points, the size of each class, and how well each
               class keeps together

Each run's features and thumbnail are computed once and cached by run key.
The distance between runs z-scores every feature across the runs compared,
weights each block equally within its group, and blends the groups.
"""

import itertools
from concurrent.futures import ProcessPoolExecutor

import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter, label
from scipy.spatial.distance import pdist, squareform
from sklearn.neighbors import KDTree

import design
import fit
import render
from datasets import DATASETS

GROUPS = {
    "texture": ["texture_overall", "texture_classes"],
    "morphology": ["morphology_overall", "morphology_classes"],
    "descriptors": ["local", "pairs", "compact", "purity"],
}
# Thumbnails are rendered at this size, and every run is aligned to this one.
THUMB = 256
REFERENCE_PARAMS = {"n_neighbors": 15, "min_dist": 0.1}
# Images for the texture and morphology features.
INK_SIZE = 128
INK_EXPOSURE = 2.0
# Points and point pairs sampled for the descriptors.
DESCRIPTOR_SAMPLE = 4000
DESCRIPTOR_PAIRS = 20000


# -- Ink images -----------------------------------------------------------------


def point_spread(n_points, size):
    """Standard deviation, in pixels, of the blur that stands in for point size.

    A dataset smaller than Fashion-MNIST has too few points to fill its
    shapes at one pixel each, so each point is spread over an area that
    grows as the points get fewer. It is zero at 70,000 points or more.
    """
    ratio = design.REFERENCE_SIZE / n_points
    return 0.0 if ratio <= 1 else 0.18 * np.sqrt(ratio) * size / 128


def ink_images(embedding, labels, reference, size=INK_SIZE, exposure=INK_EXPOSURE):
    """Return (total, per_class) opacity images in [0, 1], framed as thumbnails
    (without gap shortening). ``total`` is (size, size); ``per_class`` is
    (n_classes, size, size)."""
    x = render.align(embedding, reference)
    centre, extent = render.frame(x)
    pixels = np.floor((x - centre) * (size * 0.92 / extent) + size / 2).astype(np.int64)
    inside = ((pixels >= 0) & (pixels < size)).all(axis=1)
    n_classes = int(labels.max()) + 1
    flat = (labels[inside] * size + size - 1 - pixels[inside, 1]) * size
    counts = np.bincount(flat + pixels[inside, 0], minlength=n_classes * size * size)
    counts = counts.reshape(n_classes, size, size).astype(np.float64)
    spread = point_spread(len(x), size)
    if spread:
        counts = gaussian_filter(counts, sigma=(0, spread, spread))
    density = counts * size * size / len(x)
    return 1 - np.exp(-density.sum(axis=0) / exposure), 1 - np.exp(-density / exposure)


# -- Texture ----------------------------------------------------------------------


def texture_features(total, per_class, sigmas=(1, 2, 4, 8, 16)):
    """Band-pass energy of the ink at each spatial scale, ignoring position.

    Difference-of-Gaussians bands split the image into detail at 1-2 pixels,
    2-4 pixels and so on. Fine-band energy means grain, speckle and sharp
    edges; coarse-band energy means large solid blobs. Returns (overall,
    per-class) feature vectors.
    """

    def band_energies(image):
        blurred = [gaussian_filter(image, s) for s in sigmas]
        bands = [image - blurred[0]] + [a - b for a, b in itertools.pairwise(blurred)]
        return [np.log(np.abs(band).mean() + 1e-6) for band in bands]

    overall = band_energies(total) + [np.log(total.mean() + 1e-6)]
    classes = np.concatenate([band_energies(image) for image in per_class])
    return np.array(overall), classes


# -- Morphology -------------------------------------------------------------------

_EIGHT = np.ones((3, 3), dtype=int)


def _shape_summary(mask):
    """Area, boundary length, number of pieces and number of holes of a mask."""
    size = mask.shape[0]
    padded = np.pad(mask, 1)
    boundary = (padded[1:, :] != padded[:-1, :]).sum() + (padded[:, 1:] != padded[:, :-1]).sum()
    pieces = label(mask, structure=_EIGHT)[1]
    holes = label(~padded)[1] - 1
    return [mask.mean(), boundary / size, np.log1p(pieces), np.log1p(holes)]


def morphology_features(
    total, per_class, thresholds=(0.05, 0.15, 0.3, 0.5, 0.7, 0.9), blurs=(0, 2)
):
    """Shape of the inked region as the ink threshold rises.

    Thresholding the plot at increasing opacity and measuring area, boundary
    length, pieces and holes describes how much of the frame is filled, how
    ragged the outlines are, how fragmented the layout is, and how hollow.
    Returns (overall, per-class) feature vectors.
    """
    overall = []
    for blur in blurs:
        image = gaussian_filter(total, blur) if blur else total
        for threshold in thresholds:
            overall += _shape_summary(image > threshold)
    classes = []
    for image in per_class:
        classes += _shape_summary(gaussian_filter(image, 1) > 0.3)[:3]
    return np.array(overall), np.array(classes)


# -- Descriptors ------------------------------------------------------------------


def descriptor_blocks(embedding, labels, sample, pairs, ks=(5, 50, 500)):
    """Rotation- and arrangement-invariant summaries of an embedding.

    local:   quantiles of log k-th neighbor distance, for each k in ``ks``
             (how tightly packed points are at several scales)
    pairs:   quantiles of distances between random point pairs (overall shape)
    compact: log RMS radius of each class relative to the whole
    purity:  per class, the share of a point's 50 neighbors with its label
    """
    x = np.asarray(embedding, dtype=np.float64)
    x = x - x.mean(axis=0)
    x = x / np.sqrt((x**2).sum(axis=1).mean())
    quantiles = np.linspace(0.02, 0.98, 16)
    dists, indices = KDTree(x).query(x[sample], k=max(ks) + 1)
    local = [np.quantile(np.log(dists[:, k] + 1e-9), quantiles) for k in ks]
    pair_dists = np.linalg.norm(x[pairs[:, 0]] - x[pairs[:, 1]], axis=1)

    same = labels[indices[:, 1:51]] == labels[sample][:, None]
    compact, purity = [], []
    for c in range(int(labels.max()) + 1):
        members = x[labels == c]
        spread = np.sqrt(((members - members.mean(axis=0)) ** 2).sum(axis=1).mean())
        compact.append(np.log(spread))
        purity.append(same[labels[sample] == c].mean())
    return {
        "local": np.concatenate(local),
        "pairs": np.quantile(pair_dists, quantiles),
        "compact": np.array(compact),
        "purity": np.array(purity),
    }


# -- Per-run cache ----------------------------------------------------------------

_state = {}


def _init(name):
    """Set up a worker: the dataset, its labels, and the shared reference run."""
    dataset = DATASETS[name]
    _, labels = dataset.load()
    rng = np.random.default_rng(0)
    _state.update(
        dataset=dataset,
        labels=labels,
        palette=dataset.palette(),
        reference=fit.load_embedding(dataset, design.run_key(REFERENCE_PARAMS)),
        sample=rng.choice(len(labels), DESCRIPTOR_SAMPLE, replace=False),
        pairs=rng.integers(0, len(labels), size=(DESCRIPTOR_PAIRS, 2)),
    )


def _compute(key):
    dataset, labels, reference = _state["dataset"], _state["labels"], _state["reference"]
    embedding = fit.load_embedding(dataset, key)
    total, per_class = ink_images(embedding, labels, reference)
    texture = texture_features(total, per_class)
    morphology = morphology_features(total, per_class)
    blocks = descriptor_blocks(embedding, labels, _state["sample"], _state["pairs"])
    np.savez(
        dataset.dir / "features" / f"{key}.npz",
        texture_overall=texture[0],
        texture_classes=texture[1],
        morphology_overall=morphology[0],
        morphology_classes=morphology[1],
        **blocks,
    )
    _save_thumbnail(key, embedding)
    return key


def _save_thumbnail(key, embedding=None):
    dataset = _state["dataset"]
    if embedding is None:
        embedding = fit.load_embedding(dataset, key)
    image = render.thumbnail(
        render.align(embedding, _state["reference"]), _state["labels"], _state["palette"], THUMB
    )
    Image.fromarray(image).save(dataset.dir / "thumbs" / f"{key}.png")
    return key


def ensure(dataset, keys, workers=8):
    """Compute and cache features and thumbnails for any runs lacking them."""
    (dataset.dir / "features").mkdir(exist_ok=True)
    (dataset.dir / "thumbs").mkdir(exist_ok=True)
    missing = [
        key for key in keys
        if not (dataset.dir / "features" / f"{key}.npz").exists()
        or not (dataset.dir / "thumbs" / f"{key}.png").exists()
    ]  # fmt: skip
    if missing:
        print(f"{dataset.name}: features and thumbnails for {len(missing)} runs", flush=True)
        with ProcessPoolExecutor(workers, initializer=_init, initargs=(dataset.name,)) as pool:
            list(pool.map(_compute, missing, chunksize=8))


def refresh_thumbnails(dataset, keys, workers=8):
    """Re-render thumbnails without touching features, after a change to
    how thumbnails are drawn."""
    with ProcessPoolExecutor(workers, initializer=_init, initargs=(dataset.name,)) as pool:
        list(pool.map(_save_thumbnail, keys, chunksize=16))


def thumbnail(dataset, key):
    return Image.open(dataset.dir / "thumbs" / f"{key}.png")


# -- Distance ---------------------------------------------------------------------


def _standardised_distance(blocks):
    """Euclidean distance over z-scored features, each block weighted equally."""
    columns = []
    for values in blocks:
        values = np.asarray(values, dtype=np.float64)
        values = (values - values.mean(axis=0)) / (values.std(axis=0) + 1e-12)
        columns.append(values / np.sqrt(values.shape[1]))
    return squareform(pdist(np.hstack(columns)))


def distance(dataset, keys, groups=tuple(GROUPS)):
    """The distance between every pair of runs: the root-mean-square of each
    group's distance, after scaling each to unit median."""
    loaded = [np.load(dataset.dir / "features" / f"{key}.npz") for key in keys]
    parts = []
    for group in groups:
        d = _standardised_distance([[run[block] for run in loaded] for block in GROUPS[group]])
        parts.append(d / np.median(d[np.triu_indices_from(d, 1)]))
    return np.sqrt(np.mean([d**2 for d in parts], axis=0))
