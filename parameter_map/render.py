"""Alignment, framing and thumbnail rendering of 2D embeddings."""

import numpy as np

# The frame covers all but the most extreme 0.1% of points on each side, but
# reaches no further from the median than FRAME_REACH times the distance to
# the 5th or 95th percentile.
FRAME_REACH = 1.5
# Gap shortening, thumbnails only: along each axis, a band of at least
# GAP_BINS-ths of the range holding at most GAP_EMPTY of the points is
# shortened to at most GAP_CAP of the occupied length.
GAP_BINS = 256
GAP_EMPTY = 2e-4
GAP_CAP = 0.08
# Thumbnail shading (datashader): how counts map to opacity, and the least
# opacity given to a pixel with any points.
SHADE_HOW = "eq_hist"
MIN_ALPHA = 40


def align(embedding, reference):
    """Rotate or reflect ``embedding`` (centred, not scaled) to best match
    ``reference``, so that thumbnails share an orientation."""
    x = np.asarray(embedding, dtype=np.float64)
    ref = np.asarray(reference, dtype=np.float64)
    x, ref = x - x.mean(axis=0), ref - ref.mean(axis=0)
    u, _, vt = np.linalg.svd(x.T @ ref)
    return x @ (u @ vt)


def frame(embedding):
    """Return (centre, extent) of the square frame to draw an embedding in.

    A plot with a compact body and a halo of stray points is framed on its
    body, not its strays (see ``FRAME_REACH``).
    """
    q = np.percentile(embedding, [0.1, 5, 50, 95, 99.9], axis=0)
    lo = np.maximum(q[0], q[2] - FRAME_REACH * (q[2] - q[1]))
    hi = np.minimum(q[4], q[2] + FRAME_REACH * (q[3] - q[2]))
    return (lo + hi) / 2, max((hi - lo).max(), 1e-12)


def _shorten_axis(x):
    lo, hi = np.percentile(x, [0.1, 99.9])
    if hi <= lo:
        return x
    edges = np.linspace(lo, hi, GAP_BINS + 1)
    empty = np.histogram(np.clip(x, lo, hi), edges)[0] <= GAP_EMPTY * len(x)
    width = (hi - lo) / GAP_BINS
    cap = GAP_CAP * (~empty).sum() * width
    new = np.full(GAP_BINS, width)
    i = 0
    while i < GAP_BINS:
        j = i
        while j < GAP_BINS and empty[j]:
            j += 1
        # Only bands between occupied stretches; never the ends.
        if 0 < i < j < GAP_BINS and (j - i) * width > cap:
            new[i:j] = cap / (j - i)
        i = max(j, i + 1)
    new_edges = np.concatenate([[lo], lo + np.cumsum(new)])
    out = np.interp(x, edges, new_edges)
    out[x < lo] = x[x < lo]
    out[x > hi] = x[x > hi] - (hi - new_edges[-1])
    return out


def shorten_gaps(embedding):
    """Shorten wide, empty bands between clusters, each axis separately.

    Distances between disconnected pieces of a UMAP layout carry little
    meaning, and one far-off cluster would otherwise shrink the rest of the
    plot to a sliver. Each piece only slides: shapes are unchanged.
    """
    embedding = np.asarray(embedding, dtype=np.float64)
    return np.column_stack([_shorten_axis(embedding[:, d]) for d in range(embedding.shape[1])])


def thumbnail(embedding, labels, palette, size, margin=0.04):
    """Render a class-coloured density image as a (size, size, 3) uint8 array.

    Drawn with datashader: points are counted per class in each pixel,
    coloured by mixing the class colours, and given opacity by histogram
    equalisation of the counts; where the plot is sparse, points are drawn
    as round dots three pixels across. A sparse dataset looks pale: that is
    how its points lie. Wide empty gaps are first shortened, then the
    embedding is fitted to its ``frame`` at equal aspect; points outside it
    are not drawn. ``palette`` is an (n_classes, 3) array of colours in 0-255.
    """
    import datashader as ds
    import datashader.transfer_functions as tf
    import pandas as pd

    embedding = shorten_gaps(embedding)
    centre, extent = frame(embedding)
    half = extent / 2 / (1 - 2 * margin)
    points = pd.DataFrame(
        {
            "x": embedding[:, 0],
            "y": embedding[:, 1],
            "label": pd.Categorical(labels, categories=range(len(palette))),
        }
    )
    canvas = ds.Canvas(
        plot_width=size,
        plot_height=size,
        x_range=(centre[0] - half, centre[0] + half),
        y_range=(centre[1] - half, centre[1] + half),
    )
    counts = canvas.points(points, "x", "y", ds.by("label", ds.count()))
    colours = ["#" + "".join(f"{int(v):02x}" for v in colour) for colour in palette]
    image = tf.shade(counts, color_key=colours, how=SHADE_HOW, min_alpha=MIN_ALPHA)
    image = tf.dynspread(image, threshold=0.5, max_px=1, shape="circle")
    image = tf.set_background(image, "white")
    return np.asarray(image.to_pil().convert("RGB"))
