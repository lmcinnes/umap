"""Arrange the runs so that runs that look alike are near each other.

A UMAP of the runs themselves, under the distance between them (the
"meta-UMAP"), gives the scatter view. ShuffleSnap snaps that layout onto a
dense square grid, and swapping tiles then refines the grid directly against
the distance. Zoom level ``z`` of the map shows a 2^z x 2^z grid, each cell
standing for a square block of runs and showing one of them.
"""

import warnings

import numba
import numpy as np

META_N_NEIGHBORS = 60
META_MIN_DIST = 0.1
META_SEED = 0
# Swap refinement: how far a tile's neighborhood reaches, in cells, and how
# quickly its weight falls off.
REFINE_RADIUS = 2
REFINE_FALLOFF = 1.0
REFINE_SWEEPS = 50
# Levels coarser than this are not built: a tile standing for too many runs
# says little.
TOP_LEVEL = 2


def meta_layout(distance):
    import umap

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return umap.UMAP(
            metric="precomputed",
            n_neighbors=META_N_NEIGHBORS,
            min_dist=META_MIN_DIST,
            random_state=META_SEED,
        ).fit_transform(distance)


def snap(layout, side):
    """Assign each run a (column, row) cell on a side x side grid."""
    import shufflesnap

    grid = shufflesnap.Grid(np.ones((side, side), dtype=bool))
    return np.asarray(shufflesnap.assign(layout, grid, preset="quality").xy)


@numba.njit(cache=True)
def _cost_at(grid, distance, weights, radius, x, y, run, skip_x, skip_y):
    side, total = grid.shape[0], 0.0
    for dx in range(-radius, radius + 1):
        for dy in range(-radius, radius + 1):
            u, v = x + dx, y + dy
            if (dx == 0 and dy == 0) or u < 0 or v < 0 or u >= side or v >= side:
                continue
            if u == skip_x and v == skip_y:
                continue
            total += weights[dx + radius, dy + radius] * distance[run, grid[u, v]]
    return total


@numba.njit(cache=True)
def _sweep(grid, distance, weights, radius, order):
    """Try every pair of cells once; return the number of swaps made."""
    side, swaps = grid.shape[0], 0
    for a in order:
        ax, ay = a // side, a % side
        for b in order:
            if b <= a:
                continue
            bx, by = b // side, b % side
            ra, rb = grid[ax, ay], grid[bx, by]
            before = _cost_at(grid, distance, weights, radius, ax, ay, ra, bx, by) + _cost_at(
                grid, distance, weights, radius, bx, by, rb, ax, ay
            )
            after = _cost_at(grid, distance, weights, radius, ax, ay, rb, bx, by) + _cost_at(
                grid, distance, weights, radius, bx, by, ra, ax, ay
            )
            if after < before - 1e-12:
                grid[ax, ay], grid[bx, by] = rb, ra
                swaps += 1
    return swaps


def refine(cells, distance, seed=0):
    """Swap tiles whenever that lowers the weighted distance between each
    tile and those around it, until no swap helps; return the new cells."""
    side = round(np.sqrt(len(cells)))
    grid = np.empty((side, side), dtype=np.int64)
    grid[cells[:, 0], cells[:, 1]] = np.arange(len(cells))
    offsets = np.arange(-REFINE_RADIUS, REFINE_RADIUS + 1)
    separation = np.hypot(offsets[:, None], offsets[None, :])
    weights = np.exp(-((separation / REFINE_FALLOFF) ** 2) / 2)
    rng = np.random.default_rng(seed)
    distance = np.ascontiguousarray(distance, dtype=np.float64)
    for _ in range(REFINE_SWEEPS):
        if _sweep(grid, distance, weights, REFINE_RADIUS, rng.permutation(side * side)) == 0:
            break
    refined = np.empty_like(cells)
    xs, ys = np.indices((side, side))
    refined[grid.ravel()] = np.column_stack([xs.ravel(), ys.ravel()])
    return refined


def placement(distance, keys, cache):
    """The meta-UMAP layout and the refined grid cells of the runs.

    Cached in ``cache`` for this exact list of runs, since these dominate
    the build and do not depend on how thumbnails are drawn. Delete the
    file to recompute.
    """
    if cache.exists():
        saved = np.load(cache)
        if saved["keys"].tolist() == keys:
            return saved["layout"], saved["cells"]
    side = round(np.sqrt(len(keys)))
    layout = meta_layout(distance)
    cells = refine(snap(layout, side), distance)
    np.savez(cache, keys=np.array(keys), layout=layout, cells=cells)
    return layout, cells


def scores(cells, distance):
    """How well the grid keeps like with like.

    adjacent: mean distance between runs in cells sharing an edge, relative
              to that between random runs (lower is better; 1 is no structure)
    recall:   the share of a run's 8 nearest runs lying within two cells of it
    """
    side = cells.max() + 1
    grid = np.empty((side, side), dtype=np.int64)
    grid[cells[:, 0], cells[:, 1]] = np.arange(len(cells))
    adjacent = np.concatenate(
        [distance[grid[:-1, :], grid[1:, :]].ravel(), distance[grid[:, :-1], grid[:, 1:]].ravel()]
    )
    overall = distance[np.triu_indices_from(distance, 1)].mean()
    recall = []
    for run in range(len(cells)):
        row = distance[run].copy()
        row[run] = np.inf
        nearest = np.argpartition(row, 8)[:8]
        recall.append((np.abs(cells[nearest] - cells[run]).max(axis=1) <= 2).mean())
    return {"adjacent": adjacent.mean() / overall, "recall": float(np.mean(recall))}


def pyramid(cells, distance, top_level=TOP_LEVEL):
    """Return {level: {(x, y): (representative run, member runs)}}.

    The deepest level has one run per cell. Going up, each cell's
    representative is the one of its children's representatives closest,
    in total, to every run in the block. Representatives are nested, so the
    picture being looked at survives zooming in.
    """
    levels = int(np.log2(cells.max() + 1))
    result = {levels: {(int(x), int(y)): (i, [i]) for i, (x, y) in enumerate(cells)}}
    for level in range(levels - 1, top_level - 1, -1):
        children, current = result[level + 1], {}
        for x in range(2**level):
            for y in range(2**level):
                kids = [children[(2 * x + dx, 2 * y + dy)] for dx in (0, 1) for dy in (0, 1)]
                members = [run for _, runs in kids for run in runs]
                candidates = [representative for representative, _ in kids]
                totals = distance[np.ix_(candidates, members)].sum(axis=1)
                current[(x, y)] = (candidates[int(np.argmin(totals))], members)
        result[level] = current
    return result


def unit_square(layout):
    """Scale a 2D layout into [0, 1] at equal aspect, centred."""
    lo, hi = layout.min(axis=0), layout.max(axis=0)
    return (layout - (lo + hi) / 2) / (hi - lo).max() + 0.5
