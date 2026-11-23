"""UMATO: Uniform Manifold Approximation with Two-phase Optimization.

UMATO first lays out a small set of representative points (hubs) using the
distances between all pairs of hubs, without neighborhood approximation. It
then embeds the remaining points around that skeleton and refines their local
neighborhoods while the hubs move more slowly than the other points.

This module reuses the nearest neighbors and the fuzzy graph that
:class:`umap.UMAP` has already computed, so enabling ``UMAP(umato=True)`` only
changes the layout step.

References
----------
Hyeon Jeon, Kwon Ko, Soohyun Lee, Jake Hyun, Taehyun Yang, Gyehun Go, Jaemin
Jo and Jinwook Seo. UMATO: Bridging Local and Global Structures for Reliable
Visual Analytics with Dimensionality Reduction. IEEE Transactions on
Visualization and Computer Graphics (2025). doi:10.1109/TVCG.2025.3602735

Reference implementation: https://github.com/hyungkwonko/umato
"""

import numba
import numpy as np

# Point roles stored in ``hub_info``.
OUTLIER = 0
EXPANDED = 1
HUB = 2

MAX_DEFAULT_HUBS = 300


def default_n_hubs(n_samples):
    """Default target number of hubs: a third of the points, at most 300."""
    return int(min(MAX_DEFAULT_HUBS, max(2, n_samples // 3), n_samples - 1))


def _neighbor_frequency_order(knn_indices):
    """Order points by how often they appear in the kNN lists (most first)."""
    n_samples = knn_indices.shape[0]
    flat = knn_indices.ravel()
    counts = np.bincount(flat[flat >= 0], minlength=n_samples)
    return np.argsort(-counts, kind="stable").astype(np.int64)


@numba.njit(cache=True)
def _partition_by_knn(order, knn_indices, group_size):
    """Split the points into disjoint groups of ``group_size`` points.

    Each group starts at the most frequent point that is still unassigned and
    grows by a breadth-first walk over the kNN graph. If the walk runs out of
    unassigned points the group is filled in ``order``.

    Returns the first point of every group and the group of every point.
    """
    n_samples = knn_indices.shape[0]
    n_neighbors = knn_indices.shape[1]
    remaining = np.ones(n_samples, dtype=np.bool_)
    groups = np.full(n_samples, -1, dtype=np.int64)
    seen = np.full(n_samples, -1, dtype=np.int64)
    queue = np.empty(n_samples, dtype=np.int64)
    sources = np.empty(n_samples, dtype=np.int64)

    n_groups = 0
    order_position = 0
    while True:
        while order_position < n_samples and not remaining[order[order_position]]:
            order_position += 1
        if order_position >= n_samples:
            break

        source = order[order_position]
        remaining[source] = False
        groups[source] = n_groups
        sources[n_groups] = source
        count = 1

        queue[0] = source
        seen[source] = n_groups
        queue_head = 0
        queue_tail = 1
        while queue_head < queue_tail and count < group_size:
            node = queue[queue_head]
            queue_head += 1
            for j in range(n_neighbors):
                neighbor = knn_indices[node, j]
                if neighbor < 0:
                    continue
                if seen[neighbor] != n_groups:
                    seen[neighbor] = n_groups
                    queue[queue_tail] = neighbor
                    queue_tail += 1
                if remaining[neighbor]:
                    remaining[neighbor] = False
                    groups[neighbor] = n_groups
                    count += 1
                    if count >= group_size:
                        break

        position = order_position
        while count < group_size and position < n_samples:
            candidate = order[position]
            if remaining[candidate]:
                remaining[candidate] = False
                groups[candidate] = n_groups
                count += 1
            position += 1

        n_groups += 1

    return sources[:n_groups].copy(), groups


def select_hubs(knn_indices, n_hubs):
    """Select representative points (hubs) from the kNN graph.

    The points are split into at most ``n_hubs`` disjoint groups of equal
    size, each grown from the most frequent remaining neighbor, and the first
    point of each group becomes a hub. This follows the kNN-based variant of
    the reference UMATO implementation, so it works for any input metric.

    Parameters
    ----------
    knn_indices: array of shape (n_samples, n_neighbors)
        Nearest neighbor indices; negative entries are ignored.

    n_hubs: int
        Target number of hubs. The actual number may be smaller because every
        group has the same size.

    Returns
    -------
    hubs: array of int64
        Indices of the hubs.

    groups: array of shape (n_samples,)
        The group of every point.
    """
    n_samples = knn_indices.shape[0]
    if not 2 <= n_hubs < n_samples:
        raise ValueError("n_hubs must be at least 2 and smaller than n_samples")
    knn_indices = np.ascontiguousarray(knn_indices, dtype=np.int64)
    group_size = int(np.ceil(n_samples / n_hubs))
    order = _neighbor_frequency_order(knn_indices)
    return _partition_by_knn(order, knn_indices, group_size)


@numba.njit(cache=True)
def _expand_hubs(knn_indices, hubs):
    n_samples = knn_indices.shape[0]
    hub_info = np.zeros(n_samples, dtype=np.int8)
    queue = np.empty(n_samples, dtype=np.int64)
    queue_tail = 0
    for hub in hubs:
        if hub_info[hub] == 0:
            hub_info[hub] = 2
            queue[queue_tail] = hub
            queue_tail += 1
    queue_head = 0
    while queue_head < queue_tail:
        node = queue[queue_head]
        queue_head += 1
        for j in range(knn_indices.shape[1]):
            neighbor = knn_indices[node, j]
            if neighbor >= 0 and hub_info[neighbor] == 0:
                hub_info[neighbor] = 1
                queue[queue_tail] = neighbor
                queue_tail += 1
    return hub_info


def expand_hubs(knn_indices, hubs):
    """Label every point as a hub, an expanded point or an outlier.

    Starting from the hubs, the set is grown along the directed kNN graph
    until it no longer changes, so every neighbor of a point in the set is in
    the set as well. Points that no member of the set lists as a neighbor are
    outliers; UMATO places them only after the layout is optimized.

    Returns
    -------
    hub_info: array of int8, shape (n_samples,)
        ``HUB`` (2), ``EXPANDED`` (1) or ``OUTLIER`` (0) for every point.
    """
    knn_indices = np.ascontiguousarray(knn_indices, dtype=np.int64)
    hubs = np.asarray(hubs, dtype=np.int64)
    return _expand_hubs(knn_indices, hubs)
