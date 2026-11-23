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
import scipy.sparse
from sklearn.decomposition import PCA
from sklearn.manifold import SpectralEmbedding

import umap.distances as dist
from umap.umap_ import init_transform

# Point roles stored in ``hub_info``.
OUTLIER = 0
EXPANDED = 1
HUB = 2

MAX_DEFAULT_HUBS = 300
DEFAULT_GLOBAL_EPOCHS = 100
GLOBAL_LEARNING_RATE = 0.0065
N_NEAREST_HUBS = 10
EXPANDED_NOISE = 0.05
OUTLIER_NOISE = 0.02


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


def dense_rows(data, rows):
    """Return ``data[rows]`` as a dense, C-contiguous float32 array."""
    subset = data[rows]
    if scipy.sparse.issparse(subset):
        subset = subset.toarray()
    return np.ascontiguousarray(subset, dtype=np.float32)


def resolve_dense_metric(metric, metric_name, sparse_data):
    """Return a metric that works on dense rows of the input data.

    ``metric`` is the function UMAP uses for the input data. For sparse input
    that function expects sparse rows, so the dense function of the same name
    is used instead; the rows UMATO compares are densified first.
    """
    if not sparse_data or metric == "precomputed":
        return metric
    if callable(metric_name):
        return metric_name
    if metric_name in dist.named_distances:
        return dist.named_distances[metric_name]
    from pynndescent.distances import named_distances as pynn_named_distances

    if metric_name in pynn_named_distances:
        return pynn_named_distances[metric_name]
    raise ValueError(f"UMATO does not support metric {metric_name!r} for sparse data")


def pairwise_input_distances(X, Y, metric, metric_kwds):
    """Distances between the rows of ``X`` and ``Y`` under the input metric."""
    try:
        return dist.numba_aware_pairwise_distances(X, metric=metric, Y=Y, **metric_kwds)
    except (ValueError, TypeError):
        return dist.pairwise_special_metric(X, Y, metric=metric, kwds=metric_kwds)


def hub_distances(data, hubs, metric, metric_kwds):
    """Input-space distances between every pair of hubs.

    With ``metric="precomputed"`` the distances are read from ``data``.
    Missing or infinite entries are replaced by the largest finite distance.
    """
    if metric == "precomputed":
        rows = data[hubs]
        if scipy.sparse.issparse(rows):
            distances = rows[:, hubs].toarray().astype(np.float64)
            missing = distances == 0.0
            np.fill_diagonal(missing, False)
            distances[missing] = np.inf
        else:
            distances = np.asarray(rows, dtype=np.float64)[:, hubs]
    else:
        hub_data = dense_rows(data, hubs)
        distances = np.asarray(
            pairwise_input_distances(hub_data, None, metric, metric_kwds),
            dtype=np.float64,
        )
    finite = np.isfinite(distances)
    if not np.all(finite):
        largest = distances[finite].max() if np.any(finite) else 1.0
        distances[~finite] = largest
    np.fill_diagonal(distances, 0.0)
    return distances


def initial_hub_layout(init, hub_features, distances, hubs, n_components, random_state):
    """Initial positions of the hubs, scaled so the largest coordinate is 1.

    ``init`` is ``"pca"``, ``"random"``, ``"spectral"`` or an array with a row
    for every point (only the rows of the hubs are used).
    """
    n_hubs = len(hubs)
    if isinstance(init, str) and init == "pca":
        n_pca = min(n_components, n_hubs, hub_features.shape[1])
        layout = PCA(n_components=n_pca, random_state=random_state).fit_transform(
            hub_features
        )
        if n_pca < n_components:
            scale = 0.05 * max(np.abs(layout).max(), 1e-12)
            extra = random_state.normal(
                scale=scale, size=(n_hubs, n_components - n_pca)
            )
            layout = np.hstack([layout, extra])
    elif isinstance(init, str) and init == "random":
        layout = random_state.normal(loc=0.0, scale=0.05, size=(n_hubs, n_components))
    elif isinstance(init, str) and init == "spectral":
        positive = distances[distances > 0]
        bandwidth = np.median(positive) if positive.size else 1.0
        affinity = np.exp(-((distances / bandwidth) ** 2))
        layout = SpectralEmbedding(
            n_components=n_components,
            affinity="precomputed",
            random_state=random_state,
        ).fit_transform(affinity)
    else:
        layout = np.asarray(init, dtype=np.float64)[hubs]

    layout = np.array(layout, dtype=np.float64)
    largest = np.abs(layout).max()
    if largest > 0.0:
        layout /= largest
    return layout


def optimize_global_layout(P, Z, a, b, alpha=GLOBAL_LEARNING_RATE, n_epochs=100):
    """Optimize the hub layout with full gradients over all pairs of hubs.

    This is the global phase of UMATO. ``P`` holds the input-space distances
    between hubs scaled to a maximum of 1, and ``Z`` the initial hub layout,
    which is updated in place and returned.
    """
    for _ in range(n_epochs):
        z_diff = Z[:, np.newaxis, :] - Z[np.newaxis, :, :]
        d_squared = np.sum(z_diff * z_diff, axis=2)
        d_inverse = (1.0 / (1.0 + a * d_squared**b))[:, :, np.newaxis]

        Q = 1.0 / (0.001 + d_squared)
        np.fill_diagonal(Q, 0.0)
        Q = (1.0 - P) @ Q
        Q /= np.sum(Q, axis=1, keepdims=True)

        grad = (2.0 * a * b * P * (1e-12 + d_squared) ** (b - 1.0) - 2.0 * b * Q)[
            :, :, np.newaxis
        ]
        Z -= alpha * np.sum(grad * z_diff * d_inverse, axis=1)
    return Z


def global_hub_layout(
    data,
    hubs,
    n_components,
    a,
    b,
    metric,
    metric_kwds,
    init,
    n_epochs,
    random_state,
):
    """Lay out the hubs: the global phase of UMATO.

    Returns the hub positions, with one row per entry of ``hubs``.
    """
    distances = hub_distances(data, hubs, metric, metric_kwds)
    if metric == "precomputed":
        hub_features = distances
    else:
        hub_features = dense_rows(data, hubs)
    Z = initial_hub_layout(
        init, hub_features, distances, hubs, n_components, random_state
    )
    largest = distances.max()
    P = distances / largest if largest > 0.0 else distances
    return optimize_global_layout(P, Z, a, b, n_epochs=n_epochs).astype(np.float32)


def nearest_hubs(data, points, hubs, metric, metric_kwds, n_nearest, chunk_size=4096):
    """For each point, the positions in ``hubs`` of its nearest hubs.

    Distances are measured in the input space, in chunks of points so that
    memory stays bounded for large datasets.
    """
    result = np.empty((len(points), n_nearest), dtype=np.int64)
    if metric != "precomputed":
        hub_data = dense_rows(data, hubs)
    for start in range(0, len(points), chunk_size):
        chunk = points[start : start + chunk_size]
        if metric == "precomputed":
            rows = data[chunk]
            if scipy.sparse.issparse(rows):
                distances = rows[:, hubs].toarray().astype(np.float64)
                distances[distances == 0.0] = np.inf
            else:
                distances = np.asarray(rows, dtype=np.float64)[:, hubs]
        else:
            distances = pairwise_input_distances(
                dense_rows(data, chunk), hub_data, metric, metric_kwds
            )
        nearest = np.argpartition(distances, n_nearest - 1, axis=1)[:, :n_nearest]
        result[start : start + len(chunk)] = nearest
    return result


def embed_from_hubs(
    data, hub_layout, hubs, hub_info, metric, metric_kwds, random_state
):
    """Initial embedding of the hubs and the expanded points.

    Hubs keep their position from the global phase. Each expanded point
    starts at the mean position of its nearest hubs in the input space, plus
    a little noise. Outliers are left at the origin and placed after the
    local phase (see :func:`place_outliers`).
    """
    n_samples = hub_info.shape[0]
    n_components = hub_layout.shape[1]
    hub_layout = np.ascontiguousarray(hub_layout, dtype=np.float32)
    embedding = np.zeros((n_samples, n_components), dtype=np.float32)
    embedding[hubs] = hub_layout

    expanded = np.flatnonzero(hub_info == EXPANDED)
    if expanded.size > 0:
        n_nearest = min(N_NEAREST_HUBS, len(hubs))
        nearest = nearest_hubs(data, expanded, hubs, metric, metric_kwds, n_nearest)
        weights = np.full(nearest.shape, 1.0 / n_nearest, dtype=np.float32)
        noise = random_state.normal(
            loc=0.0, scale=EXPANDED_NOISE, size=(expanded.size, n_components)
        )
        embedding[expanded] = init_transform(nearest, weights, hub_layout) + (
            noise / n_nearest
        ).astype(np.float32)
    return embedding


@numba.njit(cache=True)
def _place_outliers(embedding, assigned, knn_indices, noise, max_passes):
    n_samples, n_components = embedding.shape
    total = np.zeros(n_components, dtype=np.float64)
    for _ in range(max_passes):
        changed = 0
        for i in range(n_samples):
            if assigned[i]:
                continue
            count = 0
            total[:] = 0.0
            for j in range(knn_indices.shape[1]):
                neighbor = knn_indices[i, j]
                if neighbor >= 0 and assigned[neighbor]:
                    total += embedding[neighbor]
                    count += 1
            if count > 0:
                for d in range(n_components):
                    embedding[i, d] = total[d] / count + noise[i, d]
                assigned[i] = True
                changed += 1
        if changed == 0:
            break

    n_assigned = 0
    total[:] = 0.0
    for i in range(n_samples):
        if assigned[i]:
            total += embedding[i]
            n_assigned += 1
    for i in range(n_samples):
        if not assigned[i]:
            for d in range(n_components):
                embedding[i, d] = total[d] / max(n_assigned, 1) + noise[i, d]
    return embedding


def place_outliers(embedding, hub_info, knn_indices, random_state, max_passes=10):
    """Place the outliers next to their already embedded nearest neighbors.

    Each outlier moves to the mean position of its embedded kNN neighbors,
    plus a little noise. Outliers whose neighbors are all outliers are placed
    in later passes; any left after ``max_passes`` go to the mean position of
    all embedded points.
    """
    outliers = hub_info == OUTLIER
    if not np.any(outliers):
        return embedding
    noise = np.zeros(embedding.shape, dtype=np.float32)
    noise[outliers] = random_state.normal(
        loc=0.0, scale=OUTLIER_NOISE, size=(int(outliers.sum()), embedding.shape[1])
    )
    return _place_outliers(
        embedding,
        ~outliers,
        np.ascontiguousarray(knn_indices, dtype=np.int64),
        noise,
        max_passes,
    )
