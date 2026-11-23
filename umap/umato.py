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
from sklearn.neighbors import KDTree
from tqdm.auto import tqdm

import umap.distances as dist
from umap.layouts import rdist
from umap.umap_ import INT32_MAX, INT32_MIN, init_transform
from umap.utils import FASTMATH_FLAGS, make_epochs_per_sample, tau_rand_int, ts

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
DEFAULT_LOCAL_EPOCHS = 50
# The local phase of the reference implementation uses a learning rate of
# 0.01 and a repulsion of 0.1. UMAP's learning_rate and repulsion_strength
# multiply these values, so their default of 1.0 matches the reference.
LOCAL_LEARNING_RATE = 0.01
LOCAL_REPULSION = 0.1
# Hubs move by this fraction of the step that other points take, so the
# skeleton from the global phase changes slowly during the local phase.
HUB_STEP_SCALE = 0.1


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


def local_graph(graph, hub_info, n_epochs):
    """The graph that the local phase of UMATO optimizes.

    Only hubs and expanded points take part, so the graph is UMAP's fuzzy
    graph restricted to them. Because the set is closed under the kNN
    relation, these rows are exactly the graph that the reference
    implementation rebuilds from the same neighbors. Edges towards a hub get
    weight 1 and edges from a hub are removed, so hubs only move when an
    attached point pulls them. Weak edges are pruned as in UMAP.

    Returns
    -------
    members: array of int64
        Indices of the hubs and expanded points; the graph uses positions in
        this array.

    graph: scipy.sparse.coo_matrix
        The weighted local graph.
    """
    members = np.flatnonzero(hub_info > OUTLIER)
    info = hub_info[members]
    subgraph = scipy.sparse.csr_matrix(graph)[members][:, members].tocoo()
    subgraph.sum_duplicates()
    weights = subgraph.data.astype(np.float32)
    weights[info[subgraph.col] == HUB] = 1.0
    weights[info[subgraph.row] == HUB] = 0.0
    if weights.size > 0:
        weights[weights < weights.max() / float(n_epochs)] = 0.0
    result = scipy.sparse.coo_matrix(
        (weights, (subgraph.row, subgraph.col)), shape=subgraph.shape
    )
    result.eliminate_zeros()
    return members, result


@numba.njit(inline="always")
def _clip(value, cutoff):
    if value > cutoff:
        return cutoff
    if value < -cutoff:
        return -cutoff
    return value


def _local_layout_single_epoch(
    embedding,
    head,
    tail,
    is_hub,
    n_vertices,
    epochs_per_sample,
    a,
    b,
    rng_state,
    gamma,
    hub_step_scale,
    dim,
    alpha,
    epochs_per_negative_sample,
    epoch_of_next_negative_sample,
    epoch_of_next_sample,
    n,
):
    for i in numba.prange(epochs_per_sample.shape[0]):
        if epoch_of_next_sample[i] <= n:
            j = head[i]
            k = tail[i]
            current = embedding[j]
            other = embedding[k]

            dist_squared = rdist(current, other)
            if dist_squared > 0.0:
                grad_coeff = -2.0 * a * b * pow(dist_squared, b - 1.0)
                grad_coeff /= a * pow(dist_squared, b) + 1.0
            else:
                grad_coeff = 0.0

            other_scale = hub_step_scale if is_hub[k] else 1.0
            for d in range(dim):
                grad_d = _clip(grad_coeff * (current[d] - other[d]), 10.0)
                current[d] += grad_d * alpha
                other[d] += -grad_d * alpha * other_scale

            epoch_of_next_sample[i] += epochs_per_sample[i]

            n_neg_samples = int(
                (n - epoch_of_next_negative_sample[i]) / epochs_per_negative_sample[i]
            )
            for p in range(n_neg_samples):
                k = tau_rand_int(rng_state) % n_vertices
                other = embedding[k]
                dist_squared = rdist(current, other)

                if dist_squared > 0.0:
                    grad_coeff = 2.0 * b
                    grad_coeff /= (0.001 + dist_squared) * (
                        a * pow(dist_squared, b) + 1
                    )
                elif j == k:
                    continue
                else:
                    grad_coeff = 0.0

                for d in range(dim):
                    if grad_coeff > 0.0:
                        grad_d = _clip(grad_coeff * (current[d] - other[d]), 10.0)
                    else:
                        grad_d = 10.0
                    current[d] += grad_d * alpha * gamma

            epoch_of_next_negative_sample[i] += (
                n_neg_samples * epochs_per_negative_sample[i]
            )


_local_layout_single_epoch_serial = numba.njit(
    _local_layout_single_epoch, fastmath=FASTMATH_FLAGS
)
_local_layout_single_epoch_parallel = numba.njit(
    _local_layout_single_epoch, fastmath=FASTMATH_FLAGS, parallel=True
)


def optimize_local_layout(
    embedding,
    head,
    tail,
    is_hub,
    n_epochs,
    epochs_per_sample,
    a,
    b,
    rng_state,
    gamma=LOCAL_REPULSION,
    learning_rate=LOCAL_LEARNING_RATE,
    negative_sample_rate=5,
    hub_step_scale=HUB_STEP_SCALE,
    parallel=False,
    verbose=False,
    tqdm_kwds=None,
):
    """Refine local neighborhoods: the local phase of UMATO.

    This is UMAP's original stochastic gradient descent with the changes of
    the reference implementation: when an edge pulls a hub, the hub moves by
    ``hub_step_scale`` of the step, repulsion is weighted by ``gamma``, and
    negative samples are drawn only from the points being optimized. The
    learning rate decays linearly. ``embedding`` is updated in place.
    """
    if negative_sample_rate <= 0:
        raise ValueError("negative_sample_rate must be greater than 0 for UMATO")
    dim = embedding.shape[1]
    n_vertices = embedding.shape[0]
    epochs_per_negative_sample = epochs_per_sample / negative_sample_rate
    epoch_of_next_negative_sample = epochs_per_negative_sample.copy()
    epoch_of_next_sample = epochs_per_sample.copy()
    is_hub = np.ascontiguousarray(is_hub, dtype=np.bool_)

    if parallel:
        epoch_fn = _local_layout_single_epoch_parallel
    else:
        epoch_fn = _local_layout_single_epoch_serial

    tqdm_kwds = dict(tqdm_kwds or {})
    tqdm_kwds.setdefault("disable", not verbose)
    alpha = learning_rate
    for n in tqdm(range(n_epochs), **tqdm_kwds):
        epoch_fn(
            embedding,
            head,
            tail,
            is_hub,
            n_vertices,
            epochs_per_sample,
            a,
            b,
            rng_state,
            gamma,
            hub_step_scale,
            dim,
            alpha,
            epochs_per_negative_sample,
            epoch_of_next_negative_sample,
            epoch_of_next_sample,
            n,
        )
        alpha = learning_rate * (1.0 - (float(n) / float(n_epochs)))
    return embedding


def exact_knn(data, n_neighbors, metric, metric_kwds, chunk_size=1024):
    """Exact nearest neighbors of every point, nearest first.

    UMAP does not keep the neighbor arrays of small datasets, whose graph it
    builds from a full distance matrix, so UMATO recomputes them here.
    """
    n_samples = data.shape[0]
    indices = np.empty((n_samples, n_neighbors), dtype=np.int64)
    distances = np.empty((n_samples, n_neighbors), dtype=np.float32)
    if metric != "precomputed":
        dense_data = dense_rows(data, np.arange(n_samples))
    for start in range(0, n_samples, chunk_size):
        rows = np.arange(start, min(start + chunk_size, n_samples))
        if metric == "precomputed":
            block = data[rows]
            if scipy.sparse.issparse(block):
                block = block.toarray().astype(np.float64)
                block[block == 0.0] = np.inf
                block[np.arange(len(rows)), rows] = 0.0
            else:
                block = np.asarray(block, dtype=np.float64)
        else:
            block = np.asarray(
                pairwise_input_distances(
                    dense_data[rows], dense_data, metric, metric_kwds
                ),
                dtype=np.float64,
            )
        nearest = np.argpartition(block, n_neighbors - 1, axis=1)[:, :n_neighbors]
        nearest_dists = np.take_along_axis(block, nearest, axis=1)
        order = np.argsort(nearest_dists, axis=1, kind="stable")
        indices[rows] = np.take_along_axis(nearest, order, axis=1)
        distances[rows] = np.take_along_axis(nearest_dists, order, axis=1)
    return indices, distances


def umato_embedding(
    data,
    graph,
    knn_indices,
    n_components,
    a,
    b,
    n_epochs,
    init,
    random_state,
    metric,
    metric_kwds,
    n_hubs=None,
    global_epochs=None,
    learning_rate=1.0,
    repulsion_strength=1.0,
    negative_sample_rate=5,
    optimizer=None,
    parallel=False,
    verbose=False,
    tqdm_kwds=None,
):
    """Embed the data with UMATO, reusing UMAP's neighbors and fuzzy graph.

    Parameters
    ----------
    data: array or sparse matrix of shape (n_samples, n_features)
        The input data, or a distance matrix if ``metric="precomputed"``.

    graph: sparse matrix of shape (n_samples, n_samples)
        UMAP's fuzzy simplicial set of the data.

    knn_indices: array of shape (n_samples, n_neighbors)
        The nearest neighbors the graph was built from.

    n_components: int
        Dimension of the embedding.

    a, b: float
        UMAP's curve parameters (from ``min_dist`` and ``spread``).

    n_epochs: int or None
        Epochs of the local phase; None uses 50.

    init: str or array
        Initial hub layout: "pca", "random", "spectral", or an array with a
        row for every point.

    random_state: numpy RandomState

    metric: callable or "precomputed"
        A metric that works on dense rows of ``data``.

    metric_kwds: dict
        Arguments of the metric.

    n_hubs: int or None
        Target number of hubs; None uses :func:`default_n_hubs`.

    global_epochs: int or None
        Epochs of the global phase; None uses 100.

    learning_rate, repulsion_strength: float
        Multipliers of the reference local learning rate (0.01) and
        repulsion (0.1).

    negative_sample_rate: int
        Negative samples per positive sample in the local phase.

    optimizer: str or None
        None or "compatibility" uses the local update of the reference
        implementation.

    parallel, verbose: bool

    tqdm_kwds: dict or None

    Returns
    -------
    embedding: array of shape (n_samples, n_components)

    aux_data: dict
        ``hubs`` (indices of the hubs) and ``hub_info`` (the role of every
        point: 2 hub, 1 expanded, 0 outlier).
    """
    n_samples = graph.shape[0]
    if n_epochs is None:
        n_epochs = DEFAULT_LOCAL_EPOCHS
    if global_epochs is None:
        global_epochs = DEFAULT_GLOBAL_EPOCHS
    if n_hubs is None:
        n_hubs = default_n_hubs(n_samples)
    if optimizer not in (None, "compatibility"):
        raise NotImplementedError(
            f"optimizer={optimizer!r} is not supported for UMATO's local phase"
        )

    if verbose:
        print(ts(), "UMATO: selecting hubs")
    hubs, _ = select_hubs(knn_indices, n_hubs)
    hub_info = expand_hubs(knn_indices, hubs)
    if verbose:
        print(
            ts(),
            f"UMATO: {len(hubs)} hubs, {int(np.sum(hub_info == EXPANDED))} "
            f"expanded points, {int(np.sum(hub_info == OUTLIER))} outliers",
        )
        print(ts(), "UMATO: global phase")
    hub_layout = global_hub_layout(
        data,
        hubs,
        n_components,
        a,
        b,
        metric,
        metric_kwds,
        init,
        global_epochs,
        random_state,
    )
    embedding = embed_from_hubs(
        data, hub_layout, hubs, hub_info, metric, metric_kwds, random_state
    )

    if verbose:
        print(ts(), "UMATO: local phase")
    members, local = local_graph(graph, hub_info, n_epochs)
    layout = embedding[members]
    if np.unique(layout, axis=0).shape[0] < layout.shape[0]:
        nn_distance = KDTree(layout).query(layout, k=2)[0][:, 1].mean()
        layout = layout + random_state.normal(
            scale=0.001 * nn_distance, size=layout.shape
        )
    lower = layout.min(axis=0)
    span = layout.max(axis=0) - lower
    span[span == 0.0] = 1.0
    layout = np.ascontiguousarray(10.0 * (layout - lower) / span, dtype=np.float32)

    rng_state = random_state.randint(INT32_MIN, INT32_MAX, 3).astype(np.int64)
    optimize_local_layout(
        layout,
        local.row,
        local.col,
        hub_info[members] == HUB,
        n_epochs,
        make_epochs_per_sample(local.data, n_epochs),
        a,
        b,
        rng_state,
        gamma=LOCAL_REPULSION * repulsion_strength,
        learning_rate=LOCAL_LEARNING_RATE * learning_rate,
        negative_sample_rate=negative_sample_rate,
        parallel=parallel,
        verbose=verbose,
        tqdm_kwds=tqdm_kwds,
    )

    embedding = np.zeros((n_samples, n_components), dtype=np.float32)
    embedding[members] = layout
    embedding = place_outliers(embedding, hub_info, knn_indices, random_state)
    return embedding, {"hubs": hubs, "hub_info": hub_info}


def _is_positive_int(value, minimum):
    return (
        isinstance(value, (int, np.integer))
        and not isinstance(value, (bool, np.bool_))
        and value >= minimum
    )


def validate_umato_parameters(model):
    """Check the parameters of a :class:`umap.UMAP` model with ``umato=True``."""
    from umap.umap_ import UMAP

    if not isinstance(model.umato, (bool, np.bool_)):
        raise ValueError("umato must be True or False")
    if type(model)._fit_embed_data is not UMAP._fit_embed_data:
        raise ValueError(f"umato=True is not supported by {type(model).__name__}")
    if model.umato_n_hubs is not None and not _is_positive_int(model.umato_n_hubs, 2):
        raise ValueError("umato_n_hubs must be None or an integer of at least 2")
    if model.umato_global_epochs is not None and not _is_positive_int(
        model.umato_global_epochs, 1
    ):
        raise ValueError("umato_global_epochs must be None or a positive integer")
    if model.n_epochs_list is not None:
        raise ValueError("A list of n_epochs is not supported with umato=True")
    if model.n_epochs is not None and model.n_epochs < 1:
        raise ValueError("n_epochs must be a positive integer with umato=True")
    if model.densmap or model.output_dens:
        raise ValueError("umato=True cannot be combined with densmap or output_dens")
    if model.output_metric not in ("euclidean", "l2"):
        raise ValueError("umato=True requires a Euclidean output_metric")
    if model.negative_sample_rate <= 0:
        raise ValueError("negative_sample_rate must be greater than 0 for UMATO")
    if isinstance(model.init, str) and model.init not in (
        "auto",
        "pca",
        "random",
        "spectral",
    ):
        raise ValueError(
            "With umato=True, init must be 'auto', 'pca', 'random', 'spectral' "
            "or an array"
        )
    if model.optimizer not in (None, "compatibility"):
        raise NotImplementedError(
            f"optimizer={model.optimizer!r} is not yet supported with umato=True"
        )


def umato_fit_embed_data(model, X, n_epochs, init, random_state):
    """Lay out a fitted :class:`umap.UMAP` model with UMATO.

    This replaces :func:`umap.umap_.simplicial_set_embedding` when
    ``umato=True``, reusing the neighbors and the fuzzy graph of ``model``.
    It also sets the fitted attributes ``umato_hubs_`` and
    ``umato_hub_info_`` (with indices into the uniqued data if
    ``unique=True``).
    """
    if model._supervised:
        raise ValueError("Supervised dimension reduction is not supported with UMATO")
    n_samples = X.shape[0]
    n_hubs = model.umato_n_hubs
    if n_hubs is None:
        n_hubs = default_n_hubs(n_samples)
    elif n_hubs >= n_samples:
        raise ValueError(
            f"umato_n_hubs ({n_hubs}) must be smaller than the number of "
            f"samples ({n_samples})"
        )

    metric = resolve_dense_metric(
        model._input_distance_func, model.metric, model._sparse_data
    )
    if getattr(model, "_small_data", False):
        knn_indices, knn_dists = exact_knn(
            X, model._n_neighbors, metric, model._metric_kwds
        )
        knn_indices[knn_dists >= model._disconnection_distance] = -1
    else:
        knn_indices = model._knn_indices

    if isinstance(model.init, str):
        hub_init = "pca" if model.init == "auto" else model.init
    else:
        hub_init = init

    embedding, aux_data = umato_embedding(
        X,
        model.graph_,
        knn_indices,
        model.n_components,
        model._a,
        model._b,
        n_epochs,
        hub_init,
        random_state,
        metric,
        model._metric_kwds,
        n_hubs=n_hubs,
        global_epochs=model.umato_global_epochs,
        learning_rate=model.learning_rate,
        repulsion_strength=model.repulsion_strength,
        negative_sample_rate=model.negative_sample_rate,
        optimizer=model.optimizer,
        parallel=model.random_state is None,
        verbose=model.verbose,
        tqdm_kwds=model.tqdm_kwds,
    )
    model.umato_hubs_ = aux_data.pop("hubs")
    model.umato_hub_info_ = aux_data.pop("hub_info")
    return embedding, aux_data
