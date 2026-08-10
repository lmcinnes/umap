import numpy as np
import numba
from scipy.sparse import csr_matrix, issparse
from sklearn.preprocessing import normalize
from sklearn.utils.extmath import randomized_svd

from umap.layouts import optimize_layout_euclidean
from umap.utils import tau_rand, tau_rand_int, ts

INT32_MIN = np.iinfo(np.int32).min + 1
INT32_MAX = np.iinfo(np.int32).max - 1


def _coarsen_graph(graph, labels, remove_diagonal=False):
    """Construct the partition map and fuzzy-union coarse graph."""
    reduction_map = csr_matrix(
        (np.ones(labels.shape[0]), labels, np.arange(labels.shape[0] + 1)),
        shape=(labels.shape[0], labels.max() + 1),
    )
    complement_graph = graph.astype(np.float64)
    complement_graph.data = np.log1p(-np.clip(complement_graph.data, 0.0, 1.0 - 1e-16))
    reduced_graph = reduction_map.T * complement_graph * reduction_map
    reduced_graph.data = 1.0 - np.exp(reduced_graph.data)
    reduced_graph.eliminate_zeros()
    if remove_diagonal:
        reduced_graph.setdiag(0.0)
        reduced_graph.eliminate_zeros()
    return reduction_map, reduced_graph.astype(np.float32)


def _initial_curve_parameters(a, b, curve_schedule):
    """Select the curve parameters used by the first coarse layout."""
    spec = _curve_schedule_spec(curve_schedule)
    return (
        np.power(a, spec["initial_a_exponent"]),
        np.power(b, spec["initial_b_exponent"]),
    )


def _next_curve_parameters(a, b, curve_schedule):
    """Select curve parameters for the next recursive coarse layout."""
    spec = _curve_schedule_spec(curve_schedule)
    return (
        np.power(a, spec["recursive_a_exponent"]),
        np.power(b, spec["recursive_b_exponent"]),
    )


def _curve_schedule_spec(curve_schedule):
    """Normalize legacy and experiment curve schedules into exponent specs."""
    if isinstance(curve_schedule, dict):
        return {
            "initial_a_exponent": float(curve_schedule.get("initial_a_exponent", 1.0)),
            "initial_b_exponent": float(curve_schedule.get("initial_b_exponent", 1.0)),
            "recursive_a_exponent": float(
                curve_schedule.get("recursive_a_exponent", 1.0)
            ),
            "recursive_b_exponent": float(
                curve_schedule.get("recursive_b_exponent", 1.0)
            ),
        }

    if curve_schedule == "original":
        return {
            "initial_a_exponent": 1.0,
            "initial_b_exponent": 1.0,
            "recursive_a_exponent": 1.0,
            "recursive_b_exponent": 1.0,
        }
    if curve_schedule == "strong_to_one":
        return {
            "initial_a_exponent": 1.0 / 4.0,
            "initial_b_exponent": 1.0 / 4.0,
            "recursive_a_exponent": 1.0 / 4.0,
            "recursive_b_exponent": 1.0 / 4.0,
        }
    raise ValueError(
        "curve_schedule must be 'strong_to_one', 'original', or an exponent spec dict"
    )


def _good_initialization(
    reduced_init, approx_n_parts, threshold, policy, coarsening_ratio=4
):
    """Resolve whether the conservative optimizer schedule should be used.

    Policies
    --------
    heuristic : activate when the coarse graph has enough parts to recurse further.
    available : activate whenever the recursive coordinates are finite with nonzero spread.
    always    : unconditionally activate; ignores coordinate quality entirely.
    """
    if policy == "heuristic":
        return approx_n_parts // coarsening_ratio > threshold
    if policy == "available":
        if reduced_init is None or not np.all(np.isfinite(reduced_init)):
            return False
        spread = np.quantile(reduced_init, 0.95, axis=0) - np.quantile(
            reduced_init, 0.05, axis=0
        )
        return bool(np.all(spread > 0.0))
    if policy == "always":
        return True
    if policy == "never":
        return False
    raise ValueError(
        "good_initialization_policy must be 'heuristic', 'available', 'always', or 'never'"
    )


def _scale_anchor(anchor):
    """Scale an anchor independently by axis into the historical [0, 10] range."""
    anchor = np.asarray(anchor, dtype=np.float32, order="C").copy()
    anchor -= anchor.min(axis=0)
    span = anchor.max(axis=0) - anchor.min(axis=0)
    span[span == 0.0] = 1.0
    anchor /= span
    anchor *= 10.0
    return anchor


def _resolve_negative_selection_range(n_vertices, mode, scale=1.0):
    """Resolve the recursive negative-selection range for coarse optimizations."""
    if mode == "coarse_n":
        return int(n_vertices)
    if mode == "fixed_200k":
        return int(200_000)
    if mode == "scaled_coarse":
        value = int(np.round(scale * n_vertices))
        value = max(1, value)
        return int(min(value, n_vertices))
    raise ValueError(
        "recursive_negative_selection_range_mode must be 'coarse_n', 'fixed_200k', or 'scaled_coarse'"
    )


def _resolve_depth_schedule(value, depth):
    """Resolve a scalar or per-depth mapping for recursive experiment controls."""
    if isinstance(value, dict):
        if depth in value:
            return value[depth]
        depth_key = str(depth)
        if depth_key in value:
            return value[depth_key]
        if "default" in value:
            return value["default"]
        raise ValueError(
            "depth schedule dict must contain the requested depth or 'default'"
        )
    return value


def _anchor_reference(
    graph,
    data,
    n_components,
    method,
    sample_size,
    random_state,
    verbose=False,
):
    """Build coordinates and a mask used to orient recursive layouts."""
    n_samples = data.shape[0]
    if method not in ("sampled_pca", "projected_pca", "full_pca"):
        raise ValueError(
            "anchor_method must be 'sampled_pca', 'projected_pca', or 'full_pca'"
        )

    sample_size = min(sample_size, n_samples)
    if sample_size < n_samples:
        sample = np.sort(
            random_state.choice(n_samples, size=sample_size, replace=False)
        )
        sample_mask = np.zeros(n_samples, dtype=np.bool_)
        sample_mask[sample] = True
        data_sample = data[sample]
    else:
        sample_mask = np.ones(n_samples, dtype=np.bool_)
        data_sample = data

    if issparse(data_sample):
        feature_mean = np.asarray(data_sample.mean(axis=0)).ravel()
        centered_sample = data_sample.toarray() - feature_mean
    else:
        data_sample = np.asarray(data_sample)
        finite = np.isfinite(data_sample)
        finite_counts = finite.sum(axis=0)
        finite_sums = np.where(finite, data_sample, 0.0).sum(axis=0)
        feature_mean = np.divide(
            finite_sums,
            finite_counts,
            out=np.zeros_like(finite_sums, dtype=np.float64),
            where=finite_counts > 0,
        )
        centered_sample = np.where(finite, data_sample, feature_mean) - feature_mean

    u, singular_values, components = randomized_svd(
        centered_sample,
        n_components=n_components,
        n_iter=1,
        n_oversamples=8,
        random_state=random_state,
    )
    if method == "sampled_pca":
        return _scale_anchor(u * singular_values), sample_mask

    if issparse(data):
        anchor = np.asarray(data @ components.T)
    else:
        projection_data = np.asarray(data)
        if not np.all(np.isfinite(projection_data)):
            projection_data = np.where(
                np.isfinite(projection_data), projection_data, feature_mean
            )
        anchor = projection_data @ components.T
    anchor -= feature_mean @ components.T
    return _scale_anchor(anchor), np.ones(n_samples, dtype=np.bool_)


def make_epochs_per_sample(weights, n_epochs):
    result = -1.0 * np.ones(weights.shape[0], dtype=np.float64)
    n_samples = n_epochs * (weights / weights.max())
    result[n_samples > 0] = float(n_epochs) / np.float64(n_samples[n_samples > 0])
    return result


@numba.njit("f8[:, ::1](f4[:, ::1], f4[:, ::1])", cache=True)
def procrustes_align(e1: np.ndarray, e2: np.ndarray) -> np.ndarray:
    e1_shift = e1 - np.sum(e1, axis=0) / e1.shape[0]
    e2_shift = e2 - np.sum(e2, axis=0) / e2.shape[0]
    e1_scale_factor = np.sqrt(np.mean(e1_shift**2))
    e2_scale_factor = np.sqrt(np.mean(e2_shift**2))
    e1_scaled = e1_shift / e1_scale_factor
    e2_scaled = e2_shift / e2_scale_factor
    covariance = e2_scaled.T @ e1_scaled
    u, s, vh = np.linalg.svd(covariance)
    if np.linalg.det(u @ vh) < 0:
        u[:, -1] *= -1
    rotation = u @ vh
    return rotation


@numba.njit(fastmath=True, parallel=True, cache=True)
def label_prop_iteration(
    indptr,
    indices,
    data,
    labels,
    rng_state,
):
    n_rows = indptr.shape[0] - 1
    result = labels.copy()

    for i in numba.prange(n_rows):
        current_l = labels[i]
        if current_l >= 0:
            continue
        # Create a local rng state for this iteration
        local_rng_state = rng_state + i
        votes = {}
        for k in range(indptr[i], indptr[i + 1]):
            j = indices[k]
            l = labels[j]
            if l in votes:
                votes[l] += data[k]
            else:
                votes[l] = data[k]

        max_vote = 1
        tie_count = 1
        for l in votes:
            if l == -1:
                continue
            elif votes[l] > max_vote:
                max_vote = votes[l]
                result[i] = l
                tie_count = 1
            elif votes[l] == max_vote:
                tie_count += 1
                if current_l == -1:
                    result[i] = l
                elif tau_rand(local_rng_state) < 1.0 / tie_count:
                    result[i] = l
            else:
                continue

    return result


@numba.njit(cache=True)
def label_outliers(indptr, indices, labels, rng_state):
    n_rows = indptr.shape[0] - 1
    max_label = labels.max()
    num_labels = max(max_label + 1, 1)

    for i in range(n_rows):
        # Create a local rng state for this iteration
        local_rng_state = rng_state + i
        if labels[i] < 0:

            node_queue = [i]
            unlabelled = True
            n_iter = 0

            while unlabelled and n_iter < 100 and len(node_queue) > 0:

                n_iter += 1
                current_node = node_queue.pop()
                for k in range(indptr[current_node], indptr[current_node + 1]):
                    j = indices[k]
                    if labels[j] >= 0:
                        labels[i] = labels[j]
                        unlabelled = False
                        break
                    else:
                        node_queue.append(j)

            if unlabelled:
                labels[i] = tau_rand_int(local_rng_state) % num_labels

    return labels


@numba.njit(cache=True)
def remap_labels(labels):
    mapping = {}
    unique_labels = np.unique(labels)
    if unique_labels[0] == -1:
        unique_labels = unique_labels[1:]
    for i, l in enumerate(unique_labels):
        mapping[l] = i
    next_label = i + 1
    for i in range(labels.shape[0]):
        if labels[i] < 0:
            labels[i] = next_label
            next_label += 1
        else:
            labels[i] = mapping[labels[i]]

    return labels


@numba.njit(cache=True)
def initialize_labels(labels, n_parts, rng_state):
    for i in range(n_parts):
        labels[tau_rand_int(rng_state) % labels.shape[0]] = i
    return labels


@numba.njit(cache=True)
def initialize_labels_from_hubs(labels, n_parts, degrees):
    hubs = np.argsort(degrees)[-n_parts:]
    for i in range(n_parts):
        labels[hubs[i]] = i
    return labels


@numba.njit(cache=True)
def initialize_labels_from_diverse_hubs(labels, n_parts, degrees, indptr, indices):
    """Choose high-degree hubs while avoiding adjacent hubs when possible."""
    candidates = np.argsort(degrees)
    blocked = np.zeros(labels.shape[0], dtype=np.bool_)
    selected = np.zeros(labels.shape[0], dtype=np.bool_)
    next_label = 0

    for candidate_index in range(candidates.shape[0] - 1, -1, -1):
        candidate = candidates[candidate_index]
        if not blocked[candidate]:
            labels[candidate] = next_label
            selected[candidate] = True
            blocked[candidate] = True
            for edge in range(indptr[candidate], indptr[candidate + 1]):
                blocked[indices[edge]] = True
            next_label += 1
            if next_label == n_parts:
                return labels

    for candidate_index in range(candidates.shape[0] - 1, -1, -1):
        candidate = candidates[candidate_index]
        if not selected[candidate]:
            labels[candidate] = next_label
            next_label += 1
            if next_label == n_parts:
                break
    return labels


def label_propagation_init(
    graph,
    data,
    subset_mask,
    a,
    b,
    n_iter=100,
    n_embedding_epochs=64,
    approx_n_parts=None,
    n_components=2,
    scaling=1.0,
    random_state=None,
    recursive_init=True,
    base_init_threshold=1024,
    depth=1,
    verbose=False,
    root_membership=None,
    recursive_parallel=True,
    curve_schedule="strong_to_one",
    good_initialization_policy="heuristic",
    coarsening_ratio=4,
    hub_selection="degree",
    remove_coarse_diagonal=False,
    recursive_repulsion_strength=4.0,
    recursive_negative_sample_rate=1,
    recursive_negative_selection_range_mode="scaled_coarse",
    recursive_negative_selection_range_scale=0.5,
):
    if root_membership is None:
        root_membership = np.arange(graph.shape[0], dtype=np.int64)

    if graph.shape[0] <= base_init_threshold:
        result = data
        # Recenter
        scale = np.log10(result.shape[0]) * 3 * (np.log2(depth + 1))
        result -= np.mean(result, 0)
        spread = np.quantile(result, 0.95, 0) - np.quantile(result, 0.05, 0)
        spread[spread == 0.0] = 1.0
        result *= scale / spread

        result = result.astype(np.float32)
        return result

    if approx_n_parts is None:
        approx_n_parts = max(
            base_init_threshold, int(graph.shape[0] // coarsening_ratio)
        )

    # Ensure we have fewer parts than samples
    approx_n_parts = min(approx_n_parts, graph.shape[0] // 2)
    if approx_n_parts < 2:
        approx_n_parts = 2

    if verbose:
        print(
            ts()
            + f" Initializing with label propagation: {approx_n_parts} parts, and a graph with {graph.shape[0]} nodes. {a=}, {b=}.",
            flush=True,
        )

    # Initialize the label propagation process
    rng_state = random_state.randint(INT32_MIN, INT32_MAX, 3).astype(np.int64)
    labels = np.full(graph.shape[0], -1, dtype=np.int32)
    degrees = np.squeeze(np.asarray(graph.sum(axis=1)))
    if hub_selection == "degree":
        labels = initialize_labels_from_hubs(labels, approx_n_parts, degrees)
    elif hub_selection == "diverse":
        labels = initialize_labels_from_diverse_hubs(
            labels, approx_n_parts, degrees, graph.indptr, graph.indices
        )
    else:
        raise ValueError("hub_selection must be 'degree' or 'diverse'")

    prev_unlabeled = np.sum(labels < 0)
    for i in range(n_iter):
        labels = label_prop_iteration(
            graph.indptr,
            graph.indices,
            graph.data,
            labels,
            rng_state,
        )
        if i % 5 == 0:
            unlabeled = np.sum(labels < 0)
            if unlabeled == 0 or unlabeled == prev_unlabeled:
                break

        prev_unlabeled = unlabeled
    # Handle outliers
    labels = label_outliers(
        graph.indptr,
        graph.indices,
        labels,
        rng_state,
    )
    # Remap labels to a contiguous range
    labels = remap_labels(labels)

    base_reduction_map, reduced_graph = _coarsen_graph(
        graph, labels, remove_diagonal=remove_coarse_diagonal
    )
    reduced_root_membership = labels[root_membership]

    if not np.all(subset_mask):
        subset_reduction_map = normalize(
            base_reduction_map[subset_mask], axis=0, norm="l1"
        )
        reduced_data = (subset_reduction_map.T * data).astype(np.float32)
    else:
        l1_normalized_reduction_map = normalize(base_reduction_map, axis=0, norm="l1")
        reduced_data = (l1_normalized_reduction_map.T * data).astype(np.float32)

    if recursive_init:
        next_a, next_b = _next_curve_parameters(a, b, curve_schedule)
        reduced_init = label_propagation_init(
            reduced_graph,
            reduced_data,
            np.ones(reduced_graph.shape[0], dtype=np.bool_),
            next_a,
            next_b,
            n_iter=n_iter,
            approx_n_parts=approx_n_parts // coarsening_ratio,
            n_embedding_epochs=int(n_embedding_epochs * np.pow(2, 0.25)),
            n_components=n_components,
            scaling=scaling,
            random_state=random_state,
            recursive_init=True,
            base_init_threshold=base_init_threshold,
            depth=depth + 1,
            verbose=verbose,
            root_membership=reduced_root_membership,
            recursive_parallel=recursive_parallel,
            curve_schedule=curve_schedule,
            good_initialization_policy=good_initialization_policy,
            coarsening_ratio=coarsening_ratio,
            hub_selection=hub_selection,
            remove_coarse_diagonal=remove_coarse_diagonal,
            recursive_repulsion_strength=recursive_repulsion_strength,
            recursive_negative_sample_rate=recursive_negative_sample_rate,
            recursive_negative_selection_range_mode=recursive_negative_selection_range_mode,
            recursive_negative_selection_range_scale=recursive_negative_selection_range_scale,
        ).astype(np.float32)
        good_initialization = _good_initialization(
            reduced_init,
            approx_n_parts,
            base_init_threshold,
            good_initialization_policy,
            coarsening_ratio,
        )
    else:
        reduced_init = None
        good_initialization = False

    epochs_per_sample = make_epochs_per_sample(reduced_graph.data, n_embedding_epochs)
    resolved_repulsion_strength = float(
        _resolve_depth_schedule(recursive_repulsion_strength, depth)
    )
    resolved_negative_sample_rate = int(
        _resolve_depth_schedule(recursive_negative_sample_rate, depth)
    )
    resolved_negative_selection_range_mode = _resolve_depth_schedule(
        recursive_negative_selection_range_mode, depth
    )
    resolved_negative_selection_range_scale = float(
        _resolve_depth_schedule(recursive_negative_selection_range_scale, depth)
    )
    negative_selection_range = _resolve_negative_selection_range(
        reduced_init.shape[0],
        resolved_negative_selection_range_mode,
        resolved_negative_selection_range_scale,
    )
    reduced_layout = optimize_layout_euclidean(
        reduced_init,
        reduced_init,
        None,
        None,
        n_embedding_epochs,
        reduced_graph.shape[0],
        epochs_per_sample,
        a,
        b,
        rng_state,
        resolved_repulsion_strength,
        0.5,
        resolved_negative_sample_rate,
        parallel=recursive_parallel,
        verbose=verbose,
        densmap_kwds={},
        tqdm_kwds={"desc": f"Init recursion depth {depth}", "position": 1},
        move_other=False,
        csr_indptr=reduced_graph.indptr,
        csr_indices=reduced_graph.indices,
        csr_data=reduced_graph.data,
        random_state=random_state,
        optimizer="adam",
        good_initialization=good_initialization,
        negative_selection_range=negative_selection_range,
    )

    data_expander = normalize(graph @ base_reduction_map, norm="l1")
    result = (
        data_expander @ reduced_layout
        + normalize(base_reduction_map, norm="l1") @ reduced_layout
    ) / 2.0

    result = (scaling * (result - result.mean(axis=0))).astype(np.float32)

    # Procustes alignment to PCA of the original data for better stability
    rotation = procrustes_align(data, result[subset_mask])
    result = result @ rotation

    return result


def recursive_init(
    graph,
    data,
    a,
    b,
    n_components=2,
    base_init_threshold=1024,
    n_embedding_epochs=64,
    approx_n_parts=None,
    random_state=None,
    verbose=False,
    recursive_parallel=True,
    pca_random_state=None,
    curve_schedule="strong_to_one",
    good_initialization_policy="heuristic",
    coarsening_ratio=4,
    hub_selection="degree",
    remove_coarse_diagonal=False,
    anchor_method="sampled_pca",
    anchor_sample_size=16384,
    recursive_repulsion_strength=4.0,
    recursive_negative_sample_rate=1,
    recursive_negative_selection_range_mode="scaled_coarse",
    recursive_negative_selection_range_scale=0.5,
):
    if random_state is None:
        random_state = np.random.RandomState()
    if pca_random_state is None:
        pca_random_state = random_state

    anchor, anchor_mask = _anchor_reference(
        graph,
        data,
        n_components,
        anchor_method,
        anchor_sample_size,
        pca_random_state,
        verbose,
    )
    initial_a, initial_b = _initial_curve_parameters(a, b, curve_schedule)
    init = label_propagation_init(
        graph,
        anchor,
        anchor_mask,
        a=initial_a,
        b=initial_b,
        n_components=n_components,
        base_init_threshold=base_init_threshold,
        n_embedding_epochs=n_embedding_epochs,
        approx_n_parts=approx_n_parts,
        random_state=random_state,
        verbose=verbose,
        root_membership=np.arange(graph.shape[0], dtype=np.int64),
        recursive_parallel=recursive_parallel,
        curve_schedule=curve_schedule,
        good_initialization_policy=good_initialization_policy,
        coarsening_ratio=coarsening_ratio,
        hub_selection=hub_selection,
        remove_coarse_diagonal=remove_coarse_diagonal,
        recursive_repulsion_strength=recursive_repulsion_strength,
        recursive_negative_sample_rate=recursive_negative_sample_rate,
        recursive_negative_selection_range_mode=recursive_negative_selection_range_mode,
        recursive_negative_selection_range_scale=recursive_negative_selection_range_scale,
    )
    return init.astype(np.float32)
