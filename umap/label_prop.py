import numpy as np
import numba

from scipy.sparse import csr_matrix, issparse
from sklearn.preprocessing import normalize
from sklearn.utils.extmath import randomized_svd

from umap.layouts import optimize_layout_euclidean
from umap.utils import tau_rand, tau_rand_int, ts
from umap.spectral import spectral_layout

INT32_MIN = np.iinfo(np.int32).min + 1
INT32_MAX = np.iinfo(np.int32).max - 1


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
    random_scale=1.0,
    random_state=None,
    recursive_init=True,
    base_init_threshold=1024,
    depth=1,
    verbose=False,
):
    if graph.shape[0] <= base_init_threshold:
        result = data
        # Recenter
        scale = (
            np.log10(result.shape[0]) * 3 * (np.log2(depth + 1))
        )  # Added log2(gamma) to scale with repulsion strength
        result -= np.mean(result, 0)
        spread = np.quantile(result, 0.95, 0) - np.quantile(result, 0.05, 0)
        spread[spread == 0.0] = 1.0
        result *= scale / spread

        return result.astype(np.float32)

    if approx_n_parts is None:
        approx_n_parts = max(base_init_threshold, int(graph.shape[0] // 4))

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
    labels = initialize_labels_from_hubs(
        labels, approx_n_parts, np.squeeze(np.asarray(graph.sum(axis=1)))
    )

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

    base_reduction_map = csr_matrix(
        (np.ones(labels.shape[0]), labels, np.arange(labels.shape[0] + 1)),
        shape=(labels.shape[0], labels.max() + 1),
    )
    complement_graph = graph.astype(np.float64)
    complement_graph.data = np.log1p(-np.clip(complement_graph.data, 0.0, 1.0 - 1e-16))
    reduced_graph = base_reduction_map.T * complement_graph * base_reduction_map
    reduced_graph.data = 1.0 - np.exp(reduced_graph.data)
    reduced_graph.eliminate_zeros()
    reduced_graph = reduced_graph.astype(np.float32)

    if not np.all(subset_mask):
        subset_reduction_map = normalize(
            base_reduction_map[subset_mask], axis=0, norm="l1"
        )
        reduced_data = (subset_reduction_map.T * data).astype(np.float32)
    else:
        l1_normalized_reduction_map = normalize(base_reduction_map, axis=0, norm="l1")
        reduced_data = (l1_normalized_reduction_map.T * data).astype(np.float32)

    if recursive_init:
        reduced_init = label_propagation_init(
            reduced_graph,
            reduced_data,
            np.ones(reduced_graph.shape[0], dtype=np.bool_),
            np.cbrt(a),
            np.cbrt(b),
            n_iter=n_iter,
            approx_n_parts=approx_n_parts // 4,
            n_embedding_epochs=int(n_embedding_epochs * np.pow(2, 0.25)),
            n_components=n_components,
            scaling=scaling,
            random_scale=random_scale,
            random_state=random_state,
            recursive_init=True,
            base_init_threshold=base_init_threshold,
            depth=depth + 1,
            verbose=verbose,
        ).astype(np.float32)
        good_initialization = approx_n_parts // 4 > base_init_threshold
    else:
        reduced_init = None
        good_initialization = False

    epochs_per_sample = make_epochs_per_sample(reduced_graph.data, n_embedding_epochs)
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
        2.0,  # 1.5,
        0.5,
        1,
        parallel=True,
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
        negative_selection_range=reduced_init.shape[0],
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
):
    if random_state is None:
        random_state = np.random.RandomState()

    n = data.shape[0]
    sample_size = min(16384, n)

    if sample_size < n:
        sample = np.sort(random_state.choice(n, size=sample_size, replace=False))
        pca_sample_mask = np.zeros(n, dtype=np.bool_)
        pca_sample_mask[sample] = True
        data_sample = data[sample]
    else:
        pca_sample_mask = np.ones(n, dtype=np.bool_)
        data_sample = data

    if not issparse(data_sample) and not np.all(np.isfinite(data_sample)):
        data_sample = np.asarray(data_sample).copy()
        finite = np.isfinite(data_sample)
        finite_counts = finite.sum(axis=0)
        finite_sums = np.where(finite, data_sample, 0.0).sum(axis=0)
        fill_values = np.divide(
            finite_sums,
            finite_counts,
            out=np.zeros_like(finite_sums, dtype=np.float64),
            where=finite_counts > 0,
        )
        data_sample = np.where(finite, data_sample, fill_values)

    X = data_sample - data_sample.mean(axis=0)
    U, S, _ = randomized_svd(
        X,
        n_components=n_components,
        n_iter=1,
        n_oversamples=8,
        random_state=random_state,
    )

    pca = (U * S).astype(np.float32, order="C")

    pca -= pca.min(axis=0)
    pca_span = pca.max(axis=0) - pca.min(axis=0)
    pca_span[pca_span == 0.0] = 1.0
    pca /= pca_span
    pca *= 10.0
    init = label_propagation_init(
        graph,
        pca,
        pca_sample_mask,
        a=np.cbrt(a),
        b=np.cbrt(b),
        n_components=n_components,
        base_init_threshold=base_init_threshold,
        n_embedding_epochs=n_embedding_epochs,
        approx_n_parts=approx_n_parts,
        random_state=random_state,
        verbose=verbose,
    )
    return init.astype(np.float32)
