import numpy as np
import numba

from scipy.sparse import csr_matrix
from sklearn.preprocessing import normalize
from sklearn.decomposition import PCA
from sklearn.manifold import SpectralEmbedding, MDS

from umap.layouts import optimize_layout_euclidean
from umap.utils import tau_rand, tau_rand_int
from umap.spectral import spectral_layout

INT32_MIN = np.iinfo(np.int32).min + 1
INT32_MAX = np.iinfo(np.int32).max - 1


def make_epochs_per_sample(weights, n_epochs):
    result = -1.0 * np.ones(weights.shape[0], dtype=np.float64)
    n_samples = n_epochs * (weights / weights.max())
    result[n_samples > 0] = float(n_epochs) / np.float64(n_samples[n_samples > 0])
    return result


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

    for i in numba.prange(n_rows):
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

            if n_iter >= 100 or len(node_queue) == 0:
                # Ensure we don't have modulo by zero
                num_labels = max(max_label + 1, 1)
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


import matplotlib.pyplot as plt


def label_propagation_init(
    graph,
    a,
    b,
    n_iter=100,
    n_epochs=32,
    approx_n_parts=None,
    n_components=2,
    scaling=1.0,
    random_scale=1.0,
    random_state=None,
    recursive_init=True,
    base_init_threshold=256,
    upscaling="partition_expander",
    depth=1,
    verbose=False,
):

    if random_state is None:
        random_state = np.random.RandomState()

    if graph.shape[0] <= base_init_threshold:
        result = spectral_layout(
            None,
            graph,
            n_components,
            random_state,
            metric="euclidean",
            metric_kwds={},
            compatibility_layout=False,
            verbose=verbose,
        )
        return result.astype(np.float32, order="C")
        # # We add a little noise to avoid local minima for optimization to come
        # embedding = noisy_scale_coords(
        #     embedding, random_state, max_coord=40, noise=0.01
        # )
        # result = random_state.normal(
        #     loc=0.0, scale=1.0, size=(graph.shape[0], n_components)
        # )
        # norms = np.linalg.norm(result, axis=1, keepdims=True)
        # result = result / norms
        # return result.astype(np.float32)

    if approx_n_parts is None:
        approx_n_parts = max(base_init_threshold, int(graph.shape[0] // 16))

    # Ensure we have fewer parts than samples
    approx_n_parts = min(approx_n_parts, graph.shape[0] // 2)
    if approx_n_parts < 2:
        approx_n_parts = 2

    # Initialize the label propagation process
    rng_state = random_state.randint(INT32_MIN, INT32_MAX, 3).astype(np.int64)
    labels = np.full(graph.shape[0], -1, dtype=np.int32)
    # labels = initialize_labels(labels, approx_n_parts, rng_state)
    labels = initialize_labels_from_hubs(
        labels, approx_n_parts, np.squeeze(np.asarray(graph.sum(axis=1)))
    )

    # Perform the label propagation iterations
    for i in range(n_iter):
        labels = label_prop_iteration(
            graph.indptr,
            graph.indices,
            graph.data,
            labels,
            rng_state,
        )

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
    normalized_reduction_map = normalize(base_reduction_map, axis=0, norm="l2")
    reduced_graph = normalized_reduction_map.T * graph * base_reduction_map
    reduced_graph.data = np.clip(reduced_graph.data, 0.0, 1.0).astype(np.float32)
    if recursive_init:
        reduced_init = label_propagation_init(
            reduced_graph,
            a,
            b,
            n_iter=n_iter,
            approx_n_parts=approx_n_parts // 4,
            n_epochs=n_epochs * 2,
            n_components=n_components,
            scaling=scaling,
            random_scale=random_scale,
            random_state=random_state,
            recursive_init=True,
            upscaling=upscaling,
            base_init_threshold=base_init_threshold,
            depth=depth + 1,
            verbose=verbose,
        )
        good_initialization = approx_n_parts // 4 > base_init_threshold
    else:
        reduced_init = None
        good_initialization = False

    epochs_per_sample = make_epochs_per_sample(reduced_graph.data, n_epochs)
    reduced_layout = optimize_layout_euclidean(
        reduced_init,
        reduced_init,
        None,
        None,
        n_epochs,
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

    if upscaling == "partition_expander":
        data_expander = normalize(
            (graph.multiply(graph.T)) @ normalized_reduction_map, norm="l1"
        )
        result = (
            data_expander @ reduced_layout
            + normalize(normalized_reduction_map, norm="l1") @ reduced_layout
        ) / 2.0
    elif upscaling == "jitter_expander":
        data_expander = normalize(
            (graph.multiply(graph.T)) @ normalized_reduction_map, norm="l1"
        )
        expanded = (
            data_expander @ reduced_layout
            + normalize(normalized_reduction_map, norm="l1") @ reduced_layout
        ) / 2.0
        jittered = reduced_layout[labels]
        jittered += random_state.normal(
            scale=random_scale / 4.0, size=(labels.shape[0], reduced_layout.shape[1])
        )
        result = (expanded + jittered) / 2.0
    else:
        result = reduced_layout[labels]
        result += random_state.normal(
            scale=random_scale, size=(labels.shape[0], reduced_layout.shape[1])
        )

    result = (scaling * (result - result.mean(axis=0))).astype(np.float32)
    return result
