import numpy as np
import numba
import scipy.sparse
from sklearn.neighbors import KDTree

from umap.layouts import small_data_layout, optimize_layout_euclidean, optimize_layout_generic
from umap.utils import reset_local_connectivity, make_epochs_per_sample
from umap.spectral import spectral_layout

import umap.distances as dist

INT32_MIN = np.iinfo(np.int32).min + 1
INT32_MAX = np.iinfo(np.int32).max - 1

@numba.njit(fastmath=True)
def highest_weight_label(node, graph_ind, graph_indptr, graph_data, partition):
    weight_by_label = dict()
    max_weight = 0.0
    best_label = -1

    for i in range(graph_indptr[node], graph_indptr[node + 1]):
        label = partition[graph_ind[i]]
        weight = graph_data[i]

        if label in weight_by_label:
            weight_by_label[label] += weight
        else:
            weight_by_label[label] = weight

        if weight_by_label[label] >= max_weight:
            best_label = label
            max_weight = weight_by_label[label]

    return best_label, max_weight


@numba.njit(parallel=True)
def propagate_labels(graph_ind, graph_indptr, graph_data, partition, node_order):
    n_changes = 0

    for i in numba.prange(node_order.shape[0]):
        node = node_order[i]
        new_label, weight = highest_weight_label(node, graph_ind, graph_indptr, graph_data,
                                                 partition)
        if new_label != partition[node]:
            n_changes += 1
        partition[node] = new_label

    return n_changes


@numba.njit()
def label_propagation_partitioning_numba(graph_ind, graph_indptr, graph_data, n_iters=30,
                                         tolerance=0.01):
    n_nodes = graph_indptr.shape[0] - 1

    partition = np.arange(n_nodes)
    node_order = partition.copy()
    np.random.shuffle(node_order)

    for n in range(n_iters):
        n_changes = propagate_labels(graph_ind, graph_indptr, graph_data, partition, node_order)
        np.random.shuffle(node_order)
        if (n_changes / n_nodes) < tolerance or len(set(partition)) <= 150:
            break

    return partition


def label_propagation_partitioning(graph, random_state, n_iters=30, tolerance=0.01):
    if scipy.sparse.issparse(graph):
        graph = graph.tocsr()
    else:
        graph = np.array(graph)
        if graph.ndim != 2:
            raise ValueError("Unrecognized graph input -- please specify an adjacency matrix")
        graph = scipy.sparse.csr_matrix(graph)

    return label_propagation_partitioning_numba(graph.indices, graph.indptr, graph.data, n_iters,
                                                tolerance)


@numba.njit()
def merged_graph_internal(graph_ind, graph_indptr, graph_data, partition):
    n_nodes = graph_indptr.shape[0] - 1
    parts = set(partition)
    part_relabelling = dict()

    for new_part, old_part in enumerate(parts):
        part_relabelling[old_part] = new_part

    relabelled_partition = np.empty_like(partition)
    for i in range(partition.shape[0]):
        relabelled_partition[i] = part_relabelling[partition[i]]

    dok = dict()
    for i in range(n_nodes):
        for k in range(graph_indptr[i], graph_indptr[i + 1]):
            j = graph_ind[k]
            merged_i = relabelled_partition[i]
            merged_j = relabelled_partition[j]
            if merged_i != merged_j:
                if (merged_i, merged_j) in dok:
                    dok[(merged_i, merged_j)] += np.log(1.0 - graph_data[k])
                else:
                    dok[(merged_i, merged_j)] = np.log(1.0 - graph_data[k])

    nnzs = len(dok)
    row = np.empty(nnzs, dtype=np.int32)
    col = np.empty(nnzs, dtype=np.int32)
    val = np.empty(nnzs, dtype=np.float32)
    for nz, key in enumerate(dok.keys()):
        i, j = key
        row[nz] = i
        col[nz] = j
        val[nz] = 1.0 - np.exp(dok[key])

    return row, col, val, relabelled_partition


def merged_graph(graph, partition):
    if scipy.sparse.issparse(graph):
        graph = graph.tocsr()
    else:
        graph = np.array(graph)
        if graph.ndim != 2:
            raise ValueError("Unrecognized graph input -- please specify an adjacency matrix")
        graph = scipy.sparse.csr_matrix(graph)

    row, col, val, relabelled_partition = merged_graph_internal(graph.indices, graph.indptr,
                                                                graph.data, partition)
    result = scipy.sparse.coo_matrix((val, (row, col))).tocsr()
    return result, relabelled_partition


@numba.njit(parallel=True)
def init_from_merged_graph_internal(init, merged_embedding, partition,
                                    graph_ind, graph_indptr, graph_data):
    for i in numba.prange(graph_indptr.shape[0] - 1):
        weights = dict()
        total_weight = 0
        for k in range(graph_indptr[i], graph_indptr[i + 1]):
            j = graph_ind[k]
            if partition[j] in weights:
                weights[partition[j]] += graph_data[k]
            else:
                weights[partition[j]] = graph_data[k]
            total_weight += graph_data[k]
        for k in weights:
            init[i, 0] += weights[k] * merged_embedding[k, 0] / total_weight
            init[i, 1] += weights[k] * merged_embedding[k, 1] / total_weight
    return init


def init_from_merged_graph(merged_embedding, partition, source_graph):
    if scipy.sparse.issparse(source_graph):
        source_graph = source_graph.tocsr()
    else:
        source_graph = np.array(source_graph)
        if source_graph.ndim != 2:
            raise ValueError("Unrecognized graph input -- please specify an adjacency matrix")
        source_graph = scipy.sparse.csr_matrix(source_graph)

    init = np.zeros((source_graph.shape[0], 2), dtype=np.float32, order='C')
    return init_from_merged_graph_internal(init, merged_embedding, partition,
                                           source_graph.indices, source_graph.indptr,
                                           source_graph.data)


def simple_simplicial_set_embedding(
    graph,
    a,
    b,
    n_epochs,
    init,
    random_state,
    output_metric=dist.named_distances_with_gradients["euclidean"],
    output_metric_kwds={},
    euclidean_output=True,
    parallel=False,
    verbose=False,
):
    """Perform a fuzzy simplicial set embedding, using a specified
    initialisation method and then minimizing the fuzzy set cross entropy
    between the 1-skeletons of the high and low dimensional fuzzy simplicial
    sets.

    Parameters
    ----------
    graph: sparse matrix
        The 1-skeleton of the high dimensional fuzzy simplicial set as
        represented by a graph for which we require a sparse matrix for the
        (weighted) adjacency matrix.

    a: float
        Parameter of differentiable approximation of right adjoint functor

    b: float
        Parameter of differentiable approximation of right adjoint functor

    n_epochs: int (optional, default 0)
        The number of training epochs to be used in optimizing the
        low dimensional embedding. Larger values result in more accurate
        embeddings. If 0 is specified a value will be selected based on
        the size of the input dataset (200 for large datasets, 500 for small).

    init: array of shape (n_samples, n_components)
        A numpy array of initial embedding positions.

    random_state: numpy RandomState or equivalent
        A state capable being used as a numpy random state.

    output_metric: function
        Function returning the distance between two points in embedding space and
        the gradient of the distance wrt the first argument.

    output_metric_kwds: dict
        Key word arguments to be passed to the output_metric function.

    euclidean_output: bool
        Whether to use the faster code specialised for euclidean output metrics

    parallel: bool (optional, default False)
        Whether to run the computation using numba parallel.
        Running in parallel is non-deterministic, and is not used
        if a random seed has been set, to ensure reproducibility.

    verbose: bool (optional, default False)
        Whether to report information on the current progress of the algorithm.

    Returns
    -------
    embedding: array of shape (n_samples, n_components)
        The optimized of ``graph`` into an ``n_components`` dimensional
        euclidean space.
    """
    graph = graph.tocoo()
    n_vertices = graph.shape[1]

    graph.data[graph.data < (graph.data.max() / float(n_epochs))] = 0.0
    graph.eliminate_zeros()

    init_data = np.array(init)
    if len(init_data.shape) == 2:
        if np.unique(init_data, axis=0).shape[0] < init_data.shape[0]:
            tree = KDTree(init_data)
            dist, ind = tree.query(init_data, k=2)
            nndist = np.mean(dist[:, 1])
            embedding = init_data + random_state.normal(
                scale=0.001 * nndist, size=init_data.shape
            ).astype(np.float32)
        else:
            embedding = init_data

    epochs_per_sample = make_epochs_per_sample(graph.data, n_epochs)

    head = graph.row
    tail = graph.col

    rng_state = random_state.randint(INT32_MIN, INT32_MAX, 3).astype(np.int64)

    embedding = (embedding - np.min(embedding, 0)) / (
        np.max(embedding, 0) - np.min(embedding, 0)
    )
    if euclidean_output:
        embedding = optimize_layout_euclidean(
            embedding,
            embedding,
            head,
            tail,
            n_epochs,
            n_vertices,
            epochs_per_sample,
            a,
            b,
            rng_state,
            1.0,
            1.0,
            5,
            parallel=parallel,
            verbose=verbose,
        )
    else:
        embedding = optimize_layout_generic(
            embedding,
            embedding,
            head,
            tail,
            n_epochs,
            n_vertices,
            epochs_per_sample,
            a,
            b,
            rng_state,
            1.0,
            1.0,
            5,
            output_metric,
            tuple(output_metric_kwds.values()),
            verbose=verbose,
        )

    return embedding


def recursive_initialise_embedding(
    graph,
    n_components,
    a,
    b,
    min_dist,
    random_state,
    output_metric=dist.named_distances_with_gradients["euclidean"],
    output_metric_kwds={},
    euclidean_output=True,
    parallel=False,
    verbose=False,
):
    partition = label_propagation_partitioning(graph, random_state)

    sub_graph, partition = merged_graph(graph, partition)
    # sub_graph = reset_local_connectivity(sub_graph,
    #                                     reset_local_metric=True)

    if sub_graph.shape[0] < 150:
        sub_layout = small_data_layout(sub_graph.toarray(),
                                  5 * random_state.random(size=(sub_graph.shape[0],
                                                             n_components)).astype(
                                      np.float32, order='C'),
                                      min_dist=min_dist)
    elif sub_graph.shape[0] < 8192:
        initialization = spectral_layout(
            None, sub_graph, n_components, random_state
        ).astype(np.float32, order='C')
        sub_layout = simple_simplicial_set_embedding(
            sub_graph,
            a,
            b,
            150,
            initialization,
            random_state,
            output_metric=output_metric,
            output_metric_kwds=output_metric_kwds,
            euclidean_output=euclidean_output,
            parallel=parallel,
            verbose=verbose,
        )
    else:
        initialization = recursive_initialise_embedding(
            sub_graph,
            n_components,
            a,
            b,
            min_dist,
            random_state,
            output_metric=output_metric,
            output_metric_kwds=output_metric_kwds,
            euclidean_output=euclidean_output,
            parallel=parallel,
            verbose=verbose,
        ).astype(np.float32, order='C')
        sub_layout = simple_simplicial_set_embedding(
            sub_graph,
            a,
            b,
            30,
            initialization,
            random_state,
            output_metric=output_metric,
            output_metric_kwds=output_metric_kwds,
            euclidean_output=euclidean_output,
            parallel=parallel,
            verbose=verbose,
        )

    return init_from_merged_graph(sub_layout, partition, graph)