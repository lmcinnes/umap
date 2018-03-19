# Author: Leland McInnes <leland.mcinnes@gmail.com>
#
# License: BSD 3 clause
from __future__ import print_function
from warnings import warn

from scipy.optimize import curve_fit
from sklearn.base import BaseEstimator
from sklearn.utils import check_random_state, check_array
from sklearn.metrics import pairwise_distances
from sklearn.preprocessing import normalize

import numpy as np
import scipy.sparse
import scipy.sparse.csgraph
import numba

import umap.distances as dist

import umap.sparse as sparse

from umap.utils import tau_rand_int, deheap_sort
from umap.rp_tree import rptree_leaf_array, make_forest
from umap.nndescent import (make_nn_descent,
                            make_initialisations,
                            make_initialized_nnd_search,
                            initialise_search)

import locale

locale.setlocale(locale.LC_NUMERIC, 'C')

INT32_MIN = np.iinfo(np.int32).min + 1
INT32_MAX = np.iinfo(np.int32).max - 1

SMOOTH_K_TOLERANCE = 1e-5
MIN_K_DIST_SCALE = 1e-3
NPY_INFINITY = np.inf


@numba.njit(parallel=True)
def smooth_knn_dist(distances, k, n_iter=64, local_connectivity=1.0,
                    bandwidth=1.0):
    """Compute a continuous version of the distance to the kth nearest
    neighbor. That is, this is similar to knn-distance but allows continuous
    k values rather than requiring an integral k. In esscence we are simply
    computing the distance such that the cardinality of fuzzy set we generate
    is k.

    Parameters
    ----------
    distances: array of shape (n_samples, n_neighbors)
        Distances to nearest neighbors for each samples. Each row should be a
        sorted list of distances to a given samples nearest neighbors.

    k: float
        The number of nearest neighbors to approximate for.

    n_iter: int (optional, default 64)
        We need to binary search for the correct distance value. This is the
        max number of iterations to use in such a search.

    local_connectivity: int (optional, default 1)
        The local connectivity required -- i.e. the number of nearest
        neighbors that should be assumed to be connected at a local level.
        The higher this value the more connected the manifold becomes
        locally. In practice this should be not more than the local intrinsic
        dimension of the manifold.

    bandwidth: float (optional, default 1)
        The target bandwidth of the kernel, larger values will produce
        larger return values.

    Returns
    -------
    knn_dist: array of shape (n_samples,)
        The distance to kth nearest neighbor, as suitably approximated.

    nn_dist: array of shape (n_samples,)
        The distance to the 1st nearest neighbor for each point.
    """
    target = np.log2(k) * bandwidth
    rho = np.zeros(distances.shape[0])
    result = np.zeros(distances.shape[0])

    for i in range(distances.shape[0]):
        lo = 0.0
        hi = NPY_INFINITY
        mid = 1.0

        # TODO: This is very inefficient, but will do for now. FIXME
        ith_distances = distances[i]
        non_zero_dists = ith_distances[ith_distances > 0.0]
        if non_zero_dists.shape[0] >= local_connectivity:
            index = int(np.floor(local_connectivity))
            interpolation = local_connectivity - index
            if index > 0:
                if interpolation <= SMOOTH_K_TOLERANCE:
                    rho[i] = non_zero_dists[index - 1]
                else:
                    rho[i] = non_zero_dists[index - 1] + interpolation * (
                        non_zero_dists[index] - non_zero_dists[index - 1]
                    )
            else:
                rho[i] = interpolation * non_zero_dists[0]
        elif non_zero_dists.shape[0] > 0:
            rho[i] = np.max(non_zero_dists)
        else:
            rho[i] = 0.0

        for n in range(n_iter):

            psum = 0.0
            for j in range(1, distances.shape[1]):
                dist = max(0, (distances[i, j] - rho[i]))
                psum += np.exp(-(dist / mid))
            val = psum

            if np.fabs(val - target) < SMOOTH_K_TOLERANCE:
                break

            if val > target:
                hi = mid
                mid = (lo + hi) / 2.0
            else:
                lo = mid
                if hi == NPY_INFINITY:
                    mid *= 2
                else:
                    mid = (lo + hi) / 2.0

        result[i] = mid

        # TODO: This is very inefficient, but will do for now. FIXME
        if rho[i] > 0.0:
            if result[i] < MIN_K_DIST_SCALE * np.mean(ith_distances):
                result[i] = MIN_K_DIST_SCALE * np.mean(ith_distances)
        else:
            if result[i] < MIN_K_DIST_SCALE * np.mean(distances):
                result[i] = MIN_K_DIST_SCALE * np.mean(distances)

    return result, rho


def nearest_neighbors(X, n_neighbors, metric, metric_kwds, angular,
                      random_state, verbose=False):
    """Compute the ``n_neighbors`` nearest points for each data point in ``X``
    under ``metric``. This may be exact, but more likely is approximated via
    nearest neighbor descent.

    Parameters
    ----------
    X: array of shape (n_samples, n_features)
        The input data to compute the k-neighbor graph of.

    n_neighbors: int
        The number of nearest neighbors to compute for each sample in ``X``.

    metric: string or callable
        The metric to use for the computation.

    metric_kwds: dict
        Any arguments to pass to the metric computation function.

    angular: bool
        Whether to use angular rp trees in NN approximation.

    random_state: np.random state
        The random state to use for approximate NN computations.

    verbose: bool
        Whether to print status data during the computation.

    Returns
    -------
    knn_indices: array of shape (n_samples, n_neighbors)
        The indices on the ``n_neighbors`` closest points in the dataset.

    knn_dists: array of shape (n_samples, n_neighbors)
        The distances to the ``n_neighbors`` closest points in the dataset.
    """
    if metric == 'precomputed':
        # Note that this does not support sparse distance matrices yet ...
        # Compute indices of n nearest neighbors
        knn_indices = np.argsort(X)[:, :n_neighbors]
        # Compute the nearest neighbor distances
        #   (equivalent to np.sort(X)[:,:n_neighbors])
        knn_dists = X[np.arange(X.shape[0])[:, None], knn_indices].copy()

        rp_forest = []
    else:
        if callable(metric):
            distance_func = metric
        elif metric in dist.named_distances:
            distance_func = dist.named_distances[metric]
        else:
            raise ValueError('Metric is neither callable, ' +
                             'nor a recognised string')

        if metric in ('cosine', 'correlation', 'dice', 'jaccard'):
            angular = True

        rng_state = random_state.randint(INT32_MIN, INT32_MAX, 3) \
            .astype(np.int64)

        if scipy.sparse.isspmatrix_csr(X):
            if metric in sparse.sparse_named_distances:
                distance_func = sparse.sparse_named_distances[metric]
                if metric in sparse.sparse_need_n_features:
                    metric_kwds['n_features'] = X.shape[1]
            else:
                raise ValueError('Metric {} not supported for sparse ' +
                                 'data'.format(metric))
            metric_nn_descent = sparse.make_sparse_nn_descent(
                distance_func, tuple(metric_kwds.values()))

            # TODO: Hacked values for now
            n_trees = 5 + int(round((X.shape[0]) ** 0.5 / 20.0))
            n_iters = max(5, int(round(np.log2(X.shape[0]))))

            rp_forest = make_forest(X, n_neighbors, n_trees, rng_state, angular)
            leaf_array = rptree_leaf_array(rp_forest)
            knn_indices, knn_dists = metric_nn_descent(X.indices,
                                                       X.indptr,
                                                       X.data,
                                                       X.shape[0],
                                                       n_neighbors,
                                                       rng_state,
                                                       max_candidates=60,
                                                       rp_tree_init=True,
                                                       leaf_array=leaf_array,
                                                       n_iters=n_iters,
                                                       verbose=verbose)
        else:
            metric_nn_descent = make_nn_descent(distance_func,
                                                tuple(metric_kwds.values()))
            # TODO: Hacked values for now
            n_trees = 5 + int(round((X.shape[0]) ** 0.5 / 20.0))
            n_iters = max(5, int(round(np.log2(X.shape[0]))))

            rp_forest = make_forest(X, n_neighbors, n_trees, rng_state, angular)
            leaf_array = rptree_leaf_array(rp_forest)
            knn_indices, knn_dists = metric_nn_descent(X,
                                                       n_neighbors,
                                                       rng_state,
                                                       max_candidates=60,
                                                       rp_tree_init=True,
                                                       leaf_array=leaf_array,
                                                       n_iters=n_iters,
                                                       verbose=verbose)

        if np.any(knn_indices < 0):
            warn('Failed to correctly find n_neighbors for some samples.'
                 'Results may be less than ideal. Try re-running with'
                 'different parameters.')

    return knn_indices, knn_dists, rp_forest


@numba.njit(parallel=True)
def compute_membership_strengths(knn_indices, knn_dists, sigmas, rhos):
    """Construct the membership strength data for the 1-skeleton of each local
    fuzzy simplicial set -- this is formed as a sparse matrix where each row is
    a local fuzzy simplicial set, with a membership strength for the
    1-simplex to each other data point.

    Parameters
    ----------
    knn_indices: array of shape (n_samples, n_neighbors)
        The indices on the ``n_neighbors`` closest points in the dataset.

    knn_dists: array of shape (n_samples, n_neighbors)
        The distances to the ``n_neighbors`` closest points in the dataset.

    sigmas: array of shape(n_samples)
        The normalization factor derived from the metric tensor approximation.

    rhos: array of shape(n_samples)
        The local connectivity adjustment.

    Returns
    -------
    rows: array of shape (n_samples * n_neighbors)
        Row data for the resulting sparse matrix (coo format)

    cols: array of shape (n_samples * n_neighbors)
        Column data for the resulting sparse matrix (coo format)

    vals: array of shape (n_samples * n_neighbors)
        Entries for the resulting sparse matrix (coo format)
    """
    n_samples = knn_indices.shape[0]
    n_neighbors = knn_indices.shape[1]

    rows = np.zeros((n_samples * n_neighbors), dtype=np.int64)
    cols = np.zeros((n_samples * n_neighbors), dtype=np.int64)
    vals = np.zeros((n_samples * n_neighbors), dtype=np.float64)

    for i in range(n_samples):
        for j in range(n_neighbors):
            if knn_indices[i, j] == -1:
                continue  # We didn't get the full knn for i
            if knn_indices[i, j] == i:
                val = 0.0
            elif knn_dists[i, j] - rhos[i] <= 0.0:
                val = 1.0
            else:
                val = np.exp(-((knn_dists[i, j] - rhos[i]) / (sigmas[i])))

            rows[i * n_neighbors + j] = i
            cols[i * n_neighbors + j] = knn_indices[i, j]
            vals[i * n_neighbors + j] = val

    return rows, cols, vals


@numba.jit()
def fuzzy_simplicial_set(X, n_neighbors, random_state,
                         metric, metric_kwds={},
                         knn_indices=None,
                         knn_dists=None,
                         angular=False,
                         set_op_mix_ratio=1.0,
                         local_connectivity=1.0, bandwidth=1.0,
                         verbose=False):
    """Given a set of data X, a neighborhood size, and a measure of distance
    compute the fuzzy simplicial set (here represented as a fuzzy graph in
    the form of a sparse matrix) associated to the data. This is done by
    locally approximating geodesic distance at each point, creating a fuzzy
    simplicial set for each such point, and then combining all the local
    fuzzy simplicial sets into a global one via a fuzzy union.

    Parameters
    ----------
    X: array of shape (n_samples, n_features)
        The data to be modelled as a fuzzy simplicial set.

    n_neighbors: int
        The number of neighbors to use to approximate geodesic distance.
        Larger numbers induce more global estimates of the manifold that can
        miss finer detail, while smaller values will focus on fine manifold
        structure to the detriment of the larger picture.

    random_state: numpy RandomState or equivalent
        A state capable being used as a numpy random state.

    metric: string or function (optional, default 'euclidean')
        The metric to use to compute distances in high dimensional space.
        If a string is passed it must match a valid predefined metric. If
        a general metric is required a function that takes two 1d arrays and
        returns a float can be provided. For performance purposes it is
        required that this be a numba jit'd function. Valid string metrics
        include:
            * euclidean
            * manhattan
            * chebyshev
            * minkowski
            * canberra
            * braycurtis
            * mahalanobis
            * wminkowski
            * seuclidean
            * cosine
            * correlation
            * haversine
            * hamming
            * jaccard
            * dice
            * russelrao
            * kulsinski
            * rogerstanimoto
            * sokalmichener
            * sokalsneath
            * yule
        Metrics that take arguments (such as minkowski, mahalanobis etc.)
        can have arguments passed via the metric_kwds dictionary. At this
        time care must be taken and dictionary elements must be ordered
        appropriately; this will hopefully be fixed in the future.

    metric_kwds: dict (optional, default {})
        Arguments to pass on to the metric, such as the ``p`` value for
        Minkowski distance.

    knn_indices: array of shape (n_samples, n_neighbors) (optional)
        If the k-nearest neighbors of each point has already been calculated
        you can pass them in here to save computation time. This should be
        an array with the indices of the k-nearest neighbors as a row for
        each data point.

    knn_dists: array of shape (n_samples, n_neighbors) (optional)
        If the k-nearest neighbors of each point has already been calculated
        you can pass them in here to save computation time. This should be
        an array with the distances of the k-nearest neighbors as a row for
        each data point.

    angular: bool (optional, default False)
        Whether to use angular/cosine distance for the random projection
        forest for seeding NN-descent to determine approximate nearest
        neighbors.

    set_op_mix_ratio: float (optional, default 1.0)
        Interpolate between (fuzzy) union and intersection as the set operation
        used to combine local fuzzy simplicial sets to obtain a global fuzzy
        simplicial sets. Both fuzzy set operations use the product t-norm.
        The value of this parameter should be between 0.0 and 1.0; a value of
        1.0 will use a pure fuzzy union, while 0.0 will use a pure fuzzy
        intersection.

    local_connectivity: int (optional, default 1)
        The local connectivity required -- i.e. the number of nearest
        neighbors that should be assumed to be connected at a local level.
        The higher this value the more connected the manifold becomes
        locally. In practice this should be not more than the local intrinsic
        dimension of the manifold.

    verbose: bool (optional, default False)
        Whether to report information on the current progress of the algorithm.

    Returns
    -------
    fuzzy_simplicial_set: coo_matrix
        A fuzzy simplicial set represented as a sparse matrix. The (i,
        j) entry of the matrix represents the membership strength of the
        1-simplex between the ith and jth sample points.
    """
    if knn_indices is None or knn_dists is None:
        knn_indices, knn_dists, _ = \
            nearest_neighbors(X, n_neighbors,
                              metric, metric_kwds, angular,
                              random_state, verbose=verbose)

    sigmas, rhos = smooth_knn_dist(knn_dists, n_neighbors,
                                   local_connectivity=local_connectivity)

    rows, cols, vals = compute_membership_strengths(knn_indices, knn_dists,
                                                    sigmas, rhos)

    result = scipy.sparse.coo_matrix((vals, (rows, cols)),
                                     shape=(X.shape[0], X.shape[0]))
    result.eliminate_zeros()

    transpose = result.transpose()

    prod_matrix = result.multiply(transpose)

    result = set_op_mix_ratio * (result + transpose - prod_matrix) + \
             (1.0 - set_op_mix_ratio) * prod_matrix

    result.eliminate_zeros()

    return result


@numba.jit()
def make_epochs_per_sample(weights, n_epochs):
    """Given a set of weights and number of epochs generate the number of
    epochs per sample for each weight.

    Parameters
    ----------
    weights: array of shape (n_1_simplices)
        The weights ofhow much we wish to sample each 1-simplex.

    n_epochs: int
        The total number of epochs we want to train for.

    Returns
    -------
    An array of number of epochs per sample, one for each 1-simplex.
    """
    result = -1.0 * np.ones(weights.shape[0], dtype=np.float64)
    n_samples = n_epochs * (weights / weights.max())
    result[n_samples > 0] = float(n_epochs) / n_samples[n_samples > 0]
    return result


def spectral_layout(graph, dim, random_state):
    """Given a graph compute the spectral embedding of the graph. This is
    simply the eigenvectors of the laplacian of the graph. Here we use the
    normalized laplacian.

    Parameters
    ----------
    graph: sparse matrix
        The (weighted) adjacency matrix of the graph as a sparse matrix.

    dim: int
        The dimension of the space into which to embed.

    random_state: numpy RandomState or equivalent
        A state capable being used as a numpy random state.

    Returns
    -------
    embedding: array of shape (n_vertices, dim)
        The spectral embedding of the graph.
    """
    n_samples = graph.shape[0]
    n_components, labels = scipy.sparse.csgraph.connected_components(graph)

    if n_components > 1:
        component_sizes = np.bincount(labels)
        largest_component = np.where(
            component_sizes == component_sizes.max())[0][0]
        graph = graph.tocsr()[labels == largest_component, :]
        graph = graph.tocsc()[:, labels == largest_component]
        graph = graph.tocoo()

    diag_data = np.asarray(graph.sum(axis=0))
    # standard Laplacian
    # D = scipy.sparse.spdiags(diag_data, 0, graph.shape[0], graph.shape[0])
    # L = D - graph
    # Normalized Laplacian
    I = scipy.sparse.identity(graph.shape[0], dtype=np.float64)
    D = scipy.sparse.spdiags(1.0 / np.sqrt(diag_data), 0, graph.shape[0],
                             graph.shape[0])
    L = I - D * graph * D

    k = dim + 1
    num_lanczos_vectors = max(2 * k + 1, int(np.sqrt(graph.shape[0])))
    try:
        eigenvalues, eigenvectors = scipy.sparse.linalg.eigsh(
            L, k,
            which='SM',
            ncv=num_lanczos_vectors,
            tol=1e-4,
            v0=np.ones(L.shape[0]),
            maxiter=graph.shape[0] * 5)
        order = np.argsort(eigenvalues)[1:k]
        if n_components == 1:
            return eigenvectors[:, order]
        else:
            init = random_state.uniform(low=-10.0, high=10.0,
                                        size=(n_samples, dim))
            init[labels == largest_component] = eigenvectors[:, order]
            return init
    except scipy.sparse.linalg.ArpackError:
        warn('WARNING: spectral initialisation failed! The eigenvector solver\n'
             'failed. This is likely due to too small an eigengap. Consider\n'
             'adding some noise or jitter to your data.\n\n'
             'Falling back to random initialisation!')
        return random_state.uniform(low=-10.0, high=10.0,
                                    size=(graph.shape[0], dim))


@numba.njit()
def clip(val):
    """Standard clamping of a value into a fixed range (in this case -4.0 to
    4.0)

    Parameters
    ----------
    val: float
        The value to be clamped.

    Returns
    -------
    The clamped value, now fixed to be in the range -4.0 to 4.0.
    """
    if val > 4.0:
        return 4.0
    elif val < -4.0:
        return -4.0
    else:
        return val


@numba.njit('f8(f8[:],f8[:])')
def rdist(x, y):
    """Reduced Euclidean distance.

    Parameters
    ----------
    x: array of shape (embedding_dim,)
    y: array of shape (embedding_dim,)

    Returns
    -------
    The squared euclidean distance between x and y
    """
    result = 0.0
    for i in range(x.shape[0]):
        result += (x[i] - y[i]) ** 2

    return result


@numba.njit()
def optimize_layout(head_embedding, tail_embedding, head, tail,
                    n_epochs, n_vertices, epochs_per_sample,
                    a, b, rng_state, gamma=1.0, initial_alpha=1.0,
                    negative_sample_rate=5.0, verbose=False):
    """Improve an embedding using stochastic gradient descent to minimize the
    fuzzy set cross entropy between the 1-skeletons of the high dimensional
    and low dimensional fuzzy simplicial sets. In practice this is done by
    sampling edges based on their membership strength (with the (1-p) terms
    coming from negative sampling similar to word2vec).

    Parameters
    ----------
    head_embedding: array of shape (n_samples, n_components)
        The initial embedding to be improved by SGD.

    tail_embedding: array of shape (source_samples, n_components)
        The reference embedding of embedded points. If not embedding new
        previously unseen points with respect to an existing embedding this
        is simply the head_embedding (again); otherwise it provides the
        existing embedding to embed with respect to.

    head: array of shape (n_1_simplices)
        The indices of the heads of 1-simplices with non-zero membership.

    tail: array of shape (n_1_simplices)
        The indices of the tails of 1-simplices with non-zero membership.

    n_epochs: int
        The number of training epochs to use in optimization.

    n_vertices: int
        The number of vertices (0-simplices) in the dataset.

    epochs_per_samples: array of shape (n_1_simplices)
        A float value of the number of epochs per 1-simplex. 1-simplices with
        weaker membership strength will have more epochs between being sampled.

    a: float
        Parameter of differentiable approximation of right adjoint functor

    b: float
        Parameter of differentiable approximation of right adjoint functor

    rng_state: array of int64, shape (3,)
        The internal state of the rng

    gamma: float (optional, default 1.0)
        Weight to apply to negative samples.

    initial_alpha: float (optional, default 1.0)
        Initial learning rate for the SGD.

    negative_sample_rate: int (optional, default 5)
        Number of negative samples to use per positive sample.

    verbose: bool (optional, default False)
        Whether to report information on the current progress of the algorithm.

    Returns
    -------
    embedding: array of shape (n_samples, n_components)
        The optimized embedding.
    """

    dim = head_embedding.shape[1]
    move_other = head_embedding.shape[0] == tail_embedding.shape[0]
    alpha = initial_alpha

    epochs_per_negative_sample = epochs_per_sample / negative_sample_rate
    epoch_of_next_negative_sample = epochs_per_negative_sample.copy()
    epoch_of_next_sample = epochs_per_sample.copy()

    for n in range(n_epochs):
        for i in range(epochs_per_sample.shape[0]):
            if epoch_of_next_sample[i] <= n:
                j = head[i]
                k = tail[i]

                current = head_embedding[j]
                other = tail_embedding[k]

                dist_squared = rdist(current, other)

                grad_coeff = (-2.0 * a * b * pow(dist_squared, b - 1.0))
                grad_coeff /= (a * pow(dist_squared, b) + 1.0)

                for d in range(dim):
                    grad_d = clip(grad_coeff * (current[d] - other[d]))
                    current[d] += grad_d * alpha
                    if move_other:
                        other[d] += -grad_d * alpha

                epoch_of_next_sample[i] += epochs_per_sample[i]

                n_neg_samples = int((n - epoch_of_next_negative_sample[i]) /
                                    epochs_per_negative_sample[i])

                for p in range(n_neg_samples):
                    k = tau_rand_int(rng_state) % n_vertices

                    other = tail_embedding[k]

                    dist_squared = rdist(current, other)

                    grad_coeff = (2.0 * gamma * b)
                    grad_coeff /= (0.001 + dist_squared) * (
                        a * pow(dist_squared, b) + 1)

                    if not np.isfinite(grad_coeff):
                        grad_coeff = 4.0

                    for d in range(dim):
                        grad_d = clip(grad_coeff * (current[d] - other[d]))
                        current[d] += grad_d * alpha

                epoch_of_next_negative_sample[i] += n_neg_samples * \
                                                    epochs_per_negative_sample[
                                                        i]

        alpha = initial_alpha * (1.0 - (float(n) / float(n_epochs)))

        if verbose and n % int(n_epochs / 10) == 0:
            print('\tcompleted ', n, ' / ', n_epochs, 'epochs')

    return head_embedding


def simplicial_set_embedding(graph, n_components,
                             initial_alpha, a, b,
                             gamma, negative_sample_rate, n_epochs,
                             init, random_state, verbose):
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

    n_components: int
        The dimensionality of the euclidean space into which to embed the data.

    initial_alpha: float
        Initial learning rate for the SGD.

    a: float
        Parameter of differentiable approximation of right adjoint functor

    b: float
        Parameter of differentiable approximation of right adjoint functor

    gamma: float (optional, default 1.0)
        Weight to apply to negative samples.

    n_edge_samples: int
        The total number of edge samples to use in the optimization step.

    init: string (optional, default 'spectral')
        How to initialize the low dimensional embedding. Options are:
            * 'spectral': use a spectral embedding of the fuzzy 1-skeleton
            * 'random': assign initial embedding positions at random.
            * A numpy array of initial embedding positions.

    random_state: numpy RandomState or equivalent
        A state capable being used as a numpy random state.

    verbose: bool (optional, default False)
        Whether to report information on the current progress of the algorithm.

    Returns
    -------
    embedding: array of shape (n_samples, n_components)
        The optimized of ``graph`` into an ``n_components`` dimensional
        euclidean space.
    """
    graph = graph.tocoo()
    graph.sum_duplicates()
    n_vertices = graph.shape[0]

    if isinstance(init, str) and init == 'random':
        embedding = random_state.uniform(low=-10.0, high=10.0,
                                         size=(graph.shape[0], n_components))
    elif isinstance(init, str) and init == 'spectral':
        # We add a little noise to avoid local minima for optimization to come
        initialisation = spectral_layout(graph, n_components, random_state)
        expansion = 10.0 / initialisation.max()
        embedding = (initialisation * expansion) + \
            random_state.normal(scale=0.001,
                                size=[graph.shape[0],
                                      n_components])
    else:
        try:
            init_data = np.array(init)
            if len(init_data.shape) == 2:
                embedding = init_data
            else:
                raise ValueError('Invalid init data passed.'
                                 'Should be "random", "spectral" or'
                                 ' a numpy array of initial embedding postions')
        except:
            raise ValueError('Invalid init data passed.'
                             'Should be "random", "spectral" or'
                             ' a numpy array of initial embedding postions')

    if n_epochs <= 0:
        # For smaller datasets we can use more epochs
        if graph.shape[0] <= 10000:
            n_epochs = 500
        else:
            n_epochs = 200

    graph.data[graph.data < (graph.data.max() / float(n_epochs))] = 0.0
    graph.eliminate_zeros()

    epochs_per_sample = make_epochs_per_sample(graph.data, n_epochs)

    head = graph.row
    tail = graph.col

    rng_state = random_state.randint(INT32_MIN, INT32_MAX, 3).astype(np.int64)
    embedding = optimize_layout(embedding, embedding, head, tail,
                                n_epochs, n_vertices,
                                epochs_per_sample, a, b, rng_state, gamma,
                                initial_alpha, negative_sample_rate,
                                verbose=verbose)

    return embedding


@numba.njit()
def init_transform(indices, weights, embedding):
    result = np.zeros((indices.shape[0], embedding.shape[1]), dtype=np.float64)

    for i in range(indices.shape[0]):
        for j in range(indices.shape[1]):
            for d in range(embedding.shape[1]):
                result[i, d] += weights[i, j] * embedding[indices[i, j], d]

    return result


def find_ab_params(spread, min_dist):
    """Fit a, b params for the differentiable curve used in lower
    dimensional fuzzy simplicial complex construction. We want the
    smooth curve (from a pre-defined family with simple gradient) that
    best matches an offset exponential decay.
    """

    def curve(x, a, b):
        return 1.0 / (1.0 + a * x ** (2 * b))

    xv = np.linspace(0, spread * 3, 300)
    yv = np.zeros(xv.shape)
    yv[xv < min_dist] = 1.0
    yv[xv >= min_dist] = np.exp(
        -(xv[xv >= min_dist] - min_dist) / spread)
    params, covar = curve_fit(curve, xv, yv)
    return params[0], params[1]


class UMAP(BaseEstimator):
    """Uniform Manifold Approximation and Projection

    Finds a low dimensional embedding of the data that approximates
    an underlying manifold.

    Parameters
    ----------
    n_neighbors: float (optional, default 15)
        The size of local neighborhood (in terms of number of neighboring
        sample points) used for manifold approximation. Larger values
        result in more global views of the manifold, while smaller
        values result in more local data being preserved. In general
        values should be in the range 2 to 100.

    n_components: int (optional, default 2)
        The dimension of the space to embed into. This defaults to 2 to
        provide easy visualization, but can reasonably be set to any
        integer value in the range 2 to 100.

    metric: string or function (optional, default 'euclidean')
        The metric to use to compute distances in high dimensional space.
        If a string is passed it must match a valid predefined metric. If
        a general metric is required a function that takes two 1d arrays and
        returns a float can be provided. For performance purposes it is
        required that this be a numba jit'd function. Valid string metrics
        include:
            * euclidean
            * manhattan
            * chebyshev
            * minkowski
            * canberra
            * braycurtis
            * mahalanobis
            * wminkowski
            * seuclidean
            * cosine
            * correlation
            * haversine
            * hamming
            * jaccard
            * dice
            * russelrao
            * kulsinski
            * rogerstanimoto
            * sokalmichener
            * sokalsneath
            * yule
        Metrics that take arguments (such as minkowski, mahalanobis etc.)
        can have arguments passed via the metric_kwds dictionary. At this
        time care must be taken and dictionary elements must be ordered
        appropriately; this will hopefully be fixed in the future.

    n_edge_samples: int (optional, default None)
        The number of edge/1-simplex samples to be used in optimizing the
        low dimensional embedding. Larger values result in more accurate
        embeddings. If None is specified a value will be selected based on
        the size of the input dataset (typically around dataset_size * 10**4).

    alpha: float (optional, default 1.0)
        The initial learning rate for the embedding optimization.

    init: string (optional, default 'spectral')
        How to initialize the low dimensional embedding. Options are:
            * 'spectral': use a spectral embedding of the fuzzy 1-skeleton
            * 'random': assign initial embedding positions at random.
            * A numpy array of initial embedding positions.

    min_dist: float (optional, default 0.1)
        The effective minimum distance between embedded points. Smaller values
        will result in a more clustered/clumped embedding where nearby points
        on the manifold are drawn closer together, while larger values will
        result on a more even dispersal of points. The value should be set
        relative to the ``spread`` value, which determines the scale at which
        embedded points will be spread out.

    spread: float (optional, default 1.0)
        The effective scale of embedded points. In combination with ``min_dist``
        this determines how clustered/clumped the embedded points are.

    set_op_mix_ratio: float (optional, default 1.0)
        Interpolate between (fuzzy) union and intersection as the set operation
        used to combine local fuzzy simplicial sets to obtain a global fuzzy
        simplicial sets. Both fuzzy set operations use the product t-norm.
        The value of this parameter should be between 0.0 and 1.0; a value of
        1.0 will use a pure fuzzy union, while 0.0 will use a pure fuzzy
        intersection.

    local_connectivity: int (optional, default 1)
        The local connectivity required -- i.e. the number of nearest
        neighbors that should be assumed to be connected at a local level.
        The higher this value the more connected the manifold becomes
        locally. In practice this should be not more than the local intrinsic
        dimension of the manifold.

    gamma: float (optional, default 1.0)
        Weighting applied to negative samples in low dimensional embedding
        optimization. Values higher than one will result in greater weight
        being given to negative samples.

    bandwidth: float (optional, default 1.0)
        The effective bandwidth of the kernel if we view the algorithm as
        similar to Laplacian eigenmaps. Larger values induce more
        connectivity and a more global view of the data, smaller values
        concentrate more locally.

    a: float (optional, default None)
        More specific parameters controlling the embedding. If None these
        values are set automatically as determined by ``min_dist`` and
        ``spread``.
    b: float (optional, default None)
        More specific parameters controlling the embedding. If None these
        values are set automatically as determined by ``min_dist`` and
        ``spread``.

    random_state: int, RandomState instance or None, optional (default: None)
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used
        by `np.random`.

    metric_kwds: dict (optional, default {})
        Arguments to pass on to the metric, such as the ``p`` value for
        Minkowski distance.

    angular_rp_forest: bool (optional, default False)
        Whether to use an angular random projection forest to initialise
        the approximate nearest neighbor search. This can be faster, but is
        mostly on useful for metric that use an angular style distance such
        as cosine, correlation etc. In the case of those metrics angular forests
        will be chosen automatically.

    verbose: bool (optional, default False)
        Controls verbosity of logging.
    """

    def __init__(self,
                 n_neighbors=15,
                 n_components=2,
                 metric='euclidean',
                 n_epochs=None,
                 alpha=1.0,
                 init='spectral',
                 spread=1.0,
                 min_dist=0.1,
                 enable_transform=False,
                 set_op_mix_ratio=1.0,
                 local_connectivity=1.0,
                 bandwidth=1.0,
                 gamma=1.0,
                 negative_sample_rate=5,
                 transform_queue_size=2.0,
                 a=None,
                 b=None,
                 random_state=None,
                 metric_kwds={},
                 angular_rp_forest=False,
                 verbose=False
                 ):

        self.n_neighbors = n_neighbors
        self.metric = metric
        self.metric_kwds = metric_kwds
        self.n_epochs = n_epochs
        self.init = init
        self.n_components = n_components
        self.gamma = gamma
        self.initial_alpha = alpha
        self.alpha = alpha

        self.spread = spread
        self.min_dist = min_dist
        self.enable_transform = enable_transform
        self.set_op_mix_ratio = set_op_mix_ratio
        self.local_connectivity = local_connectivity
        self.bandwidth = bandwidth
        self.negative_sample_rate = negative_sample_rate
        self.random_state = random_state
        self.angular_rp_forest = angular_rp_forest
        self.transform_queue_size = transform_queue_size
        self.verbose = verbose

        self._transform_available = False

        if a is None or b is None:
            self.a, self.b = find_ab_params(self.spread, self.min_dist)
        else:
            self.a = a
            self.b = b

        if set_op_mix_ratio < 0.0 or set_op_mix_ratio > 1.0:
            raise ValueError('set_op_mix_ratio must be between 0.0 and 1.0')

        if self.verbose:
            print("UMAP(n_neighbors={}, n_components={}, metric='{}', "
                  " gamma={}, n_epochs={}, alpha={}, init='{}', "
                  "spread={}, min_dist={}, a={}, b={}, random_state={}, "
                  "metric_kwds={}, verbose={})".format(
                      n_neighbors, n_components, metric, gamma,
                n_epochs, alpha, init, spread,
                      min_dist, a, b, random_state, metric_kwds, verbose))

    def fit(self, X, y=None):
        """Fit X into an embedded space.

        Parameters
        ----------
        X : array, shape (n_samples, n_features) or (n_samples, n_samples)
            If the metric is 'precomputed' X must be a square distance
            matrix. Otherwise it contains a sample per row. If the method
            is 'exact', X may be a sparse matrix of type 'csr', 'csc'
            or 'coo'.
        """

        # Handle other array dtypes (TODO: do this properly)
        X = check_array(X, accept_sparse='csr').astype(np.float64)
        self._raw_data = X

        if X.shape[0] <= self.n_neighbors:
            raise ValueError('n_neighbors must be smaller than the dataset '
                             'size!')

        if scipy.sparse.isspmatrix_csr(X) and not X.has_sorted_indices:
            X.sort_indices()
            self._sparse_data = True
        else:
            self._sparse_data = False

        random_state = check_random_state(self.random_state)

        if self.verbose:
            print("Construct fuzzy simplicial set")

        # Handle small cases efficiently by computing all distances
        if X.shape[0] < 4096:
            self._small_data = True
            dmat = pairwise_distances(X, metric=self.metric, **self.metric_kwds)
            self.graph_ = fuzzy_simplicial_set(
                dmat,
                self.n_neighbors,
                random_state,
                'precomputed',
                self.metric_kwds,
                None,
                None,
                self.angular_rp_forest,
                self.set_op_mix_ratio,
                self.local_connectivity,
                self.bandwidth,
                self.verbose
            )
        else:
            self._small_data = False
            # Standard case
            (self._knn_indices,
             self._knn_dists,
             self._rp_forest) = nearest_neighbors(X, self.n_neighbors,
                                                  self.metric, self.metric_kwds,
                                                  self.angular_rp_forest,
                                                  random_state, self.verbose)

            self.graph_ = fuzzy_simplicial_set(
                X,
                self.n_neighbors,
                random_state,
                self.metric,
                self.metric_kwds,
                self._knn_indices,
                self._knn_dists,
                self.angular_rp_forest,
                self.set_op_mix_ratio,
                self.local_connectivity,
                self.bandwidth,
                self.verbose
            )

            self._raw_data = X
            self._transform_available = True
            self._search_graph = scipy.sparse.lil_matrix((X.shape[0],
                                                          X.shape[0]),
                                                         dtype=np.int8)
            self._search_graph.rows = self._knn_indices
            self._search_graph.data = (self._knn_dists != 0).astype(np.int8)
            self._search_graph = self._search_graph.maximum(
                self._search_graph.transpose()).tocsr()

            if callable(self.metric):
                self._distance_func = self.metric
            elif self.metric in dist.named_distances:
                self._distance_func = dist.named_distances[self.metric]
            else:
                raise ValueError('Metric is neither callable, ' +
                                 'nor a recognised string')
            self._dist_args = tuple(self.metric_kwds.values())

            self._random_init, self._tree_init = make_initialisations(
                self._distance_func,
                self._dist_args)
            self._search = make_initialized_nnd_search(self._distance_func,
                                                       self._dist_args)

        if self.n_epochs is None:
            n_epochs = 0
        else:
            n_epochs = self.n_epochs

        if self.verbose:
            print("Construct embedding")

        self.embedding_ = simplicial_set_embedding(
            self.graph_,
            self.n_components,
            self.initial_alpha,
            self.a,
            self.b,
            self.gamma,
            self.negative_sample_rate,
            n_epochs,
            self.init,
            random_state,
            self.verbose
        )

        return self

    def fit_transform(self, X, y=None):
        """Fit X into an embedded space and return that transformed
        output.

        Parameters
        ----------
        X : array, shape (n_samples, n_features) or (n_samples, n_samples)
            If the metric is 'precomputed' X must be a square distance
            matrix. Otherwise it contains a sample per row.

        Returns
        -------
        X_new : array, shape (n_samples, n_components)
            Embedding of the training data in low-dimensional space.
        """
        self.fit(X)
        return self.embedding_

    def transform(self, X):
        """Transform X into the existing embedded space and return that
        transformed output.

        Parameters
        ----------
        X : array, shape (n_samples, n_features)
            New data to be transformed.

        Returns
        -------
        X_new : array, shape (n_samples, n_components)
            Embedding of the new data in low-dimensional space.
        """
        if self._sparse_data:
            raise ValueError('Transform not available for sparse input.')

        X = check_array(X, dtype=np.float64, order='C')
        random_state = check_random_state(self.random_state)
        rng_state = random_state.randint(INT32_MIN, INT32_MAX, 3).astype(
            np.int64)

        if self._small_data:
            dmat = pairwise_distances(X, self._raw_data,
                                      metric=self.metric, **self.metric_kwds)
            indices = np.argsort(dmat)
            dists = np.sort(dmat)  # TODO: more efficient approach
            indices = indices[:, :self.n_neighbors]
            dists = dists[:, :self.n_neighbors]
        else:
            init = initialise_search(self._rp_forest, self._raw_data,
                                     X,
                                     int(self.n_neighbors *
                                         self.transform_queue_size),
                                     self._random_init, self._tree_init,
                                     rng_state)
            result = self._search(self._raw_data,
                                  self._search_graph.indptr,
                                  self._search_graph.indices,
                                  init,
                                  X)

            indices, dists = deheap_sort(result)
            indices = indices[:, :self.n_neighbors]
            dists = dists[:, :self.n_neighbors]

        sigmas, rhos = smooth_knn_dist(indices, self.n_neighbors,
                                       local_connectivity=self.local_connectivity)

        rows, cols, vals = compute_membership_strengths(indices, dists,
                                                        sigmas, rhos)

        graph = scipy.sparse.coo_matrix((vals, (rows, cols)),
                                        shape=(X.shape[0],
                                               self._raw_data.shape[0]))

        csr_graph = normalize(graph.tocsr(), norm='l1')
        inds = csr_graph.indices.reshape(X.shape[0], self.n_neighbors)
        weights = csr_graph.data.reshape(X.shape[0], self.n_neighbors)
        embedding = init_transform(inds, weights, self.embedding_)

        if self.n_epochs is None:
            # For smaller datasets we can use more epochs
            if graph.shape[0] <= 10000:
                n_epochs = 300
            else:
                n_epochs = 100
        else:
            n_epochs = self.n_epochs

        graph.data[graph.data < (graph.data.max() / float(n_epochs))] = 0.0
        graph.eliminate_zeros()

        epochs_per_sample = make_epochs_per_sample(graph.data, n_epochs)

        head = graph.row
        tail = graph.col

        embedding = optimize_layout(embedding, self.embedding_, head, tail,
                                    n_epochs, X.shape[0],
                                    epochs_per_sample, self.a, self.b,
                                    rng_state, self.gamma,
                                    self.initial_alpha,
                                    self.negative_sample_rate,
                                    verbose=self.verbose)

        return embedding
