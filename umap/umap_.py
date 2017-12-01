# Author: Leland McInnes <leland.mcinnes@gmail.com>
#
# License: BSD 3 clause
from __future__ import print_function
from collections import deque, namedtuple
from warnings import warn

from scipy.optimize import curve_fit
from sklearn.base import BaseEstimator
from sklearn.utils import check_random_state, check_array

import numpy as np
import scipy.sparse
import numba

import umap.distances as dist

import umap.sparse as sparse

from umap.utils import (tau_rand_int,
                        tau_rand,
                        norm,
                        make_heap,
                        heap_push,
                        rejection_sample,
                        build_candidates)

import locale

locale.setlocale(locale.LC_NUMERIC, 'C')

INT32_MIN = np.iinfo(np.int32).min + 1
INT32_MAX = np.iinfo(np.int32).max - 1

SMOOTH_K_TOLERANCE = 1e-5
MIN_K_DIST_SCALE = 1e-3
NPY_INFINITY = np.inf


@numba.njit()
def random_projection_cosine_split(data, indices, rng_state):
    """Given a set of ``indices`` for data points from ``data``, create
    a random hyperplane to split the data, returning two arrays indices
    that fall on either side of the hyperplane. This is the basis for a
    random projection tree, which simply uses this splitting recursively.

    This particular split uses cosine distance to determine the hyperplane
    and which side each data sample falls on.

    Parameters
    ----------
    data: array of shape (n_samples, n_features)
        The original data to be split

    indices: array of shape (tree_node_size,)
        The indices of the elements in the ``data`` array that are to
        be split in the current operation.

    rng_state: array of int64, shape (3,)
        The internal state of the rng

    Returns
    -------
    indices_left: array
        The elements of ``indices`` that fall on the "left" side of the
        random hyperplane.

    indices_right: array
        The elements of ``indices`` that fall on the "left" side of the
        random hyperplane.
    """
    dim = data.shape[1]

    # Select two random points, set the hyperplane between them
    left_index = tau_rand_int(rng_state) % indices.shape[0]
    right_index = tau_rand_int(rng_state) % indices.shape[0]
    right_index += left_index == right_index
    right_index = right_index % indices.shape[0]
    left = indices[left_index]
    right = indices[right_index]

    left_norm = norm(data[left])
    right_norm = norm(data[right])

    # Compute the normal vector to the hyperplane (the vector between
    # the two points)
    hyperplane_vector = np.empty(dim, dtype=np.float32)

    for d in range(dim):
        hyperplane_vector[d] = ((data[left, d] / left_norm) -
                                (data[right, d] / right_norm))

    hyperplane_norm = norm(hyperplane_vector)
    for d in range(dim):
        hyperplane_vector[d] = hyperplane_vector[d] / hyperplane_norm

    # For each point compute the margin (project into normal vector)
    # If we are on lower side of the hyperplane put in one pile, otherwise
    # put it in the other pile (if we hit hyperplane on the nose, flip a coin)
    n_left = 0
    n_right = 0
    side = np.empty(indices.shape[0], np.int8)
    for i in range(indices.shape[0]):
        margin = 0.0
        for d in range(dim):
            margin += hyperplane_vector[d] * data[indices[i], d]

        if margin == 0:
            side[i] = tau_rand_int(rng_state) % 2
            if side[i] == 0:
                n_left += 1
            else:
                n_right += 1
        elif margin > 0:
            side[i] = 0
            n_left += 1
        else:
            side[i] = 1
            n_right += 1

    # Now that we have the counts allocate arrays
    indices_left = np.empty(n_left, dtype=np.int64)
    indices_right = np.empty(n_right, dtype=np.int64)

    # Populate the arrays with indices according to which side they fell on
    n_left = 0
    n_right = 0
    for i in range(side.shape[0]):
        if side[i] == 0:
            indices_left[n_left] = indices[i]
            n_left += 1
        else:
            indices_right[n_right] = indices[i]
            n_right += 1

    return indices_left, indices_right


@numba.njit()
def random_projection_split(data, indices, rng_state):
    """Given a set of ``indices`` for data points from ``data``, create
    a random hyperplane to split the data, returning two arrays indices
    that fall on either side of the hyperplane. This is the basis for a
    random projection tree, which simply uses this splitting recursively.

    This particular split uses euclidean distance to determine the hyperplane
    and which side each data sample falls on.

    Parameters
    ----------
    data: array of shape (n_samples, n_features)
        The original data to be split

    indices: array of shape (tree_node_size,)
        The indices of the elements in the ``data`` array that are to
        be split in the current operation.

    rng_state: array of int64, shape (3,)
        The internal state of the rng

    Returns
    -------
    indices_left: array
        The elements of ``indices`` that fall on the "left" side of the
        random hyperplane.

    indices_right: array
        The elements of ``indices`` that fall on the "left" side of the
        random hyperplane.
    """
    dim = data.shape[1]

    # Select two random points, set the hyperplane between them
    left_index = tau_rand_int(rng_state) % indices.shape[0]
    right_index = tau_rand_int(rng_state) % indices.shape[0]
    right_index += left_index == right_index
    right_index = right_index % indices.shape[0]
    left = indices[left_index]
    right = indices[right_index]

    # Compute the normal vector to the hyperplane (the vector between
    # the two points) and the offset from the origin
    hyperplane_offset = 0.0
    hyperplane_vector = np.empty(dim, dtype=np.float32)

    for d in range(dim):
        hyperplane_vector[d] = data[left, d] - data[right, d]
        hyperplane_offset -= hyperplane_vector[d] * (
            data[left, d] + data[right, d]) / 2.0

    # For each point compute the margin (project into normal vector, add offset)
    # If we are on lower side of the hyperplane put in one pile, otherwise
    # put it in the other pile (if we hit hyperplane on the nose, flip a coin)
    n_left = 0
    n_right = 0
    side = np.empty(indices.shape[0], np.int8)
    for i in range(indices.shape[0]):
        margin = hyperplane_offset
        for d in range(dim):
            margin += hyperplane_vector[d] * data[indices[i], d]

        if margin == 0:
            side[i] = tau_rand_int(rng_state) % 2
            if side[i] == 0:
                n_left += 1
            else:
                n_right += 1
        elif margin > 0:
            side[i] = 0
            n_left += 1
        else:
            side[i] = 1
            n_right += 1

    # Now that we have the counts allocate arrays
    indices_left = np.empty(n_left, dtype=np.int64)
    indices_right = np.empty(n_right, dtype=np.int64)

    # Populate the arrays with indices according to which side they fell on
    n_left = 0
    n_right = 0
    for i in range(side.shape[0]):
        if side[i] == 0:
            indices_left[n_left] = indices[i]
            n_left += 1
        else:
            indices_right[n_right] = indices[i]
            n_right += 1

    return indices_left, indices_right


RandomProjectionTreeNode = namedtuple('RandomProjectionTreeNode',
                                      ['indices', 'is_leaf',
                                       'left_child', 'right_child'])


def make_tree(data, indices, rng_state, leaf_size=30, angular=False):
    """Construct a random projection tree based on ``data`` with leaves
    of size at most ``leaf_size``.

    Parameters
    ----------
    data: array of shape (n_samples, n_features)
        The original data to be split

    indices: array of shape (tree_node_size,)
        The indices of the elements in the ``data`` array that are to
        be split in the current operation. This should be np.arange(
        data.shape[0]) for a full tree build, and may be smaller when being
        called recursively for tree construction.

    rng_state: array of int64, shape (3,)
        The internal state of the rng

    leaf_size: int (optional, default 30)
        The maximum size of any leaf node in the tree. Any node in the tree
        with more than ``leaf_size`` will be split further to create child
        nodes.

    angular: bool (optional, default False)
        Whether to use cosine/angular distance to create splits in the tree,
        or euclidean distance.

    Returns
    -------
    node: RandomProjectionTreeNode
        A random projection tree node which links to its child nodes. This
        provides the full tree below the returned node.
    """
    is_sparse = scipy.sparse.isspmatrix_csr(data)

    # Make a tree recursively until we get below the leaf size
    if indices.shape[0] > leaf_size:
        if is_sparse:
            inds = data.indices
            indptr = data.indptr
            spdata = data.data

            if angular:
                (left_indices,
                 right_indices) = sparse.sparse_random_projection_cosine_split(
                    inds,
                    indptr,
                    spdata,
                    indices,
                    rng_state)
            else:
                left_indices, right_indices = \
                    sparse.sparse_random_projection_split(
                        inds,
                        indptr,
                        spdata,
                        indices,
                        rng_state)
        else:
            if angular:
                (left_indices,
                 right_indices) = random_projection_cosine_split(data,
                                                                 indices,
                                                                 rng_state)
            else:
                left_indices, right_indices = random_projection_split(data,
                                                                      indices,
                                                                      rng_state)
        left_node = make_tree(data,
                              left_indices,
                              rng_state,
                              leaf_size,
                              angular)
        right_node = make_tree(data,
                               right_indices,
                               rng_state,
                               leaf_size,
                               angular)

        node = RandomProjectionTreeNode(indices, False, left_node, right_node)
    else:
        node = RandomProjectionTreeNode(indices, True, None, None)

    return node


def get_leaves(tree):
    """Return the set of leaf nodes of a random projection tree.

    Parameters
    ----------
    tree: RandomProjectionTreeNode
        The root node of the tree to get leaves of.

    Returns
    -------
    leaves: list
        A list of arrays of indices of points in each leaf node.
    """
    if tree.is_leaf:
        return [tree.indices]
    else:
        return get_leaves(tree.left_child) + get_leaves(tree.right_child)


def rptree_leaf_array(data, n_neighbors, rng_state, n_trees=10, angular=False):
    """Generate an array of sets of candidate nearest neighbors by
    constructing a random projection forest and taking the leaves of all the
    trees. Any given tree has leaves that are a set of potential nearest
    neighbors. Given enough trees the set of all such leaves gives a good
    likelihood of getting a good set of nearest neighbors in composite. Since
    such a random projection forest is inexpensive to compute, this can be a
    useful means of seeding other nearest neighbor algorithms.

    Parameters
    ----------
    data: array of shape (n_samples, n_features)
        The data for which to generate nearest neighbor approximations.

    n_neighbors: int
        The number of nearest neighbors to attempt to approximate.

    rng_state: array of int64, shape (3,)
        The internal state of the rng

    n_trees: int (optional, default 10)
        The number of trees to build in the forest construction.

    angular: bool (optional, default False)
        Whether to use angular/cosine distance for random projection tree
        construction.

    Returns
    -------
    leaf_array: array of shape (n_leaves, max(10, n_neighbors))
        Each row of leaf array is a list of indices found in a given leaf.
        Since not all leaves are the same size the arrays are padded out with -1
        to ensure we can return a single ndarray.
    """
    leaves = []
    try:
        leaf_size = max(10, n_neighbors)
        for t in range(n_trees):
            tree = make_tree(data,
                             np.arange(data.shape[0]),
                             rng_state,
                             leaf_size=leaf_size,
                             angular=angular)
            leaves += get_leaves(tree)

        leaf_array = -1 * np.ones([len(leaves), leaf_size], dtype=np.int64)
        for i, leaf in enumerate(leaves):
            leaf_array[i, :len(leaf)] = leaf
    except (RuntimeError, RecursionError):
        warn('Random Projection forest initialisation failed due to recursion'
             'limit being reached. Something is a little strange with your '
             'data, and this may take longer than normal to compute.')
        leaf_array = np.array([[-1]])

    return leaf_array


def make_nn_descent(dist, dist_args):
    """Create a numba accelerated version of nearest neighbor descent
    specialised for the given distance metric and metric arguments. Numba
    doesn't support higher order functions directly, but we can instead JIT
    compile the version of NN-descent for any given metric.

    Parameters
    ----------
    dist: function
        A numba JITd distance function which, given two arrays computes a
        dissimilarity between them.

    dist_args: tuple
        Any extra arguments that need to be passed to the distance function
        beyond the two arrays to be compared.

    Returns
    -------
    A numba JITd function for nearest neighbor descent computation that is
    specialised to the given metric.
    """
    @numba.njit(parallel=True)
    def nn_descent(data, n_neighbors, rng_state, max_candidates=50,
                   n_iters=10, delta=0.001, rho=0.5,
                   rp_tree_init=True, leaf_array=None, verbose=False):
        n_vertices = data.shape[0]

        current_graph = make_heap(data.shape[0], n_neighbors)
        for i in range(data.shape[0]):
            indices = rejection_sample(n_neighbors, data.shape[0], rng_state)
            for j in range(indices.shape[0]):
                d = dist(data[i], data[indices[j]], *dist_args)
                heap_push(current_graph, i, d, indices[j], 1)
                heap_push(current_graph, indices[j], d, i, 1)

        if rp_tree_init:
            for n in range(leaf_array.shape[0]):
                for i in range(leaf_array.shape[1]):
                    if leaf_array[n, i] < 0:
                        break
                    for j in range(i + 1, leaf_array.shape[1]):
                        if leaf_array[n, j] < 0:
                            break
                        d = dist(data[leaf_array[n, i]], data[leaf_array[n, j]],
                                 *dist_args)
                        heap_push(current_graph, leaf_array[n, i], d,
                                  leaf_array[n, j],
                                  1)
                        heap_push(current_graph, leaf_array[n, j], d,
                                  leaf_array[n, i],
                                  1)

        for n in range(n_iters):
            if verbose:
                print("\t", n, " / ", n_iters)

            candidate_neighbors = build_candidates(current_graph, n_vertices,
                                                   n_neighbors, max_candidates,
                                                   rng_state)

            c = 0
            for i in range(n_vertices):
                for j in range(max_candidates):
                    p = int(candidate_neighbors[0, i, j])
                    if p < 0 or tau_rand(rng_state) < rho:
                        continue
                    for k in range(max_candidates):
                        q = int(candidate_neighbors[0, i, k])
                        if q < 0 or not candidate_neighbors[2, i, j] and not \
                                candidate_neighbors[2, i, k]:
                            continue

                        d = dist(data[p], data[q], *dist_args)
                        c += heap_push(current_graph, p, d, q, 1)
                        c += heap_push(current_graph, q, d, p, 1)

            if c <= delta * n_neighbors * data.shape[0]:
                break

        return current_graph[:2]

    return nn_descent


@numba.njit(parallel=True)
def smooth_knn_dist(distances, k, n_iter=128):
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

    n_iter: int (optiona, default 128)
        We need to binary search for the correct distance value. This is the
        max number of iterations to use in such a search.

    Returns
    -------
    knn_dist: array of shape (n_samples,)
        The distance to kth nearest neighbor, as suitably approximated.

    nn_dist: array of shape (n_samples,)
        The distance to the 1st nearest neighbor for each point.
    """
    target = np.log2(k)
    rho = np.zeros(distances.shape[0])
    result = np.zeros(distances.shape[0])

    for i in range(distances.shape[0]):
        lo = 0.0
        hi = NPY_INFINITY
        mid = 1.0

        # TODO: This is very inefficient, but will do for now. FIXME
        ith_distances = distances[i]
        non_zero_dists = ith_distances[ith_distances > 0.0]
        if non_zero_dists.shape[0] > 0:
            rho[i] = np.min(non_zero_dists)
        else:
            rho[i] = 0.0

        for n in range(n_iter):

            psum = 0.0
            for j in range(1, distances.shape[1]):
                psum += np.exp(-((distances[i, j] - rho[i]) / mid))
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


@numba.jit(parallel=True)
def fuzzy_simplicial_set(X, n_neighbors, random_state,
                         metric, metric_kwds={}, angular=False,
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

    angular: bool (optional, default False)
        Whether to use angular/cosine distance for the random projection
        forest for seeding NN-descent to determine approximate nearest
        neighbors.

    verbose: bool (optional, default False)
        Whether to report information on the current progress of the algorithm.

    Returns
    -------
    fuzzy_simplicial_set: coo_matrix
        A fuzzy simplicial set represented as a sparse matrix. The (i,
        j) entry of the matrix represents the membership strength of the
        1-simplex between the ith and jth sample points.
    """

    rows = np.zeros((X.shape[0] * n_neighbors), dtype=np.int64)
    cols = np.zeros((X.shape[0] * n_neighbors), dtype=np.int64)
    vals = np.zeros((X.shape[0] * n_neighbors), dtype=np.float64)

    if metric == 'precomputed':
        # Compute indices of n neares neighbors
        knn_indices = np.argsort(X)[:,:n_neighbors]
        # Compute the neares neighbor distances
        #   (equivalent to np.sort(X)[:,:n_neighbors])
        knn_dists = X[np.arange(X.shape[0])[:,None], knn_indices].copy()
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
            leaf_array = rptree_leaf_array(X, n_neighbors,
                                        rng_state, n_trees=10,
                                        angular=angular)
            tmp_indices, knn_dists = metric_nn_descent(X.indices,
                                                    X.indptr,
                                                    X.data,
                                                    X.shape[0],
                                                    n_neighbors,
                                                    rng_state,
                                                    max_candidates=60,
                                                    rp_tree_init=True,
                                                    leaf_array=leaf_array,
                                                    verbose=verbose)
        else:
            metric_nn_descent = make_nn_descent(distance_func,
                                                tuple(metric_kwds.values()))
            leaf_array = rptree_leaf_array(X, n_neighbors,
                                        rng_state, n_trees=10,
                                        angular=angular)
            tmp_indices, knn_dists = metric_nn_descent(X,
                                                    n_neighbors,
                                                    rng_state,
                                                    max_candidates=60,
                                                    rp_tree_init=True,
                                                    leaf_array=leaf_array,
                                                    verbose=verbose)
        knn_indices = tmp_indices.astype(np.int64)

        if np.any(knn_indices < 0):
            warn('Failed to correctly find n_neighbors for some samples.'
                'Results may be less than ideal. Try re-running with'
                'different parameters.')

        for i in range(knn_indices.shape[0]):
            order = np.argsort(knn_dists[i])
            knn_dists[i] = knn_dists[i][order]
            knn_indices[i] = knn_indices[i][order]

    sigmas, rhos = smooth_knn_dist(knn_dists, n_neighbors)

    for i in range(knn_indices.shape[0]):
        for j in range(n_neighbors):
            if knn_indices[i, j] == -1:
                continue  # We didn't get the full knn for i
            if knn_indices[i, j] == i:
                val = 0.0
            elif knn_dists[i, j] == 0.0:
                val = 1.0
            else:
                val = np.exp(-((knn_dists[i, j] - rhos[i]) / sigmas[i]))

            rows[i * n_neighbors + j] = i
            cols[i * n_neighbors + j] = knn_indices[i, j]
            vals[i * n_neighbors + j] = val

    result = scipy.sparse.coo_matrix((vals, (rows, cols)),
                                     shape=(X.shape[0], X.shape[0]))
    result.eliminate_zeros()

    transpose = result.transpose()

    prod_matrix = result.multiply(transpose)

    result = result + transpose - prod_matrix
    result.eliminate_zeros()

    return result


@numba.jit()
def create_sampler(probabilities):
    """Create the data necessary for a Walker alias sampler. This allows for
    efficient sampling from a dataset according to an array of weights as to
    how relatively likely each element of the dataset is to be sampled.

    Parameters
    ----------
    probabilities: array of shape (n_items_for_sampling,)
        An array of weights (which can be viewed as the desired probabilities
        of a multinomial distribution when l1 normalised).

    Returns
    -------
    prob: array of shape (n_items_for_sampling,)
        The probabilities of selecting an element or its alias

    alias: array of shape (n_items_for_sampling,)
        The alternate choice if the element is not to be selected.
    """
    prob = np.zeros(probabilities.shape[0], dtype=np.float64)
    alias = np.zeros(probabilities.shape[0], dtype=np.int64)

    norm_prob = probabilities.shape[0] * probabilities / probabilities.sum()
    norm_prob[np.isnan(norm_prob)] = 0.0

    is_small = (norm_prob < 1)
    small = np.where(is_small)[0]
    large = np.where(~is_small)[0]

    # We can use deque or just operate on arrays;
    # benchmarks to determine this at a later date
    small = deque(small)
    large = deque(large)

    while small and large:
        j = small.pop()
        k = large.pop()

        prob[j] = norm_prob[j]
        alias[j] = k

        norm_prob[k] -= (1.0 - norm_prob[j])

        if norm_prob[k] < 1.0:
            small.append(k)
        else:
            large.append(k)

    while small:
        prob[small.pop()] = 1.0

    while large:
        prob[large.pop()] = 1.0

    return prob, alias


@numba.njit()
def sample(prob, alias, rng_state):
    """Given data for a Walker alias sampler, perform sampling.

    Parameters
    ----------
    prob: array of shape (n_items_for_sampling,)
        The probabilities of selecting an element or its alias

    alias: array of shape (n_items_for_sampling,)
        The alternate choice if the element is not to be selected.

    rng_state: array of int64, shape (3,)
        The internal state of the rng

    Returns
    -------
    The index of the sampled item.
    """
    k = tau_rand_int(rng_state) % prob.shape[0]
    u = tau_rand(rng_state)

    if u < prob[k]:
        return k
    else:
        return alias[k]


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
        return eigenvectors[:, order]
    except scipy.sparse.linalg.ArpackError:
        warn('WARNING: spectral initialisation failed! The eigenvector solver\n'
             'failed. This is likely due to too small an eigengap. Consider\n'
             'adding some noise or jitter to your data.\n\n'
             'Falling back to random initialisation!')
        return random_state.uniform(low=-10.0, high=10.0,
                                    size=(graph.shape[0], 2))


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
def optimize_layout(embedding, positive_head, positive_tail,
                    n_edge_samples, n_vertices, prob, alias,
                    a, b, rng_state, gamma=1.0, initial_alpha=1.0,
                    negative_sample_rate=5, verbose=False):
    """Improve an embedding using stochastic gradient descent to minimize the
    fuzzy set cross entropy between the 1-skeletons of the high dimensional
    and low dimensional fuzzy simplicial sets. In practice this is done by
    sampling edges based on their membership strength (with the (1-p) terms
    coming from negative sampling similar to word2vec).

    Parameters
    ----------
    embedding: array of shape (n_samples, n_components)
        The initial embedding to be improved by SGD.

    positive_head: array of shape (n_1_simplices)
        The indices of the heads of 1-simplices with non-zero membership.

    positive_tail: array of shape (n_1_simplices)
        The indices of the tails of 1-simplices with non-zero membership.

    n_edge_samples: int
        The total number of edge samples to use in the optimization step.

    n_vertices: int
        The number of vertices (0-simplices) in the dataset.

    prob: array of shape (n_1_simplices)
        Walker alias sampler data.

    alias: array of shape (n_1_simplices)
        Walker alias sampler data

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
    dim = embedding.shape[1]
    alpha = initial_alpha

    for i in range(n_edge_samples):

        if i % negative_sample_rate == 0:
            is_negative_sample = False
        else:
            is_negative_sample = True

        if is_negative_sample:
            edge = tau_rand_int(rng_state) % (n_vertices ** 2)
            j = edge // n_vertices
            k = edge % n_vertices
        else:
            edge = sample(prob, alias, rng_state)
            j = positive_head[edge]
            k = positive_tail[edge]

        current = embedding[j]
        other = embedding[k]

        dist_squared = rdist(current, other)

        if is_negative_sample:

            grad_coeff = (2.0 * gamma * b)
            grad_coeff /= (0.001 + dist_squared) * (
                a * pow(dist_squared, b) + 1)

            if not np.isfinite(grad_coeff):
                grad_coeff = 8.0

        else:

            grad_coeff = (-2.0 * a * b * pow(dist_squared, b - 1.0))
            grad_coeff /= (a * pow(dist_squared, b) + 1.0)

        for d in range(dim):
            grad_d = clip(grad_coeff * (current[d] - other[d]))
            current[d] += grad_d * alpha
            other[d] += -grad_d * alpha

        if i % 10000 == 0:
            # alpha = np.exp(
            #     -0.69314718055994529 * (
            #     (3 * i) / n_edge_samples) ** 2) * initial_alpha
            alpha = (1.0 - np.sqrt(float(i) / n_edge_samples)) * initial_alpha
            if alpha < (initial_alpha * 0.000001):
                alpha = initial_alpha * 0.000001

        if verbose and i % int(n_edge_samples / 10) == 0:
            print("\t", i, " / ", n_edge_samples)

    return embedding


def simplicial_set_embedding(graph, n_components,
                             initial_alpha, a, b,
                             gamma, n_edge_samples,
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

    prob, alias = create_sampler(graph.data)

    if isinstance(init, str) and init == 'random':
        embedding = random_state.uniform(low=-10.0, high=10.0,
                                         size=(graph.shape[0], 2))
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

    if n_edge_samples <= 0:
        n_edge_samples = (graph.shape[0] // 150) * 1000000

    positive_head = graph.row
    positive_tail = graph.col

    rng_state = random_state.randint(INT32_MIN, INT32_MAX, 3).astype(np.int64)
    embedding = optimize_layout(embedding, positive_head, positive_tail,
                                n_edge_samples, n_vertices,
                                prob, alias, a, b, rng_state, gamma,
                                initial_alpha, verbose=verbose)

    return embedding


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

    gamma: float (optional, default 1.0)
        Weighting applied to negative samples in low dimensional embedding
        optimization. Values higher than one will result in greater weight
        being given to negative samples.

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
                 gamma=1.0,
                 n_edge_samples=None,
                 alpha=1.0,
                 init='spectral',
                 spread=1.0,
                 min_dist=0.1,
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
        self.n_edge_samples = n_edge_samples
        self.init = init
        self.n_components = n_components
        self.gamma = gamma
        self.initial_alpha = alpha
        self.alpha = alpha

        self.spread = spread
        self.min_dist = min_dist
        self.random_state = random_state
        self.angular_rp_forest = angular_rp_forest
        self.verbose = verbose

        # if metric in dist.named_distances:
        #     self._metric = dist.named_distances[self.metric]
        # elif callable(metric):
        #     self._metric = self.metric
        # else:
        #     raise ValueError('Supplied metric is neither '
        #                      'a recognised string, nor callable')

        if a is None or b is None:
            self.a, self.b = find_ab_params(self.spread, self.min_dist)
        else:
            self.a = a
            self.b = b

        if self.verbose:
            print("UMAP(n_neighbors={}, n_components={}, metric='{}', "
                  " gamma={}, n_edge_samples={}, alpha={}, init='{}', "
                  "spread={}, min_dist={}, a={}, b={}, random_state={}, "
                  "metric_kwds={}, verbose={})".format(
                      n_neighbors, n_components, metric, gamma,
                      n_edge_samples, alpha, init, spread,
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

        if scipy.sparse.isspmatrix_csr(X) and not X.has_sorted_indices:
            X.sort_indices()

        random_state = check_random_state(self.random_state)

        if self.verbose:
            print("Construct fuzzy simplicial set")

        graph = fuzzy_simplicial_set(
            X,
            self.n_neighbors,
            random_state,
            self.metric,
            self.metric_kwds,
            self.angular_rp_forest,
            self.verbose
        )

        if self.n_edge_samples is None:
            n_edge_samples = 0
        else:
            n_edge_samples = self.n_edge_samples

        if self.verbose:
            print("Construct embedding")

        self.embedding_ = simplicial_set_embedding(
            graph,
            self.n_components,
            self.initial_alpha,
            self.a,
            self.b,
            self.gamma,
            n_edge_samples,
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
