# Author: Leland McInnes <leland.mcinnes@gmail.com>
# Enough simple sparse operations in numba to enable sparse UMAP
#
# License: BSD 3 clause

import numpy as np
import numba

from .umap_ import (tau_rand_int,
                    norm,
                    make_heap,
                    heap_push,
                    rejection_sample,
                    build_candidates)


# Just reproduce a simpler version of numpy unique (not numba supported yet)
@numba.njit()
def arr_unique(arr):
    aux = np.sort(arr)
    flag = np.concatenate(([True], aux[1:] != aux[:-1]))
    return aux[flag]


# Just reproduce a simpler version of numpy union1d (not numba supported yet)
@numba.njit()
def arr_union(ar1, ar2):
    return arr_unique(np.concatenate(ar1, ar2))


# Just reproduce a simpler version of numpy intersect1d (not numba supported
# yet)
@numba.njit()
def arr_intersect(ar1, ar2):
    aux = np.concatenate((ar1, ar2))
    aux.sort()
    return aux[:-1][aux[1:] == aux[:-1]]


@numba.njit()
def sparse_sum(ind1, data1, ind2, data2):
    result_ind = arr_union(ind1, ind2)
    result_data = np.zeros(result_ind.shape[0], dtype=np.float32)

    i1 = 0
    i2 = 0
    nnz = 0

    # pass through both index lists
    while (i1 < ind1.shape[0] and i2 < ind2.shape[0]):
        j1 = ind1[i1]
        j2 = ind2[i2]

        if j1 == j2:
            val = data1[i1] + data2[i2]
            if val != 0:
                result_ind[nnz] = j1
                result_data[nnz] = val
                nnz += 1
            i1 += 1
            i2 += 1
        elif j1 < j2:
            val = data1[i1]
            if val != 0:
                result_ind[nnz] = j1
                result_data[nnz] = val
                nnz += 1
            i1 += 1
        else:
            val = data2[i2]
            if val != 0:
                result_ind[nnz] = j2
                result_data[nnz] = val
                nnz += 1
            i2 += 1

    # pass over the tails
    while i1 < ind1.shape[0]:
        val = data1[i1]
        if val != 0:
            result_ind[nnz] = j1
            result_data[nnz] = val
            nnz += 1
        i1 += 1

    while i2 < ind2.shape[0]:
        val = data2[i2]
        if val != 0:
            result_ind[nnz] = j2
            result_data[nnz] = val
            nnz += 1
        i2 += 1

    # truncate to the correct length in case there were zeros created
    result_ind = result_ind[:nnz]
    result_data = result_data[:nnz]

    return result_ind, result_data


@numba.njit()
def sparse_diff(ind1, data1, ind2, data2):
    return sparse_sum(ind1, data1, ind2, -data2)


@numba.njit()
def sparse_mul(ind1, data1, ind2, data2):
    result_ind = arr_intersect(ind1, ind2)
    result_data = np.zeros(result_ind.shape[0], dtype=np.float32)

    i1 = 0
    i2 = 0
    nnz = 0

    # pass through both index lists
    while (i1 < ind1.shape[0] and i2 < ind2.shape[0]):
        j1 = ind1[i1]
        j2 = ind2[i2]

        if j1 == j2:
            val = data1[i1] * data2[i2]
            if val != 0:
                result_ind[nnz] = j1
                result_data[nnz] = val
                nnz += 1
            i1 += 1
            i2 += 1
        elif j1 < j2:
            i1 += 1
        else:
            i2 += 1

    # truncate to the correct length in case there were zeros created
    result_ind = result_ind[:nnz]
    result_data = result_data[:nnz]

    return result_ind, result_data


@numba.njit()
def sparse_random_projection_cosine_split(inds,
                                          indptr,
                                          data,
                                          indices,
                                          rng_state):
    """Given a set of ``indices`` for data points from a sparse data set
    presented in csr sparse format as inds, indptr and data, create
    a random hyperplane to split the data, returning two arrays indices
    that fall on either side of the hyperplane. This is the basis for a
    random projection tree, which simply uses this splitting recursively.

    This particular split uses cosine distance to determine the hyperplane
    and which side each data sample falls on.

    Parameters
    ----------
    inds: array
        CSR format index array of the matrix

    indptr: array
        CSR format index pointer array of the matrix

    data: array
        CSR format data array of the matrix

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
    # Select two random points, set the hyperplane between them
    left_index = tau_rand_int(rng_state) % indices.shape[0]
    right_index = tau_rand_int(rng_state) % indices.shape[0]
    right_index += left_index == right_index
    right_index = right_index % indices.shape[0]
    left = indices[left_index]
    right = indices[right_index]

    left_inds = inds[indptr[left]:indptr[left+1]]
    left_data = data[indptr[left]:indptr[left+1]]
    right_inds = inds[indptr[right]:indptr[right+1]]
    right_data = data[indptr[right]:indptr[right+1]]

    left_norm = norm(left_data)
    right_norm = norm(right_data)

    # Compute the normal vector to the hyperplane (the vector between
    # the two points)
    normalized_left_data = left_data / left_norm
    normalized_right_data = right_data / right_norm
    hyperplane_inds, hyperplane_data = sparse_diff(left_inds,
                                                   normalized_left_data,
                                                   right_inds,
                                                   normalized_right_data)

    hyperplane_norm = norm(hyperplane_data)
    for d in range(hyperplane_data.shape[0]):
        hyperplane_data[d] = hyperplane_data[d] / hyperplane_norm

    # For each point compute the margin (project into normal vector)
    # If we are on lower side of the hyperplane put in one pile, otherwise
    # put it in the other pile (if we hit hyperplane on the nose, flip a coin)
    n_left = 0
    n_right = 0
    side = np.empty(indices.shape[0], np.int8)
    for i in range(indices.shape[0]):
        margin = 0.0

        i_inds = inds[indptr[indices[i]]:indptr[indices[i]+1]]
        i_data = data[indptr[indices[i]]:indptr[indices[i]+1]]

        mul_inds, mul_data = sparse_mul(hyperplane_inds,
                                        hyperplane_data,
                                        i_inds,
                                        i_data)
        for d in range(mul_data.shape[0]):
            margin += mul_data[d]

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
def sparse_random_projection_split(inds,
                                   indptr,
                                   data,
                                   indices,
                                   rng_state):
    """Given a set of ``indices`` for data points from a sparse data set
    presented in csr sparse format as inds, indptr and data, create
    a random hyperplane to split the data, returning two arrays indices
    that fall on either side of the hyperplane. This is the basis for a
    random projection tree, which simply uses this splitting recursively.

    This particular split uses cosine distance to determine the hyperplane
    and which side each data sample falls on.

    Parameters
    ----------
    inds: array
        CSR format index array of the matrix

    indptr: array
        CSR format index pointer array of the matrix

    data: array
        CSR format data array of the matrix

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
    # Select two random points, set the hyperplane between them
    left_index = tau_rand_int(rng_state) % indices.shape[0]
    right_index = tau_rand_int(rng_state) % indices.shape[0]
    right_index += left_index == right_index
    right_index = right_index % indices.shape[0]
    left = indices[left_index]
    right = indices[right_index]

    left_inds = inds[indptr[left]:indptr[left+1]]
    left_data = data[indptr[left]:indptr[left+1]]
    right_inds = inds[indptr[right]:indptr[right+1]]
    right_data = data[indptr[right]:indptr[right+1]]

    # Compute the normal vector to the hyperplane (the vector between
    # the two points) and the offset from the origin
    hyperplane_offset = 0.0
    hyperplane_inds, hyperplane_data = sparse_diff(left_inds,
                                                   left_data,
                                                   right_inds,
                                                   right_data)
    offset_inds, offset_data = sparse_sum(left_inds,
                                          left_data,
                                          right_inds,
                                          right_data)
    offset_data = offset_data / 2.0
    offset_inds, offset_data = sparse_mul(hyperplane_inds,
                                          hyperplane_data,
                                          offset_inds,
                                          offset_data)

    for d in range(offset_data.shape[0]):
        hyperplane_offset -= offset_data[d]

    # For each point compute the margin (project into normal vector, add offset)
    # If we are on lower side of the hyperplane put in one pile, otherwise
    # put it in the other pile (if we hit hyperplane on the nose, flip a coin)
    n_left = 0
    n_right = 0
    side = np.empty(indices.shape[0], np.int8)
    for i in range(indices.shape[0]):
        margin = hyperplane_offset
        i_inds = inds[indptr[indices[i]]:indptr[indices[i]+1]]
        i_data = data[indptr[indices[i]]:indptr[indices[i]+1]]

        mul_inds, mul_data = sparse_mul(hyperplane_inds,
                                        hyperplane_data,
                                        i_inds,
                                        i_data)
        for d in range(mul_data.shape[0]):
            margin += mul_data[d]

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


def make_sparse_nn_descent(sparse_dist, dist_args):
    """Create a numba accelerated version of nearest neighbor descent
    specialised for the given distance metric and metric arguments on sparse
    matrix data provided in CSR ind, indptr and data format. Numba
    doesn't support higher order functions directly, but we can instead JIT
    compile the version of NN-descent for any given metric.

    Parameters
    ----------
    sparse_dist: function
        A numba JITd distance function which, given four arrays (two sets of
        indices and data) computes a dissimilarity between them.

    dist_args: tuple
        Any extra arguments that need to be passed to the distance function
        beyond the two arrays to be compared.

    Returns
    -------
    A numba JITd function for nearest neighbor descent computation that is
    specialised to the given metric.
    """
    @numba.njit(parallel=True)
    def nn_descent(inds, indptr, data, n_vertices, n_neighbors, rng_state,
                   max_candidates=50,
                   n_iters=10, delta=0.001, rho=0.5,
                   rp_tree_init=True, leaf_array=None, verbose=False):

        current_graph = make_heap(n_vertices, n_neighbors)
        for i in range(n_vertices):
            indices = rejection_sample(n_neighbors, n_vertices, rng_state)
            for j in range(indices.shape[0]):

                from_inds = inds[indptr[i]:indptr[i+1]]
                from_data = data[indptr[i]:indptr[i+1]]

                to_inds = inds[indptr[indices[j]]:indptr[indices[j]+1]]
                to_data = data[indptr[indices[j]]:indptr[indices[j]+1]]

                d = sparse_dist(from_inds, from_data,
                                to_inds, to_data,
                                *dist_args)

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

                        from_inds = inds[indptr[leaf_array[n, i]]:indptr[leaf_array[n, i] + 1]]
                        from_data = data[indptr[leaf_array[n, i]]:indptr[leaf_array[n, i] + 1]]

                        to_inds = inds[
                                  indptr[leaf_array[n, j]]:indptr[leaf_array[n, j] + 1]]
                        to_data = data[
                                  indptr[leaf_array[n, j]]:indptr[leaf_array[n, j] + 1]]

                        d = sparse_dist(from_inds, from_data,
                                        to_inds, to_data,
                                        *dist_args)

                        heap_push(current_graph, leaf_array[n, i], d,
                                  leaf_array[n, j],
                                  1)
                        heap_push(current_graph, leaf_array[n, j], d,
                                  leaf_array[n, i],
                                  1)

        for n in range(n_iters):
            if verbose:
                print("\tnn descent iteration ", n, " / ", n_iters)

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

                        from_inds = inds[indptr[p]:indptr[p + 1]]
                        from_data = data[indptr[p]:indptr[p + 1]]

                        to_inds = inds[
                                  indptr[q]:indptr[q + 1]]
                        to_data = data[
                                  indptr[q]:indptr[q + 1]]

                        d = sparse_dist(from_inds, from_data,
                                        to_inds, to_data,
                                        *dist_args)

                        c += heap_push(current_graph, p, d, q, 1)
                        c += heap_push(current_graph, q, d, p, 1)

            if c <= delta * n_neighbors * n_vertices:
                break

        return current_graph[:2]

    return nn_descent

@numba.njit()
def sparse_euclidean(ind1, data1, ind2, data2):
    aux_inds, aux_data = sparse_diff(ind1, data1, ind2, data2)
    result = 0.0
    for i in range(aux_data.shape[0]):
        result += aux_data[i]**2
    return np.sqrt(result)


@numba.njit()
def sparse_manhattan(ind1, data1, ind2, data2):
    aux_inds, aux_data = sparse_diff(ind1, data1, ind2, data2)
    result = 0.0
    for i in range(aux_data.shape[0]):
        result += np.abs(aux_data[i])
    return result


@numba.njit()
def sparse_chebyshev(ind1, data1, ind2, data2):
    aux_inds, aux_data = sparse_diff(ind1, data1, ind2, data2)
    result = 0.0
    for i in range(aux_data.shape[0]):
        result = np.max(result, np.abs(aux_data[i]))
    return result


@numba.njit()
def sparse_minkowski(ind1, data1, ind2, data2, p):
    aux_inds, aux_data = sparse_diff(ind1, data1, ind2, data2)
    result = 0.0
    for i in range(aux_data.shape[0]):
        result += np.abs(aux_data[i])**p
    return result ** (1.0 / p)


@numba.njit()
def sparse_cosine(ind1, data1, ind2, data2):
    aux_inds, aux_data = sparse_mul(ind1, data1, ind2, data2)
    result = 0.0
    norm1 = norm(data1)
    norm2 = norm(data2)

    for i in range(aux_data.shape[0]):
        result += aux_data[i]

    return 1.0 - (result / np.sqrt(norm1 * norm2))

sparse_named_distances = {
    'euclidean' : sparse_euclidean,
    'manhattan' : sparse_manhattan,
    'l1'        : sparse_manhattan,
    'taxicab'   : sparse_manhattan,
    'chebyshev' : sparse_chebyshev,
    'linf'      : sparse_chebyshev,
    'linfty'    : sparse_chebyshev,
    'linfinity' : sparse_chebyshev,
    'minkowski' : sparse_minkowski,
    'cosine'    : sparse_cosine,
}
