# Author: Leland McInnes <leland.mcinnes@gmail.com>
# Enough simple sparse operations in numba to enable sparse UMAP
#
# License: BSD 3 clause

import numpy as np
import numba

from .umap_ import tau_rand_int, norm

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
