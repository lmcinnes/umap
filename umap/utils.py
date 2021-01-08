# Author: Leland McInnes <leland.mcinnes@gmail.com>
#
# License: BSD 3 clause

import time

import numpy as np
import numba
from sklearn.utils.validation import check_is_fitted
import scipy.sparse


@numba.njit(parallel=True)
def fast_knn_indices(X, n_neighbors):
    """A fast computation of knn indices.

    Parameters
    ----------
    X: array of shape (n_samples, n_features)
        The input data to compute the k-neighbor indices of.

    n_neighbors: int
        The number of nearest neighbors to compute for each sample in ``X``.

    Returns
    -------
    knn_indices: array of shape (n_samples, n_neighbors)
        The indices on the ``n_neighbors`` closest points in the dataset.
    """
    knn_indices = np.empty((X.shape[0], n_neighbors), dtype=np.int32)
    for row in numba.prange(X.shape[0]):
        # v = np.argsort(X[row])  # Need to call argsort this way for numba
        v = X[row].argsort(kind="quicksort")
        v = v[:n_neighbors]
        knn_indices[row] = v
    return knn_indices


@numba.njit("i4(i8[:])")
def tau_rand_int(state):
    """A fast (pseudo)-random number generator.

    Parameters
    ----------
    state: array of int64, shape (3,)
        The internal state of the rng

    Returns
    -------
    A (pseudo)-random int32 value
    """
    state[0] = (((state[0] & 4294967294) << 12) & 0xFFFFFFFF) ^ (
        (((state[0] << 13) & 0xFFFFFFFF) ^ state[0]) >> 19
    )
    state[1] = (((state[1] & 4294967288) << 4) & 0xFFFFFFFF) ^ (
        (((state[1] << 2) & 0xFFFFFFFF) ^ state[1]) >> 25
    )
    state[2] = (((state[2] & 4294967280) << 17) & 0xFFFFFFFF) ^ (
        (((state[2] << 3) & 0xFFFFFFFF) ^ state[2]) >> 11
    )

    return state[0] ^ state[1] ^ state[2]


@numba.njit("f4(i8[:])")
def tau_rand(state):
    """A fast (pseudo)-random number generator for floats in the range [0,1]

    Parameters
    ----------
    state: array of int64, shape (3,)
        The internal state of the rng

    Returns
    -------
    A (pseudo)-random float32 in the interval [0, 1]
    """
    integer = tau_rand_int(state)
    return abs(float(integer) / 0x7FFFFFFF)


@numba.njit()
def norm(vec):
    """Compute the (standard l2) norm of a vector.

    Parameters
    ----------
    vec: array of shape (dim,)

    Returns
    -------
    The l2 norm of vec.
    """
    result = 0.0
    for i in range(vec.shape[0]):
        result += vec[i] ** 2
    return np.sqrt(result)


@numba.njit(parallel=True)
def submatrix(dmat, indices_col, n_neighbors):
    """Return a submatrix given an orginal matrix and the indices to keep.

    Parameters
    ----------
    dmat: array, shape (n_samples, n_samples)
        Original matrix.

    indices_col: array, shape (n_samples, n_neighbors)
        Indices to keep. Each row consists of the indices of the columns.

    n_neighbors: int
        Number of neighbors.

    Returns
    -------
    submat: array, shape (n_samples, n_neighbors)
        The corresponding submatrix.
    """
    n_samples_transform, n_samples_fit = dmat.shape
    submat = np.zeros((n_samples_transform, n_neighbors), dtype=dmat.dtype)
    for i in numba.prange(n_samples_transform):
        for j in numba.prange(n_neighbors):
            submat[i, j] = dmat[i, indices_col[i, j]]
    return submat


# Generates a timestamp for use in logging messages when verbose=True
def ts():
    return time.ctime(time.time())


# I'm not enough of a numba ninja to numba this successfully.
# np.arrays of lists, which are objects...
def csr_unique(matrix, return_index=True, return_inverse=True, return_counts=True):
    """Find the unique elements of a sparse csr matrix.
    We don't explicitly construct the unique matrix leaving that to the user
    who may not want to duplicate a massive array in memory.
    Returns the indices of the input array that give the unique values.
    Returns the indices of the unique array that reconstructs the input array.
    Returns the number of times each unique row appears in the input matrix.

    matrix: a csr matrix
    return_index = bool, optional
        If true, return the row indices of 'matrix'
    return_inverse: bool, optional
        If true, return the the indices of the unique array that can be
           used to reconstruct 'matrix'.
    return_counts = bool, optional
        If true, returns the number of times each unique item appears in 'matrix'

    The unique matrix can computed via
    unique_matrix = matrix[index]
    and the original matrix reconstructed via
    unique_matrix[inverse]
    """
    lil_matrix = matrix.tolil()
    rows = [x + y for x, y in zip(lil_matrix.rows, lil_matrix.data)]
    return_values = return_counts + return_inverse + return_index
    return np.unique(
        rows,
        return_index=return_index,
        return_inverse=return_inverse,
        return_counts=return_counts,
    )[1 : (return_values + 1)]


def disconnected_vertices(model):
    """
    Returns a boolean vector indicating which vertices are disconnected from the umap graph.
    These vertices will often be scattered across the space and make it difficult to focus on the main
    manifold.  They can either be filtered and have UMAP re-run or simply filtered from the interactive plotting tool
    via the subset_points parameter.
    Use ~disconnected_vertices(model) to only plot the connected points.
    Parameters
    ----------
    model: a trained UMAP model

    Returns
    -------
    A boolean vector indicating which points are disconnected
    """
    check_is_fitted(model, "graph_")
    if model.unique:
        vertices_disconnected = (
            np.array(model.graph_[model._unique_inverse_].sum(axis=1)).flatten() == 0
        )
    else:
        vertices_disconnected = np.array(model.graph_.sum(axis=1)).flatten() == 0
    return vertices_disconnected
