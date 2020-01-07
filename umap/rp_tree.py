# Author: Leland McInnes <leland.mcinnes@gmail.com>
#
# License: BSD 3 clause
from __future__ import print_function
from collections import deque, namedtuple
from warnings import warn

import numpy as np
import numba

from umap.sparse import arr_unique, sparse_mul, sparse_diff, sparse_sum

from umap.utils import tau_rand_int, norm

import scipy.sparse
import locale

locale.setlocale(locale.LC_NUMERIC, "C")

# Used for a floating point "nearly zero" comparison
EPS = 1e-8

RandomProjectionTreeNode = namedtuple(
    "RandomProjectionTreeNode",
    ["indices", "is_leaf", "hyperplane", "offset", "left_child", "right_child"],
)

FlatTree = namedtuple("FlatTree", ["hyperplanes", "offsets", "children", "indices"])


@numba.njit(fastmath=True)
def angular_random_projection_split(data, indices, rng_state):
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

    if abs(left_norm) < EPS:
        left_norm = 1.0

    if abs(right_norm) < EPS:
        right_norm = 1.0

    # Compute the normal vector to the hyperplane (the vector between
    # the two points)
    hyperplane_vector = np.empty(dim, dtype=np.float32)

    for d in range(dim):
        hyperplane_vector[d] = (data[left, d] / left_norm) - (
            data[right, d] / right_norm
        )

    hyperplane_norm = norm(hyperplane_vector)
    if abs(hyperplane_norm) < EPS:
        hyperplane_norm = 1.0

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

        if abs(margin) < EPS:
            side[i] = abs(tau_rand_int(rng_state)) % 2
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

    return indices_left, indices_right, hyperplane_vector, None


@numba.njit(fastmath=True, nogil=True)
def euclidean_random_projection_split(data, indices, rng_state):
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
        hyperplane_offset -= (
            hyperplane_vector[d] * (data[left, d] + data[right, d]) / 2.0
        )

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

        if abs(margin) < EPS:
            side[i] = abs(tau_rand_int(rng_state)) % 2
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

    return indices_left, indices_right, hyperplane_vector, hyperplane_offset


@numba.njit(fastmath=True)
def sparse_angular_random_projection_split(inds, indptr, data, indices, rng_state):
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

    left_inds = inds[indptr[left] : indptr[left + 1]]
    left_data = data[indptr[left] : indptr[left + 1]]
    right_inds = inds[indptr[right] : indptr[right + 1]]
    right_data = data[indptr[right] : indptr[right + 1]]

    left_norm = norm(left_data)
    right_norm = norm(right_data)

    if abs(left_norm) < EPS:
        left_norm = 1.0

    if abs(right_norm) < EPS:
        right_norm = 1.0

    # Compute the normal vector to the hyperplane (the vector between
    # the two points)
    normalized_left_data = left_data / left_norm
    normalized_right_data = right_data / right_norm
    hyperplane_inds, hyperplane_data = sparse_diff(
        left_inds, normalized_left_data, right_inds, normalized_right_data
    )

    hyperplane_norm = norm(hyperplane_data)
    if abs(hyperplane_norm) < EPS:
        hyperplane_norm = 1.0
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

        i_inds = inds[indptr[indices[i]] : indptr[indices[i] + 1]]
        i_data = data[indptr[indices[i]] : indptr[indices[i] + 1]]

        mul_inds, mul_data = sparse_mul(
            hyperplane_inds, hyperplane_data, i_inds, i_data
        )
        for d in range(mul_data.shape[0]):
            margin += mul_data[d]

        if abs(margin) < EPS:
            side[i] = abs(tau_rand_int(rng_state)) % 2
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

    hyperplane = np.vstack((hyperplane_inds, hyperplane_data))

    return indices_left, indices_right, hyperplane, None


@numba.njit(fastmath=True)
def sparse_euclidean_random_projection_split(inds, indptr, data, indices, rng_state):
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

    left_inds = inds[indptr[left] : indptr[left + 1]]
    left_data = data[indptr[left] : indptr[left + 1]]
    right_inds = inds[indptr[right] : indptr[right + 1]]
    right_data = data[indptr[right] : indptr[right + 1]]

    # Compute the normal vector to the hyperplane (the vector between
    # the two points) and the offset from the origin
    hyperplane_offset = 0.0
    hyperplane_inds, hyperplane_data = sparse_diff(
        left_inds, left_data, right_inds, right_data
    )
    offset_inds, offset_data = sparse_sum(left_inds, left_data, right_inds, right_data)
    offset_data = offset_data / 2.0
    offset_inds, offset_data = sparse_mul(
        hyperplane_inds, hyperplane_data, offset_inds, offset_data
    )

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
        i_inds = inds[indptr[indices[i]] : indptr[indices[i] + 1]]
        i_data = data[indptr[indices[i]] : indptr[indices[i] + 1]]

        mul_inds, mul_data = sparse_mul(
            hyperplane_inds, hyperplane_data, i_inds, i_data
        )
        for d in range(mul_data.shape[0]):
            margin += mul_data[d]

        if abs(margin) < EPS:
            side[i] = abs(tau_rand_int(rng_state)) % 2
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

    hyperplane = np.vstack((hyperplane_inds, hyperplane_data))

    return indices_left, indices_right, hyperplane, hyperplane_offset


def make_euclidean_tree(data, indices, rng_state, leaf_size=30):
    if indices.shape[0] > leaf_size:
        (
            left_indices,
            right_indices,
            hyperplane,
            offset,
        ) = euclidean_random_projection_split(data, indices, rng_state)

        left_node = make_euclidean_tree(data, left_indices, rng_state, leaf_size)
        right_node = make_euclidean_tree(data, right_indices, rng_state, leaf_size)

        node = RandomProjectionTreeNode(
            None, False, hyperplane, offset, left_node, right_node
        )
    else:
        node = RandomProjectionTreeNode(indices, True, None, None, None, None)

    return node


def make_angular_tree(data, indices, rng_state, leaf_size=30):
    if indices.shape[0] > leaf_size:
        (
            left_indices,
            right_indices,
            hyperplane,
            offset,
        ) = angular_random_projection_split(data, indices, rng_state)

        left_node = make_angular_tree(data, left_indices, rng_state, leaf_size)
        right_node = make_angular_tree(data, right_indices, rng_state, leaf_size)

        node = RandomProjectionTreeNode(
            None, False, hyperplane, offset, left_node, right_node
        )
    else:
        node = RandomProjectionTreeNode(indices, True, None, None, None, None)

    return node


def make_sparse_euclidean_tree(inds, indptr, data, indices, rng_state, leaf_size=30):
    if indices.shape[0] > leaf_size:
        (
            left_indices,
            right_indices,
            hyperplane,
            offset,
        ) = sparse_euclidean_random_projection_split(
            inds, indptr, data, indices, rng_state
        )

        left_node = make_sparse_euclidean_tree(
            inds, indptr, data, left_indices, rng_state, leaf_size
        )
        right_node = make_sparse_euclidean_tree(
            inds, indptr, data, right_indices, rng_state, leaf_size
        )

        node = RandomProjectionTreeNode(
            None, False, hyperplane, offset, left_node, right_node
        )
    else:
        node = RandomProjectionTreeNode(indices, True, None, None, None, None)

    return node


def make_sparse_angular_tree(inds, indptr, data, indices, rng_state, leaf_size=30):
    if indices.shape[0] > leaf_size:
        (
            left_indices,
            right_indices,
            hyperplane,
            offset,
        ) = sparse_angular_random_projection_split(
            inds, indptr, data, indices, rng_state
        )

        left_node = make_sparse_angular_tree(
            inds, indptr, data, left_indices, rng_state, leaf_size
        )
        right_node = make_sparse_angular_tree(
            inds, indptr, data, right_indices, rng_state, leaf_size
        )

        node = RandomProjectionTreeNode(
            None, False, hyperplane, offset, left_node, right_node
        )
    else:
        node = RandomProjectionTreeNode(indices, True, None, None, None, None)

    return node


def make_tree(data, rng_state, leaf_size=30, angular=False):
    """Construct a random projection tree based on ``data`` with leaves
    of size at most ``leaf_size``.
    Parameters
    ----------
    data: array of shape (n_samples, n_features)
        The original data to be split
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
    indices = np.arange(data.shape[0])

    # Make a tree recursively until we get below the leaf size
    if is_sparse:
        inds = data.indices
        indptr = data.indptr
        spdata = data.data

        if angular:
            return make_sparse_angular_tree(
                inds, indptr, spdata, indices, rng_state, leaf_size
            )
        else:
            return make_sparse_euclidean_tree(
                inds, indptr, spdata, indices, rng_state, leaf_size
            )
    else:
        if angular:
            return make_angular_tree(data, indices, rng_state, leaf_size)
        else:
            return make_euclidean_tree(data, indices, rng_state, leaf_size)


def num_nodes(tree):
    """Determine the number of nodes in a tree"""
    if tree.is_leaf:
        return 1
    else:
        return 1 + num_nodes(tree.left_child) + num_nodes(tree.right_child)


def num_leaves(tree):
    """Determine the number of leaves in a tree"""
    if tree.is_leaf:
        return 1
    else:
        return num_leaves(tree.left_child) + num_leaves(tree.right_child)


def max_sparse_hyperplane_size(tree):
    """Determine the most number on non zeros in a hyperplane entry"""
    if tree.is_leaf:
        return 0
    else:
        return max(
            tree.hyperplane.shape[1],
            max_sparse_hyperplane_size(tree.left_child),
            max_sparse_hyperplane_size(tree.right_child),
        )


def recursive_flatten(
    tree, hyperplanes, offsets, children, indices, node_num, leaf_num
):
    if tree.is_leaf:
        children[node_num, 0] = -leaf_num
        indices[leaf_num, : tree.indices.shape[0]] = tree.indices
        leaf_num += 1
        return node_num, leaf_num
    else:
        if len(tree.hyperplane.shape) > 1:
            # sparse case
            hyperplanes[node_num][:, : tree.hyperplane.shape[1]] = tree.hyperplane
        else:
            hyperplanes[node_num] = tree.hyperplane
        offsets[node_num] = tree.offset
        children[node_num, 0] = node_num + 1
        old_node_num = node_num
        node_num, leaf_num = recursive_flatten(
            tree.left_child,
            hyperplanes,
            offsets,
            children,
            indices,
            node_num + 1,
            leaf_num,
        )
        children[old_node_num, 1] = node_num + 1
        node_num, leaf_num = recursive_flatten(
            tree.right_child,
            hyperplanes,
            offsets,
            children,
            indices,
            node_num + 1,
            leaf_num,
        )
        return node_num, leaf_num


def flatten_tree(tree, leaf_size):
    n_nodes = num_nodes(tree)
    n_leaves = num_leaves(tree)

    if len(tree.hyperplane.shape) > 1:
        # sparse case
        max_hyperplane_nnz = max_sparse_hyperplane_size(tree)
        hyperplanes = np.zeros(
            (n_nodes, tree.hyperplane.shape[0], max_hyperplane_nnz), dtype=np.float32
        )
    else:
        hyperplanes = np.zeros((n_nodes, tree.hyperplane.shape[0]), dtype=np.float32)

    offsets = np.zeros(n_nodes, dtype=np.float32)
    children = -1 * np.ones((n_nodes, 2), dtype=np.int64)
    indices = -1 * np.ones((n_leaves, leaf_size), dtype=np.int64)
    recursive_flatten(tree, hyperplanes, offsets, children, indices, 0, 0)
    return FlatTree(hyperplanes, offsets, children, indices)


@numba.njit()
def select_side(hyperplane, offset, point, rng_state):
    margin = offset
    for d in range(point.shape[0]):
        margin += hyperplane[d] * point[d]

    if abs(margin) < EPS:
        side = abs(tau_rand_int(rng_state)) % 2
        if side == 0:
            return 0
        else:
            return 1
    elif margin > 0:
        return 0
    else:
        return 1


@numba.njit()
def search_flat_tree(point, hyperplanes, offsets, children, indices, rng_state):
    node = 0
    while children[node, 0] > 0:
        side = select_side(hyperplanes[node], offsets[node], point, rng_state)
        if side == 0:
            node = children[node, 0]
        else:
            node = children[node, 1]

    return indices[-children[node, 0]]


@numba.njit()
def sparse_select_side(hyperplane, offset, point_inds, point_data, rng_state):
    margin = offset

    hyperplane_inds = arr_unique(hyperplane[0])
    hyperplane_data = hyperplane[1, : hyperplane_inds.shape[0]]

    aux_inds, aux_data = sparse_mul(
        hyperplane_inds, hyperplane_data, point_inds, point_data
    )

    for d in range(aux_data.shape[0]):
        margin += aux_data[d]

    if margin == 0:
        side = abs(tau_rand_int(rng_state)) % 2
        if side == 0:
            return 0
        else:
            return 1
    elif margin > 0:
        return 0
    else:
        return 1


@numba.njit()
def search_sparse_flat_tree(
    point_inds, point_data, hyperplanes, offsets, children, indices, rng_state
):
    node = 0
    while children[node, 0] > 0:
        side = sparse_select_side(
            hyperplanes[node], offsets[node], point_inds, point_data, rng_state
        )
        if side == 0:
            node = children[node, 0]
        else:
            node = children[node, 1]

    return indices[-children[node, 0]]


def make_forest(data, n_neighbors, n_trees, rng_state, angular=False):
    """Build a random projection forest with ``n_trees``.

    Parameters
    ----------
    data
    n_neighbors
    n_trees
    rng_state
    angular

    Returns
    -------
    forest: list
        A list of random projection trees.
    """
    result = []
    leaf_size = max(10, n_neighbors)
    try:
        result = [
            flatten_tree(make_tree(data, rng_state, leaf_size, angular), leaf_size)
            for i in range(n_trees)
        ]
    except (RuntimeError, RecursionError, SystemError):
        warn(
            "Random Projection forest initialisation failed due to recursion"
            "limit being reached. Something is a little strange with your "
            "data, and this may take longer than normal to compute."
        )

    return result


def rptree_leaf_array(rp_forest):
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
    if len(rp_forest) > 0:
        leaf_array = np.vstack([tree.indices for tree in rp_forest])
    else:
        leaf_array = np.array([[-1]])

    return leaf_array
