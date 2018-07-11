# Author: Leland McInnes <leland.mcinnes@gmail.com>
# Enough simple sparse operations in numba to enable sparse UMAP
#
# License: BSD 3 clause
from __future__ import print_function
import numpy as np
import numba

from umap.utils import (
    tau_rand_int,
    tau_rand,
    norm,
    make_heap,
    heap_push,
    rejection_sample,
    build_candidates,
    deheap_sort,
)

import locale

locale.setlocale(locale.LC_NUMERIC, "C")

# Just reproduce a simpler version of numpy unique (not numba supported yet)
@numba.njit()
def arr_unique(arr):
    aux = np.sort(arr)
    flag = np.concatenate((np.ones(1, dtype=np.bool_), aux[1:] != aux[:-1]))
    return aux[flag]


# Just reproduce a simpler version of numpy union1d (not numba supported yet)
@numba.njit()
def arr_union(ar1, ar2):
    if ar1.shape[0] == 0:
        return ar2
    elif ar2.shape[0] == 0:
        return ar1
    else:
        return arr_unique(np.concatenate((ar1, ar2)))


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
    while i1 < ind1.shape[0] and i2 < ind2.shape[0]:
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
            result_ind[nnz] = i1
            result_data[nnz] = val
            nnz += 1
        i1 += 1

    while i2 < ind2.shape[0]:
        val = data2[i2]
        if val != 0:
            result_ind[nnz] = i2
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
    while i1 < ind1.shape[0] and i2 < ind2.shape[0]:
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
    def nn_descent(
        inds,
        indptr,
        data,
        n_vertices,
        n_neighbors,
        rng_state,
        max_candidates=50,
        n_iters=10,
        delta=0.001,
        rho=0.5,
        rp_tree_init=True,
        leaf_array=None,
        verbose=False,
    ):

        current_graph = make_heap(n_vertices, n_neighbors)
        for i in range(n_vertices):
            indices = rejection_sample(n_neighbors, n_vertices, rng_state)
            for j in range(indices.shape[0]):

                from_inds = inds[indptr[i] : indptr[i + 1]]
                from_data = data[indptr[i] : indptr[i + 1]]

                to_inds = inds[indptr[indices[j]] : indptr[indices[j] + 1]]
                to_data = data[indptr[indices[j]] : indptr[indices[j] + 1]]

                d = sparse_dist(from_inds, from_data, to_inds, to_data, *dist_args)

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

                        from_inds = inds[
                            indptr[leaf_array[n, i]] : indptr[leaf_array[n, i] + 1]
                        ]
                        from_data = data[
                            indptr[leaf_array[n, i]] : indptr[leaf_array[n, i] + 1]
                        ]

                        to_inds = inds[
                            indptr[leaf_array[n, j]] : indptr[leaf_array[n, j] + 1]
                        ]
                        to_data = data[
                            indptr[leaf_array[n, j]] : indptr[leaf_array[n, j] + 1]
                        ]

                        d = sparse_dist(
                            from_inds, from_data, to_inds, to_data, *dist_args
                        )

                        heap_push(
                            current_graph, leaf_array[n, i], d, leaf_array[n, j], 1
                        )
                        heap_push(
                            current_graph, leaf_array[n, j], d, leaf_array[n, i], 1
                        )

        for n in range(n_iters):
            if verbose:
                print("\t", n, " / ", n_iters)

            candidate_neighbors = build_candidates(
                current_graph, n_vertices, n_neighbors, max_candidates, rng_state
            )

            c = 0
            for i in range(n_vertices):
                for j in range(max_candidates):
                    p = int(candidate_neighbors[0, i, j])
                    if p < 0 or tau_rand(rng_state) < rho:
                        continue
                    for k in range(max_candidates):
                        q = int(candidate_neighbors[0, i, k])
                        if (
                            q < 0
                            or not candidate_neighbors[2, i, j]
                            and not candidate_neighbors[2, i, k]
                        ):
                            continue

                        from_inds = inds[indptr[p] : indptr[p + 1]]
                        from_data = data[indptr[p] : indptr[p + 1]]

                        to_inds = inds[indptr[q] : indptr[q + 1]]
                        to_data = data[indptr[q] : indptr[q + 1]]

                        d = sparse_dist(
                            from_inds, from_data, to_inds, to_data, *dist_args
                        )

                        c += heap_push(current_graph, p, d, q, 1)
                        c += heap_push(current_graph, q, d, p, 1)

            if c <= delta * n_neighbors * n_vertices:
                break

        return deheap_sort(current_graph)

    return nn_descent


@numba.njit()
def general_sset_intersection(
    indptr1,
    indices1,
    data1,
    indptr2,
    indices2,
    data2,
    result_row,
    result_col,
    result_val,
    mix_weight=0.5,
):

    left_min = max(data1.min() / 2.0, 1.0e-8)
    right_min = max(data2.min() / 2.0, 1.0e-8)

    for idx in range(result_row.shape[0]):
        i = result_row[idx]
        j = result_col[idx]

        left_val = left_min
        for k in range(indptr1[i], indptr1[i + 1]):
            if indices1[k] == j:
                left_val = data1[k]

        right_val = right_min
        for k in range(indptr2[i], indptr2[i + 1]):
            if indices2[k] == j:
                right_val = data2[k]

        if left_val > left_min or right_val > right_min:
            if mix_weight < 0.5:
                result_val[idx] = left_val * pow(
                    right_val, mix_weight / (1.0 - mix_weight)
                )
            else:
                result_val[idx] = (
                    pow(left_val, (1.0 - mix_weight) / mix_weight) * right_val
                )

    return


@numba.njit()
def sparse_euclidean(ind1, data1, ind2, data2):
    aux_inds, aux_data = sparse_diff(ind1, data1, ind2, data2)
    result = 0.0
    for i in range(aux_data.shape[0]):
        result += aux_data[i] ** 2
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
        result = max(result, np.abs(aux_data[i]))
    return result


@numba.njit()
def sparse_minkowski(ind1, data1, ind2, data2, p=2.0):
    aux_inds, aux_data = sparse_diff(ind1, data1, ind2, data2)
    result = 0.0
    for i in range(aux_data.shape[0]):
        result += np.abs(aux_data[i]) ** p
    return result ** (1.0 / p)


@numba.njit()
def sparse_hamming(ind1, data1, ind2, data2, n_features):
    num_not_equal = sparse_diff(ind1, data1, ind2, data2)[0].shape[0]
    return float(num_not_equal) / n_features


@numba.njit()
def sparse_canberra(ind1, data1, ind2, data2):
    abs_data1 = np.abs(data1)
    abs_data2 = np.abs(data2)
    denom_inds, denom_data = sparse_sum(ind1, abs_data1, ind2, abs_data2)
    denom_data = 1.0 / denom_data
    numer_inds, numer_data = sparse_diff(ind1, data1, ind2, data2)
    numer_data = np.abs(numer_data)

    val_inds, val_data = sparse_mul(numer_inds, numer_data, denom_inds, denom_data)

    return np.sum(val_data)


@numba.njit()
def sparse_bray_curtis(ind1, data1, ind2, data2):  # pragma: no cover
    abs_data1 = np.abs(data1)
    abs_data2 = np.abs(data2)
    denom_inds, denom_data = sparse_sum(ind1, abs_data1, ind2, abs_data2)

    if denom_data.shape[0] == 0:
        return 0.0

    denominator = np.sum(denom_data)

    numer_inds, numer_data = sparse_diff(ind1, data1, ind2, data2)
    numer_data = np.abs(numer_data)

    numerator = np.sum(numer_data)

    return float(numerator) / denominator


@numba.njit()
def sparse_jaccard(ind1, data1, ind2, data2):
    num_non_zero = arr_union(ind1, ind2).shape[0]
    num_equal = arr_intersect(ind1, ind2).shape[0]

    if num_non_zero == 0:
        return 0.0
    else:
        return float(num_non_zero - num_equal) / num_non_zero


@numba.njit()
def sparse_matching(ind1, data1, ind2, data2, n_features):
    num_true_true = arr_intersect(ind1, ind2).shape[0]
    num_non_zero = arr_union(ind1, ind2).shape[0]
    num_not_equal = num_non_zero - num_true_true

    return float(num_not_equal) / n_features


@numba.njit()
def sparse_dice(ind1, data1, ind2, data2):
    num_true_true = arr_intersect(ind1, ind2).shape[0]
    num_non_zero = arr_union(ind1, ind2).shape[0]
    num_not_equal = num_non_zero - num_true_true

    if num_not_equal == 0.0:
        return 0.0
    else:
        return num_not_equal / (2.0 * num_true_true + num_not_equal)


@numba.njit()
def sparse_kulsinski(ind1, data1, ind2, data2, n_features):
    num_true_true = arr_intersect(ind1, ind2).shape[0]
    num_non_zero = arr_union(ind1, ind2).shape[0]
    num_not_equal = num_non_zero - num_true_true

    if num_not_equal == 0:
        return 0.0
    else:
        return float(num_not_equal - num_true_true + n_features) / (
            num_not_equal + n_features
        )


@numba.njit()
def sparse_rogers_tanimoto(ind1, data1, ind2, data2, n_features):
    num_true_true = arr_intersect(ind1, ind2).shape[0]
    num_non_zero = arr_union(ind1, ind2).shape[0]
    num_not_equal = num_non_zero - num_true_true

    return (2.0 * num_not_equal) / (n_features + num_not_equal)


@numba.njit()
def sparse_russellrao(ind1, data1, ind2, data2, n_features):
    if ind1.shape[0] == ind2.shape[0] and np.all(ind1 == ind2):
        return 0.0

    num_true_true = arr_intersect(ind1, ind2).shape[0]

    if num_true_true == np.sum(data1 != 0) and num_true_true == np.sum(data2 != 0):
        return 0.0
    else:
        return float(n_features - num_true_true) / (n_features)


@numba.njit()
def sparse_sokal_michener(ind1, data1, ind2, data2, n_features):
    num_true_true = arr_intersect(ind1, ind2).shape[0]
    num_non_zero = arr_union(ind1, ind2).shape[0]
    num_not_equal = num_non_zero - num_true_true

    return (2.0 * num_not_equal) / (n_features + num_not_equal)


@numba.njit()
def sparse_sokal_sneath(ind1, data1, ind2, data2):
    num_true_true = arr_intersect(ind1, ind2).shape[0]
    num_non_zero = arr_union(ind1, ind2).shape[0]
    num_not_equal = num_non_zero - num_true_true

    if num_not_equal == 0.0:
        return 0.0
    else:
        return num_not_equal / (0.5 * num_true_true + num_not_equal)


@numba.njit()
def sparse_cosine(ind1, data1, ind2, data2):
    aux_inds, aux_data = sparse_mul(ind1, data1, ind2, data2)
    result = 0.0
    norm1 = norm(data1)
    norm2 = norm(data2)

    for i in range(aux_data.shape[0]):
        result += aux_data[i]

    if norm1 == 0.0 and norm2 == 0.0:
        return 0.0
    elif norm1 == 0.0 or norm2 == 0.0:
        return 1.0
    else:
        return 1.0 - (result / (norm1 * norm2))


@numba.njit()
def sparse_correlation(ind1, data1, ind2, data2, n_features):

    mu_x = 0.0
    mu_y = 0.0
    dot_product = 0.0

    if ind1.shape[0] == 0 and ind2.shape[0] == 0:
        return 0.0
    elif ind1.shape[0] == 0 or ind2.shape[0] == 0:
        return 1.0

    for i in range(data1.shape[0]):
        mu_x += data1[i]
    for i in range(data2.shape[0]):
        mu_y += data2[i]

    mu_x /= n_features
    mu_y /= n_features

    shifted_data1 = np.empty(data1.shape[0], dtype=np.float32)
    shifted_data2 = np.empty(data2.shape[0], dtype=np.float32)

    for i in range(data1.shape[0]):
        shifted_data1[i] = data1[i] - mu_x
    for i in range(data2.shape[0]):
        shifted_data2[i] = data2[i] - mu_y

    norm1 = np.sqrt(
        (norm(shifted_data1) ** 2) + (n_features - ind1.shape[0]) * (mu_x ** 2)
    )
    norm2 = np.sqrt(
        (norm(shifted_data2) ** 2) + (n_features - ind2.shape[0]) * (mu_y ** 2)
    )

    dot_prod_inds, dot_prod_data = sparse_mul(ind1, shifted_data1, ind2, shifted_data2)

    common_indices = set(dot_prod_inds)

    for i in range(dot_prod_data.shape[0]):
        dot_product += dot_prod_data[i]

    for i in range(ind1.shape[0]):
        if ind1[i] not in common_indices:
            dot_product -= shifted_data1[i] * (mu_y)

    for i in range(ind2.shape[0]):
        if ind2[i] not in common_indices:
            dot_product -= shifted_data2[i] * (mu_x)

    all_indices = arr_union(ind1, ind2)
    dot_product += mu_x * mu_y * (n_features - all_indices.shape[0])

    if norm1 == 0.0 and norm2 == 0.0:
        return 0.0
    elif dot_product == 0.0:
        return 1.0
    else:
        return 1.0 - (dot_product / (norm1 * norm2))


sparse_named_distances = {
    # general minkowski distances
    "euclidean": sparse_euclidean,
    "manhattan": sparse_manhattan,
    "l1": sparse_manhattan,
    "taxicab": sparse_manhattan,
    "chebyshev": sparse_chebyshev,
    "linf": sparse_chebyshev,
    "linfty": sparse_chebyshev,
    "linfinity": sparse_chebyshev,
    "minkowski": sparse_minkowski,
    # Other distances
    "canberra": sparse_canberra,
    # 'braycurtis': sparse_bray_curtis,
    # Binary distances
    "hamming": sparse_hamming,
    "jaccard": sparse_jaccard,
    "dice": sparse_dice,
    "matching": sparse_matching,
    "kulsinski": sparse_kulsinski,
    "rogerstanimoto": sparse_rogers_tanimoto,
    "russellrao": sparse_russellrao,
    "sokalmichener": sparse_sokal_michener,
    "sokalsneath": sparse_sokal_sneath,
    "cosine": sparse_cosine,
    "correlation": sparse_correlation,
}

sparse_need_n_features = (
    "hamming",
    "matching",
    "kulsinski",
    "rogerstanimoto",
    "russellrao",
    "sokalmichener",
    "correlation",
)
