# Author: Leland McInnes <leland.mcinnes@gmail.com>
# Enough simple sparse operations in numba to enable sparse UMAP
#
# License: BSD 3 clause
from __future__ import print_function

import locale

import numba
import numpy as np

from umap.utils import norm

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
            result_ind[nnz] = ind1[i1]
            result_data[nnz] = val
            nnz += 1
        i1 += 1

    while i2 < ind2.shape[0]:
        val = data2[i2]
        if val != 0:
            result_ind[nnz] = ind2[i2]
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
    right_complement=False,
    mix_weight=0.5,
):

    left_min = max(data1.min() / 2.0, 1.0e-8)
    if right_complement:
        right_min = min(
            max((1.0 - data2).min() / 2.0, 1.0e-8), 1e-4
        )  # All right vals may be large!
    else:
        right_min = min(
            max(data2.min() / 2.0, 1.0e-8), 1e-4
        )  # All right vals may be large!

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
                if right_complement:
                    right_val = 1.0 - data2[k]
                else:
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
def general_sset_union(
    indptr1,
    indices1,
    data1,
    indptr2,
    indices2,
    data2,
    result_row,
    result_col,
    result_val,
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

        result_val[idx] = left_val + right_val - left_val * right_val

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
    denom_inds, denom_data = sparse_sum(ind1, data1, ind2, data2)
    denom_data = np.abs(denom_data)

    if denom_data.shape[0] == 0:
        return 0.0

    denominator = np.sum(denom_data)

    if denominator == 0:
        return 0.0

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
def sparse_hellinger(ind1, data1, ind2, data2):
    aux_inds, aux_data = sparse_mul(ind1, data1, ind2, data2)
    result = 0.0
    norm1 = np.sum(data1)
    norm2 = np.sum(data2)
    sqrt_norm_prod = np.sqrt(norm1 * norm2)

    for i in range(aux_data.shape[0]):
        result += np.sqrt(aux_data[i])

    if norm1 == 0.0 and norm2 == 0.0:
        return 0.0
    elif norm1 == 0.0 or norm2 == 0.0:
        return 1.0
    elif result > sqrt_norm_prod:
        return 0.0
    else:
        return np.sqrt(1.0 - (result / sqrt_norm_prod))


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


@numba.njit()
def approx_log_Gamma(x):
    if x == 1:
        return 0
    #    x2= 1/(x*x);
    return (
        x * np.log(x) - x + 0.5 * np.log(2.0 * np.pi / x) + 1.0 / (x * 12.0)
    )  # + x2*(-1.0/360.0 + x2* (1.0/1260.0 + x2*(-1.0/(1680.0)


#                + x2*(1.0/1188.0 + x2*(-691.0/360360.0 + x2*(1.0/156.0 + x2*(-3617.0/122400.0 + x2*(43687.0/244188.0 + x2*(-174611.0/125400.0)
#                + x2*(77683.0/5796.0 + x2*(-236364091.0/1506960.0 + x2*(657931.0/300.0))))))))))))


@numba.njit()
def log_beta(x, y):
    a = min(x, y)
    b = max(x, y)
    if b < 5:
        value = -np.log(b)
        for i in range(1, int(a)):
            value += np.log(i) - np.log(b + i)
        return value
    else:
        return approx_log_Gamma(x) + approx_log_Gamma(y) - approx_log_Gamma(x + y)


@numba.njit()
def log_single_beta(x):
    return (
        np.log(2.0) * (-2.0 * x + 0.5) + 0.5 * np.log(2.0 * np.pi / x) + 0.125 / x
    )  # + x2*(-1.0/192.0 + x2* (1.0/640.0 + x2*(-17.0/(14336.0)


#                + x2*(31.0/18432.0 + x2*(-691.0/180224.0 + x2*(5461.0/425984.0 + x2*(-929569.0/15728640.0 + x2*(3189151.0/8912896.0 + x2*(-221930581.0/79691776.0)
#                + x2*(4722116521.0/176160768.0 + x2*(-968383680827.0/3087007744.0 + x2*(14717667114151.0/3355443200.0 ))))))))))))


@numba.njit()
def sparse_ll_dirichlet(ind1, data1, ind2, data2):
    # The probability of rolling data2 in sum(data2) trials on a die that rolled data1 in sum(data1) trials
    n1 = np.sum(data1)
    n2 = np.sum(data2)

    if n1 == 0 and n2 == 0:
        return 0.0
    elif n1 == 0 or n2 == 0:
        return 1e8

    log_b = 0.0
    i1 = 0
    i2 = 0
    while i1 < ind1.shape[0] and i2 < ind2.shape[0]:
        j1 = ind1[i1]
        j2 = ind2[i2]

        if j1 == j2:
            if data1[i1] * data2[i2] != 0:
                log_b += log_beta(data1[i1], data2[i2])
            i1 += 1
            i2 += 1
        elif j1 < j2:
            i1 += 1
        else:
            i2 += 1

    self_denom1 = 0.0
    for d1 in data1:

        self_denom1 += log_single_beta(d1)

    self_denom2 = 0.0
    for d2 in data2:
        self_denom2 += log_single_beta(d2)

    return np.sqrt(
        1.0 / n2 * (log_b - log_beta(n1, n2) - (self_denom2 - log_single_beta(n2)))
        + 1.0 / n1 * (log_b - log_beta(n2, n1) - (self_denom1 - log_single_beta(n1)))
    )


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
    "ll_dirichlet": sparse_ll_dirichlet,
    "braycurtis": sparse_bray_curtis,
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
    "hellinger": sparse_hellinger,
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

SPARSE_SPECIAL_METRICS = {
    sparse_hellinger: "hellinger",
    sparse_ll_dirichlet: "ll_dirichlet",
}
