import numba
import numpy as np
from tqdm.auto import tqdm

import umap.distances as dist
from umap.utils import adaptive_bucket_sort, tau_rand_int


@numba.njit(inline="always")
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


@numba.njit(
    "f4(f4[::1],f4[::1])",
    fastmath=True,
    cache=True,
    locals={
        "result": numba.types.float32,
        "diff": numba.types.float32,
        "dim": numba.types.intp,
        "i": numba.types.intp,
    },
)
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
    dim = x.shape[0]
    for i in range(dim):
        diff = x[i] - y[i]
        result += diff * diff

    return result


def _optimize_layout_euclidean_single_epoch(
    head_embedding,
    tail_embedding,
    head,
    tail,
    n_vertices,
    epochs_per_sample,
    a,
    b,
    rng_state_per_sample,
    gamma,
    dim,
    move_other,
    alpha,
    epochs_per_negative_sample,
    epoch_of_next_negative_sample,
    epoch_of_next_sample,
    n,
    densmap_flag,
    dens_phi_sum,
    dens_re_sum,
    dens_re_cov,
    dens_re_std,
    dens_re_mean,
    dens_lambda,
    dens_R,
    dens_mu,
    dens_mu_tot,
):
    for i in numba.prange(epochs_per_sample.shape[0]):
        if epoch_of_next_sample[i] <= n:
            j = head[i]
            k = tail[i]

            current = head_embedding[j]
            other = tail_embedding[k]

            dist_squared = rdist(current, other)

            if densmap_flag:
                phi = 1.0 / (1.0 + a * pow(dist_squared, b))
                dphi_term = (
                    a * b * pow(dist_squared, b - 1) / (1.0 + a * pow(dist_squared, b))
                )

                q_jk = phi / dens_phi_sum[k]
                q_kj = phi / dens_phi_sum[j]

                drk = q_jk * (
                    (1.0 - b * (1 - phi)) / np.exp(dens_re_sum[k]) + dphi_term
                )
                drj = q_kj * (
                    (1.0 - b * (1 - phi)) / np.exp(dens_re_sum[j]) + dphi_term
                )

                re_std_sq = dens_re_std * dens_re_std
                weight_k = (
                    dens_R[k]
                    - dens_re_cov * (dens_re_sum[k] - dens_re_mean) / re_std_sq
                )
                weight_j = (
                    dens_R[j]
                    - dens_re_cov * (dens_re_sum[j] - dens_re_mean) / re_std_sq
                )

                grad_cor_coeff = (
                    dens_lambda
                    * dens_mu_tot
                    * (weight_k * drk + weight_j * drj)
                    / (dens_mu[i] * dens_re_std)
                    / n_vertices
                )

            if dist_squared > 0.0:
                grad_coeff = -2.0 * a * b * pow(dist_squared, b - 1.0)
                grad_coeff /= a * pow(dist_squared, b) + 1.0
            else:
                grad_coeff = 0.0

            for d in range(dim):
                grad_d = clip(grad_coeff * (current[d] - other[d]))

                if densmap_flag:
                    # FIXME: grad_cor_coeff might be referenced before assignment

                    grad_d += clip(2 * grad_cor_coeff * (current[d] - other[d]))

                current[d] += grad_d * alpha
                if move_other:
                    other[d] += -grad_d * alpha

            epoch_of_next_sample[i] += epochs_per_sample[i]

            n_neg_samples = int(
                (n - epoch_of_next_negative_sample[i]) / epochs_per_negative_sample[i]
            )

            for p in range(n_neg_samples):
                k = tau_rand_int(rng_state_per_sample[j]) % n_vertices

                other = tail_embedding[k]

                dist_squared = rdist(current, other)

                if dist_squared > 0.0:
                    grad_coeff = 2.0 * gamma * b
                    grad_coeff /= (0.001 + dist_squared) * (
                        a * pow(dist_squared, b) + 1
                    )
                elif j == k:
                    continue
                else:
                    grad_coeff = 0.0

                for d in range(dim):
                    if grad_coeff > 0.0:
                        grad_d = clip(grad_coeff * (current[d] - other[d]))
                    else:
                        grad_d = 0
                    current[d] += grad_d * alpha

            epoch_of_next_negative_sample[i] += (
                n_neg_samples * epochs_per_negative_sample[i]
            )


@numba.njit(
    "void(f4[:, ::1], f4[:, ::1], i4[::1], i4[::1], i8, f8[::1], f8, f8, f8, i8, f8, f8[::1], f8[::1], f8[::1], i8, f4[:, ::1], i4[::1], i4[::1], i8)",
    fastmath=True,
    parallel=True,
    locals={
        "updates": numba.types.float32[:, ::1],
        "from_node": numba.types.intp,
        "to_node": numba.types.intp,
        "raw_index": numba.types.intp,
        "dist_squared": numba.types.float32,
        "grad_coeff": numba.types.float32,
        "grad_d": numba.types.float32,
    },
)
def optimize_layout_euclidean_single_epoch_fast(
    head_embedding,
    tail_embedding,
    csr_indptr,
    csr_indices,
    n_vertices,
    epochs_per_sample,
    a,
    b,
    gamma,
    dim,
    alpha,
    epochs_per_negative_sample,
    epoch_of_next_negative_sample,
    epoch_of_next_sample,
    n,
    updates,
    from_node_order,
    to_node_order,
    block_size=4096,
):
    n_from_vertices = csr_indptr.shape[0] - 1
    for block_start in range(0, n_from_vertices, block_size):
        block_end = min(block_start + block_size, n_from_vertices)
        for node_idx in numba.prange(block_start, block_end):
            from_node = from_node_order[node_idx]
            current = head_embedding[from_node]

            for raw_index in range(csr_indptr[from_node], csr_indptr[from_node + 1]):
                if epoch_of_next_sample[raw_index] <= n:
                    to_node = csr_indices[raw_index]
                    other = tail_embedding[to_node]

                    dist_squared = rdist(current, other)

                    if dist_squared > 0.0:
                        grad_coeff = -2.0 * a * b * pow(dist_squared, b - 1.0)
                        grad_coeff /= a * pow(dist_squared, b) + 1.0
                        for d in range(dim):
                            grad_d = grad_coeff * (current[d] - other[d])
                            updates[from_node, d] += grad_d * alpha

                    epoch_of_next_sample[raw_index] += epochs_per_sample[raw_index]

                    n_neg_samples = int(
                        (n - epoch_of_next_negative_sample[raw_index])
                        / epochs_per_negative_sample[raw_index]
                    )

                    for p in range(n_neg_samples):
                        to_node = to_node_order[(raw_index * (n + p + 1)) % n_vertices]
                        other = tail_embedding[to_node]

                        dist_squared = rdist(current, other)

                        if dist_squared > 0.0:
                            grad_coeff = 2.0 * gamma * b
                            grad_coeff /= (0.001 + dist_squared) * (
                                a * pow(dist_squared, b) + 1
                            )

                            if grad_coeff > 0.0:
                                for d in range(dim):
                                    grad_d = clip(grad_coeff * (current[d] - other[d]))
                                    updates[from_node, d] += grad_d * alpha

                    epoch_of_next_negative_sample[raw_index] += (
                        n_neg_samples * epochs_per_negative_sample[raw_index]
                    )

        for node_idx in numba.prange(block_start, block_end):
            from_node = from_node_order[node_idx]
            for d in range(dim):
                head_embedding[from_node, d] += updates[from_node, d]


@numba.njit(inline="always")
def get_range_limits(center, range_size, array_length):
    half_size = range_size // 2
    start = center - half_size

    # Handle boundary conditions
    if start < 0:
        start = 0
    elif start + range_size > array_length:
        start = max(0, array_length - range_size)

    return start


@numba.njit(
    "void(f4[:, ::1], f4[:, ::1], i4[::1], i4[::1], i8, f8[::1], f8, f8, f8, i8, f8, f8[::1], f8[::1], f8[::1], i8, f4[:, ::1], f4[:, ::1], f4[:, ::1], f8, f8, i4[::1], i4[::1], i8, i8)",
    fastmath=True,
    parallel=True,
    locals={
        "updates": numba.types.float32[:, ::1],
        "from_node": numba.types.intp,
        "to_node": numba.types.intp,
        "raw_index": numba.types.intp,
        "dist_squared": numba.types.float32,
        "grad_coeff": numba.types.float32,
        "grad_d": numba.types.float32,
    },
)
def optimize_layout_euclidean_single_epoch_adam(
    head_embedding,
    tail_embedding,
    csr_indptr,
    csr_indices,
    n_vertices,
    epochs_per_sample,
    a,
    b,
    gamma,
    dim,
    alpha,
    epochs_per_negative_sample,
    epoch_of_next_negative_sample,
    epoch_of_next_sample,
    n,
    updates,
    adam_m,
    adam_v,
    beta1,
    beta2,
    from_node_order,
    to_node_order,
    block_size=256,
    negative_selection_range=200_000,
):
    n_from_vertices = csr_indptr.shape[0] - 1
    negative_selection_range = max(200, min(n_vertices, negative_selection_range))
    # negative_sample_scaling = np.pow(negative_selection_range / n_vertices, 0.1667)
    negative_sample_scaling = 1.0
    # average_negative_rdist = 0.0
    # total_negative_samples = 0
    # rep_grad_coeff_average = 0.0
    c = gamma
    for block_start in range(0, n_from_vertices, block_size):
        block_end = min(block_start + block_size, n_from_vertices)
        # for node_idx in numba.prange(block_start, block_end):
        for raw_idx in numba.prange(block_start, block_end):
            node_idx = to_node_order[raw_idx]
            from_node = from_node_order[node_idx]
            current = head_embedding[from_node]

            for raw_index in range(csr_indptr[from_node], csr_indptr[from_node + 1]):
                if epoch_of_next_sample[raw_index] <= n:
                    to_node = csr_indices[raw_index]
                    other = tail_embedding[to_node]

                    dist_squared = rdist(current, other)

                    if dist_squared > 0.0:
                        grad_coeff = -2.0 * a * b * pow(dist_squared, b - 1.0)
                        grad_coeff /= a * pow(dist_squared, b) + 1.0
                        for d in range(dim):
                            grad_d = grad_coeff * (current[d] - other[d])
                            updates[from_node, d] += grad_d

                    epoch_of_next_sample[raw_index] += epochs_per_sample[raw_index]

                    n_neg_samples = int(
                        (n - epoch_of_next_negative_sample[raw_index])
                        / epochs_per_negative_sample[raw_index]
                    )

                    for p in range(n_neg_samples):
                        to_node_raw_selection = (
                            raw_index * (n + p + 1)
                        ) % negative_selection_range
                        range_start = get_range_limits(
                            node_idx, negative_selection_range, n_vertices
                        )
                        # to_node_subset = to_node_order[range_start:range_end]
                        # to_node = to_node_subset[to_node_raw_selection]
                        to_node = from_node_order[
                            (range_start + to_node_raw_selection) % n_vertices
                        ]
                        # to_node = to_node_order[(raw_index * (n + p + 1)) % n_vertices]
                        other = tail_embedding[to_node]

                        dist_squared = rdist(current, other)
                        # average_negative_rdist += dist_squared
                        # total_negative_samples += 1

                        if dist_squared > 0.0:
                            grad_coeff = negative_sample_scaling * 2.0 * gamma * b
                            grad_coeff /= (0.001 + dist_squared) * (
                                a * pow(dist_squared, b) + 1
                            )
                            # if from_node % 100 == 0:
                            #     rep_grad_coeff_average += grad_coeff

                            if grad_coeff > 0.0:
                                # for d in range(dim):
                                #     grad_d = param_clip(
                                #         grad_coeff * (current[d] - other[d]), -1, 1
                                #     )
                                #     updates[from_node, d] += grad_d
                                grad_norm = np.sqrt(
                                    grad_coeff * grad_coeff * dist_squared
                                )
                                # if grad_norm > 16.0:
                                scale = c * np.tanh(grad_norm / c) / grad_norm
                                # else:
                                #     scale = 1.0
                                for d in range(dim):
                                    updates[from_node, d] += (
                                        grad_coeff * (current[d] - other[d]) * scale
                                    )

                    epoch_of_next_negative_sample[raw_index] += (
                        n_neg_samples * epochs_per_negative_sample[raw_index]
                    )

        for node_idx in numba.prange(block_start, block_end):
            from_node = from_node_order[node_idx]
            for d in range(dim):
                if updates[from_node, d] != 0.0:
                    adam_m[from_node, d] = (
                        beta1 * adam_m[from_node, d]
                        + (1.0 - beta1) * updates[from_node, d]
                    )
                    adam_v[from_node, d] = (
                        beta2 * adam_v[from_node, d]
                        + (1.0 - beta2) * updates[from_node, d] ** 2
                    )
                    m_est = adam_m[from_node, d] / (1.0 - pow(beta1, n))
                    v_est = adam_v[from_node, d] / (1.0 - pow(beta2, n))
                    head_embedding[from_node, d] += (
                        alpha * m_est / (np.sqrt(v_est) + 1e-4)
                    )


@numba.njit(
    fastmath=True,
    parallel=True,
    locals={
        "updates": numba.types.float32[:, ::1],
        "from_node": numba.types.intp,
        "to_node": numba.types.intp,
        "raw_index": numba.types.intp,
        "dist_squared": numba.types.float32,
        "grad_coeff": numba.types.float32,
        "grad_cor_coeff": numba.types.float32,
        "grad_d": numba.types.float32,
        "dphi_term": numba.types.float32,
        "phi": numba.types.float32,
        "q_jk": numba.types.float32,
        "q_kj": numba.types.float32,
        "drk": numba.types.float32,
        "drj": numba.types.float32,
    },
)
def optimize_layout_euclidean_single_epoch_fast_densmap(
    head_embedding,
    tail_embedding,
    csr_indptr,
    csr_indices,
    n_vertices,
    epochs_per_sample,
    a,
    b,
    gamma,
    dim,
    alpha,
    epochs_per_negative_sample,
    epoch_of_next_negative_sample,
    epoch_of_next_sample,
    n,
    dens_phi_sum,
    dens_re_sum,
    dens_re_cov,
    dens_re_std,
    dens_re_mean,
    dens_lambda,
    dens_R,
    dens_mu,
    dens_mu_tot,
    updates,
    node_order,
    block_size=256,
    densmap_flag=True,
):
    for block_start in range(0, head_embedding.shape[0], block_size):
        block_end = min(block_start + block_size, head_embedding.shape[0])
        for node_idx in numba.prange(block_start, block_end):
            from_node = node_order[node_idx]
            current = head_embedding[from_node]

            for raw_index in range(csr_indptr[from_node], csr_indptr[from_node + 1]):
                if epoch_of_next_sample[raw_index] <= n:
                    to_node = csr_indices[raw_index]
                    other = tail_embedding[to_node]

                    dist_squared = rdist(current, other)

                    if densmap_flag:
                        phi = 1.0 / (1.0 + a * pow(dist_squared, b))
                        dphi_term = (
                            a
                            * b
                            * pow(dist_squared, b - 1)
                            / (1.0 + a * pow(dist_squared, b))
                        )

                        q_jk = phi / dens_phi_sum[to_node]
                        q_kj = phi / dens_phi_sum[from_node]

                        drk = q_jk * (
                            (1.0 - b * (1 - phi)) / np.exp(dens_re_sum[to_node])
                            + dphi_term
                        )
                        drj = q_kj * (
                            (1.0 - b * (1 - phi)) / np.exp(dens_re_sum[from_node])
                            + dphi_term
                        )

                        re_std_sq = dens_re_std * dens_re_std
                        weight_k = (
                            dens_R[to_node]
                            - dens_re_cov
                            * (dens_re_sum[to_node] - dens_re_mean)
                            / re_std_sq
                        )
                        weight_j = (
                            dens_R[from_node]
                            - dens_re_cov
                            * (dens_re_sum[from_node] - dens_re_mean)
                            / re_std_sq
                        )

                        grad_cor_coeff = (
                            dens_lambda
                            * dens_mu_tot
                            * (weight_k * drk + weight_j * drj)
                            / (dens_mu[raw_index] * dens_re_std)
                            / n_vertices
                        )
                    else:
                        grad_cor_coeff = 0.0

                    if dist_squared > 0.0:
                        grad_coeff = -2.0 * a * b * pow(dist_squared, b - 1.0)
                        grad_coeff /= a * pow(dist_squared, b) + 1.0
                        for d in range(dim):
                            grad_d = clip(grad_coeff * (current[d] - other[d]))
                            grad_d += clip(2 * grad_cor_coeff * (current[d] - other[d]))

                            updates[from_node, d] += grad_d * alpha

                    epoch_of_next_sample[raw_index] += epochs_per_sample[raw_index]

                    n_neg_samples = int(
                        (n - epoch_of_next_negative_sample[raw_index])
                        / epochs_per_negative_sample[raw_index]
                    )

                    for p in range(n_neg_samples):
                        to_node = node_order[(raw_index * (n + p + 1)) % n_vertices]

                        other = tail_embedding[to_node]

                        dist_squared = rdist(current, other)

                        if dist_squared > 0.0:
                            grad_coeff = 2.0 * gamma * b
                            grad_coeff /= (0.001 + dist_squared) * (
                                a * pow(dist_squared, b) + 1
                            )

                            for d in range(dim):
                                if grad_coeff > 0.0:
                                    grad_d = clip(grad_coeff * (current[d] - other[d]))
                                    updates[from_node, d] += grad_d * alpha

                    epoch_of_next_negative_sample[raw_index] += (
                        n_neg_samples * epochs_per_negative_sample[raw_index]
                    )

        for node_idx in numba.prange(block_start, block_end):
            from_node = node_order[node_idx]
            for d in range(dim):
                head_embedding[from_node, d] += updates[from_node, d]


@numba.njit(
    fastmath=True,
    parallel=True,
    locals={
        "updates": numba.types.float32[:, ::1],
        "from_node": numba.types.intp,
        "to_node": numba.types.intp,
        "raw_index": numba.types.intp,
        "dist_squared": numba.types.float32,
        "grad_coeff": numba.types.float32,
        "grad_d": numba.types.float32,
    },
)
def optimize_layout_euclidean_single_epoch_adam_densmap(
    head_embedding,
    tail_embedding,
    csr_indptr,
    csr_indices,
    n_vertices,
    epochs_per_sample,
    a,
    b,
    gamma,
    dim,
    alpha,
    epochs_per_negative_sample,
    epoch_of_next_negative_sample,
    epoch_of_next_sample,
    n,
    dens_phi_sum,
    dens_re_sum,
    dens_re_cov,
    dens_re_std,
    dens_re_mean,
    dens_lambda,
    dens_R,
    dens_mu,
    dens_mu_tot,
    updates,
    adam_m,
    adam_v,
    beta1,
    beta2,
    node_order,
    block_size=256,
    densmap_flag=True,
):
    for block_start in range(0, n_vertices, block_size):
        block_end = min(block_start + block_size, n_vertices)
        for node_idx in numba.prange(block_start, block_end):
            from_node = node_order[node_idx]
            current = head_embedding[from_node]

            for raw_index in range(csr_indptr[from_node], csr_indptr[from_node + 1]):
                if epoch_of_next_sample[raw_index] <= n:
                    to_node = csr_indices[raw_index]
                    other = tail_embedding[to_node]

                    dist_squared = rdist(current, other) / 2

                    if densmap_flag:
                        phi = 1.0 / (1.0 + a * pow(dist_squared, b))
                        dphi_term = (
                            a
                            * b
                            * pow(dist_squared, b - 1)
                            / (1.0 + a * pow(dist_squared, b))
                        )

                        q_jk = phi / dens_phi_sum[to_node]
                        q_kj = phi / dens_phi_sum[from_node]

                        drk = q_jk * (
                            (1.0 - b * (1 - phi)) / np.exp(dens_re_sum[to_node])
                            + dphi_term
                        )
                        drj = q_kj * (
                            (1.0 - b * (1 - phi)) / np.exp(dens_re_sum[from_node])
                            + dphi_term
                        )

                        re_std_sq = dens_re_std * dens_re_std
                        weight_k = (
                            dens_R[to_node]
                            - dens_re_cov
                            * (dens_re_sum[to_node] - dens_re_mean)
                            / re_std_sq
                        )
                        weight_j = (
                            dens_R[from_node]
                            - dens_re_cov
                            * (dens_re_sum[from_node] - dens_re_mean)
                            / re_std_sq
                        )

                        grad_cor_coeff = (
                            dens_lambda
                            * dens_mu_tot
                            * (weight_k * drk + weight_j * drj)
                            / (dens_mu[raw_index] * dens_re_std)
                            / n_vertices
                        )
                    else:
                        grad_cor_coeff = 0.0

                    if dist_squared > 0.0:
                        grad_coeff = -2.0 * a * b * pow(dist_squared, b - 1.0)
                        grad_coeff /= a * pow(dist_squared, b) + 1.0
                        for d in range(dim):
                            grad_d = clip(grad_coeff * (current[d] - other[d]))
                            grad_d += clip(2 * grad_cor_coeff * (current[d] - other[d]))
                            updates[from_node, d] += grad_d

                    epoch_of_next_sample[raw_index] += epochs_per_sample[raw_index]

                    n_neg_samples = int(
                        (n - epoch_of_next_negative_sample[raw_index])
                        / epochs_per_negative_sample[raw_index]
                    )

                    for p in range(n_neg_samples):
                        to_node = node_order[(raw_index * (n + p + 1)) % n_vertices]

                        other = tail_embedding[to_node]

                        dist_squared = rdist(current, other) / 4

                        if dist_squared > 0.0:
                            grad_coeff = 2.0 * gamma * b
                            grad_coeff /= (0.001 + dist_squared) * (
                                a * pow(dist_squared, b) + 1
                            )

                            if grad_coeff > 0.0:
                                for d in range(dim):
                                    grad_d = clip(grad_coeff * (current[d] - other[d]))
                                    updates[from_node, d] += grad_d

                    epoch_of_next_negative_sample[raw_index] += (
                        n_neg_samples * epochs_per_negative_sample[raw_index]
                    )

        for node_idx in numba.prange(block_start, block_end):
            from_node = node_order[node_idx]
            for d in range(dim):
                if updates[from_node, d] != 0.0:
                    adam_m[from_node, d] = (
                        beta1 * adam_m[from_node, d]
                        + (1.0 - beta1) * updates[from_node, d]
                    )
                    adam_v[from_node, d] = (
                        beta2 * adam_v[from_node, d]
                        + (1.0 - beta2) * updates[from_node, d] ** 2
                    )
                    m_est = adam_m[from_node, d] / (1.0 - pow(beta1, n))
                    v_est = adam_v[from_node, d] / (1.0 - pow(beta2, n))
                    head_embedding[from_node, d] += (
                        alpha * m_est / (np.sqrt(v_est) + 1e-4)
                    )


@numba.njit(
    "void(f4[:, ::1], f4[:, ::1], i4[::1], i4[::1], f8, f8, f4[::1], f4[::1])",
    fastmath=True,
    parallel=True,
    cache=True,
    locals={
        "i": numba.types.intp,
        "j": numba.types.intp,
        "k": numba.types.intp,
        "current": numba.types.float32[::1],
        "other": numba.types.float32[::1],
        "dist_squared": numba.types.float32,
        "phi": numba.types.float32,
    },
)
def _optimize_layout_euclidean_densmap_epoch_init_coo(
    head_embedding,
    tail_embedding,
    head,
    tail,
    a,
    b,
    re_sum,
    phi_sum,
):
    re_sum.fill(0)
    phi_sum.fill(0)

    for i in numba.prange(head.shape[0]):
        j = head[i]
        k = tail[i]

        current = head_embedding[j]
        other = tail_embedding[k]
        dist_squared = rdist(current, other)

        phi = 1.0 / (1.0 + a * pow(dist_squared, b))

        re_sum[j] += phi * dist_squared
        re_sum[k] += phi * dist_squared
        phi_sum[j] += phi
        phi_sum[k] += phi

    epsilon = 1e-8
    for i in range(re_sum.shape[0]):
        re_sum[i] = np.log(epsilon + (re_sum[i] / phi_sum[i]))


@numba.njit(
    "void(f4[:, ::1], f4[:, ::1], i4[::1], i4[::1], f8, f8, f4[::1], f4[::1])",
    fastmath=True,
    parallel=True,
    cache=True,
    locals={
        "j": numba.types.intp,
        "k": numba.types.intp,
        "raw_index": numba.types.intp,
        "current": numba.types.float32[::1],
        "other": numba.types.float32[::1],
        "dist_squared": numba.types.float32,
        "phi": numba.types.float32,
    },
)
def _optimize_layout_euclidean_densmap_epoch_init_csr(
    head_embedding,
    tail_embedding,
    indptr,
    indices,
    a,
    b,
    re_sum,
    phi_sum,
):
    re_sum.fill(0)
    phi_sum.fill(0)

    for j in numba.prange(indptr.shape[0] - 1):
        for raw_index in range(indptr[j], indptr[j + 1]):
            k = indices[raw_index]

            current = head_embedding[j]
            other = tail_embedding[k]
            dist_squared = rdist(current, other)

            phi = 1.0 / (1.0 + a * pow(dist_squared, b))

            re_sum[j] += phi * dist_squared
            re_sum[k] += phi * dist_squared
            phi_sum[j] += phi
            phi_sum[k] += phi

    epsilon = 1e-8
    for i in range(re_sum.shape[0]):
        re_sum[i] = np.log(epsilon + (re_sum[i] / phi_sum[i]))


_nb_optimize_layout_euclidean_single_epoch = numba.njit(
    _optimize_layout_euclidean_single_epoch, fastmath=True, parallel=False
)

_nb_optimize_layout_euclidean_single_epoch_parallel = numba.njit(
    _optimize_layout_euclidean_single_epoch, fastmath=True, parallel=True
)


def _get_optimize_layout_euclidean_single_epoch_fn(parallel: bool = False):
    if parallel:
        return _nb_optimize_layout_euclidean_single_epoch_parallel
    else:
        return _nb_optimize_layout_euclidean_single_epoch


def _create_alpha_schedule(optimizer, n_epochs, initial_alpha, good_initialization):
    """Create alpha (learning rate) schedule based on optimizer and initialization quality."""
    if optimizer == "compatibility":
        return np.linspace(initial_alpha, 0.0, n_epochs, endpoint=False)

    elif optimizer in ["standard", "densmap_standard"]:
        if good_initialization:
            n_warm_up_epochs = int(max(200, n_epochs / 8))
            raw_alpha_schedule = np.asarray(
                [
                    (1.0 - (float(n) / float(n_warm_up_epochs))) ** 2
                    for n in range(n_warm_up_epochs)
                ]
                + [0.0] * (n_epochs - n_warm_up_epochs)
            )
            return 0.25 * raw_alpha_schedule * initial_alpha + 0.005
        else:
            raw_alpha_schedule = np.asarray(
                [
                    0.25 * (1.0 - (float(n) / float(n_epochs))) ** 2
                    for n in range(n_epochs)
                ]
            )
            return raw_alpha_schedule * initial_alpha

    elif optimizer in ["adam", "densmap_adam"]:
        if good_initialization:
            n_warm_up_epochs = int(min(100, n_epochs / 8))
        else:
            n_warm_up_epochs = int(min(n_epochs / 2, 100))

        if good_initialization:
            return np.concatenate(
                [
                    [
                        (0.5 * initial_alpha - 0.1)
                        * (1.0 - (float(n) / float(n_warm_up_epochs))) ** 2
                        + 0.1
                        for n in range(n_warm_up_epochs)
                    ],
                    [
                        0.15
                        * (
                            1.0
                            - (
                                float(n - n_warm_up_epochs)
                                / float(n_epochs - n_warm_up_epochs)
                            )
                        )
                        + 0.05
                        for n in range(n_warm_up_epochs, n_epochs)
                    ],
                ]
            )
        else:
            return np.concatenate(
                [
                    [
                        (2.0 * initial_alpha - 0.1)
                        * (1.0 - (float(n) / float(n_warm_up_epochs))) ** 2
                        + 0.1
                        for n in range(n_warm_up_epochs)
                    ],
                    [
                        0.15
                        * (
                            1.0
                            - (
                                float(n - n_warm_up_epochs)
                                / float(n_epochs - n_warm_up_epochs)
                            )
                        )
                        + 0.05
                        for n in range(n_warm_up_epochs, n_epochs)
                    ],
                ]
            )


def _create_momentum_schedule(optimizer, n_epochs, good_initialization):
    """Create momentum schedule based on optimizer and initialization quality."""
    if optimizer in ["adam", "densmap_adam"]:
        return np.zeros(n_epochs, dtype=np.float32)

    elif optimizer in ["standard", "densmap_standard"]:
        if good_initialization:
            n_warm_up_epochs = int(max(200, n_epochs / 8))
            raw_alpha_schedule = np.asarray(
                [
                    (1.0 - (float(n) / float(n_warm_up_epochs))) ** 2
                    for n in range(n_warm_up_epochs)
                ]
                + [0.0] * (n_epochs - n_warm_up_epochs)
            )
            return np.asarray(
                [0.5 * (1.0 - raw_alpha_schedule[n]) for n in range(n_warm_up_epochs)]
                + [0.5] * (n_epochs - n_warm_up_epochs)
            )
        else:
            raw_alpha_schedule = np.asarray(
                [
                    0.25 * (1.0 - (float(n) / float(n_epochs))) ** 2
                    for n in range(n_epochs)
                ]
            )
            return np.asarray(
                [0.5 * (1.0 - raw_alpha_schedule[n]) for n in range(n_epochs)]
            )

    return np.zeros(n_epochs, dtype=np.float32)


def _create_adam_schedules(
    optimizer,
    n_epochs,
    good_initialization,
    gamma,
    n_vertices,
    negative_selection_range,
):
    """Create beta1, beta2, and gamma schedules for Adam optimizer."""
    if optimizer not in ["adam", "densmap_adam"]:
        return None, None, None, None

    if good_initialization:
        n_warm_up_epochs = int(min(100, n_epochs / 4))
    else:
        n_warm_up_epochs = int(min(n_epochs // 2, 100))  # Use n_epochs/2 but cap at 100

    beta1_schedule = np.concatenate(
        [
            [
                0.2 + (0.7 * (float(n) / float(n_warm_up_epochs)))
                for n in range(n_warm_up_epochs)
            ],
            np.full(n_epochs - n_warm_up_epochs, 0.9),
        ]
    )

    beta2_schedule = np.concatenate(
        [
            [
                0.79 + (0.2 * ((float(n) / float(n_warm_up_epochs))))
                for n in range(n_warm_up_epochs)
            ],
            np.full(n_epochs - n_warm_up_epochs, 0.99),
        ]
    )

    if good_initialization:
        # gamma_schedule = (
        #     np.concatenate(
        #         [
        #             [
        #                 0.75 * np.sqrt(float(n) / float(n_warm_up_epochs))
        #                 for n in range(n_warm_up_epochs)
        #             ],
        #             [
        #                 0.25
        #                 * (
        #                     1.0
        #                     - (
        #                         float(n - n_warm_up_epochs)
        #                         / float(n_epochs - n_warm_up_epochs)
        #                     )
        #                 )
        #                 + 0.5
        #                 for n in range(n_warm_up_epochs, n_epochs)
        #             ],
        #         ]
        #     )
        #     * gamma
        # )
        # gamma_schedule = np.linspace(gamma, 0.25, n_epochs, dtype=np.float32)
        gamma_schedule = np.full(n_epochs, 0.5 * gamma, dtype=np.float32)
    else:
        # gamma_schedule = (
        #     np.concatenate(
        #         [
        #             [
        #                 1.0 * np.sqrt(float(n) / float(n_warm_up_epochs))
        #                 for n in range(n_warm_up_epochs)
        #             ],
        #             [
        #                 0.25
        #                 * (
        #                     1.0
        #                     - float(n - n_warm_up_epochs)
        #                     / float(n_epochs - n_warm_up_epochs)
        #                 )
        #                 + 0.75
        #                 for n in range(n_warm_up_epochs, n_epochs)
        #             ],
        #         ]
        #     )
        #     * gamma
        # )
        # gamma_schedule = np.linspace(gamma, 0.25, n_epochs, dtype=np.float32)
        gamma_schedule = np.full(n_epochs, gamma, dtype=np.float32)
        # gamma_schedule = (
        #     np.linspace(1, 0, n_epochs, dtype=np.float32) ** 2 * gamma + gamma
        # )

    # negative_selection_range = min(n_vertices, max(negative_selection_range, 1024))
    # selection_range_warmup = min(max(n_warm_up_epochs, 200), n_epochs)
    # selection_range_warmup = n_warm_up_epochs
    # negative_selection_range_schedule = np.concatenate(
    #     [
    #         (
    #             (np.linspace(1, 0, selection_range_warmup))
    #             * (n_vertices - negative_selection_range)
    #             + negative_selection_range
    #         ).astype(np.int32),
    #         np.full(
    #             n_epochs - selection_range_warmup,
    #             negative_selection_range,
    #             dtype=np.int32,
    #         ),
    #     ]
    # )
    # negative_selection_range_schedule = np.linspace(
    #     n_vertices,
    #     negative_selection_range,
    #     n_epochs,
    #     dtype=np.int32,
    # )
    negative_selection_range_schedule = np.full(
        n_epochs, negative_selection_range, dtype=np.int32
    )
    # scale = n_vertices / negative_selection_range
    # negative_selection_range_schedule = np.round(
    #     n_vertices / np.linspace(1, scale, n_epochs, dtype=np.float32)
    # ).astype(np.int32)

    return (
        beta1_schedule,
        beta2_schedule,
        gamma_schedule,
        negative_selection_range_schedule,
    )


def optimize_layout_euclidean(
    head_embedding,
    tail_embedding,
    head,
    tail,
    n_epochs,
    n_vertices,
    epochs_per_sample,
    a,
    b,
    rng_state,
    gamma=1.0,
    initial_alpha=1.0,
    negative_sample_rate=5.0,
    parallel=False,
    verbose=False,
    densmap_kwds=None,
    tqdm_kwds=None,
    move_other=False,
    csr_indptr=None,
    csr_indices=None,
    optimizer="adam",
    good_initialization=False,
    random_state=None,
    negative_selection_range=200_000,
):
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
    n_epochs: int, or list of int
        The number of training epochs to use in optimization, or a list of
        epochs at which to save the embedding. In case of a list, the optimization
        will use the maximum number of epochs in the list, and will return a list
        of embedding in the order of increasing epoch, regardless of the order in
        the epoch list.
    n_vertices: int
        The number of vertices (0-simplices) in the dataset.
    epochs_per_sample: array of shape (n_1_simplices)
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
    parallel: bool (optional, default False)
        Whether to run the computation using numba parallel.
        Running in parallel is non-deterministic, and is not used
        if a random seed has been set, to ensure reproducibility.
    verbose: bool (optional, default False)
        Whether to report information on the current progress of the algorithm.
    densmap_kwds: dict (optional, default None)
        Auxiliary data for densMAP
    tqdm_kwds: dict (optional, default None)
        Keyword arguments for tqdm progress bar.
    move_other: bool (optional, default False)
        Whether to adjust tail_embedding alongside head_embedding
    csr_indptr: array of int (optional, default None)
        CSR indptr array for the graph of 1-simplices.
        If provided, the optimization will use a faster version
        of the optimization code that does not require the head and tail arrays.
    csr_indices: array of int (optional, default None)
        CSR indices array for the graph of 1-simplices.
    optimizer: str (optional, default "standard")
        The optimizer to use for the optimization. Can be one of "standard", "adam",
        "compatibility", "densmap_adam" or "densmap_standard".
    good_initialization: bool (optional, default False)
        Whether the initial embedding is already a good representation of the data.
        If True, the optimization will use a different learning rate schedules etc.
        This is only used if optimizer is "standard" or "adam".
    random_state: np.random.RandomState, optional (default None)
        A random number generator instance to use for reproducibility. If None, the global numpy random state is used.

    Returns
    -------
    embedding: array of shape (n_samples, n_components)
        The optimized embedding.
    """

    dim = head_embedding.shape[1]
    if random_state is None:
        random_state = np.random.RandomState()

    if optimizer not in [
        "standard",
        "adam",
        "compatibility",
        "densmap_standard",
        "densmap_adam",
    ]:
        raise ValueError(
            f"Unknown optimizer {optimizer}. Must be one of 'standard', 'adam', 'compatibility', 'densmap_standard' or 'densmap_adam'."
        )

    if optimizer != "compatibility" and (csr_indptr is None or csr_indices is None):
        raise ValueError(
            "When using an optimizer other than 'compatibility', csr_indptr and csr_indices must be provided."
        )

    epochs_per_negative_sample = epochs_per_sample / negative_sample_rate
    epoch_of_next_negative_sample = epochs_per_negative_sample.copy()
    epoch_of_next_sample = epochs_per_sample.copy()
    updates = np.zeros((head_embedding.shape[0], dim), dtype=np.float32)
    node_order = np.arange(head_embedding.shape[0], dtype=np.int32)
    if tail_embedding.shape[0] != head_embedding.shape[0] or optimizer == "adam":
        to_node_order = np.arange(tail_embedding.shape[0], dtype=np.int32)
    else:
        to_node_order = node_order
    block_size = 4096

    epochs_list = None
    embedding_list = []
    if isinstance(n_epochs, list):
        epochs_list = n_epochs
        n_epochs = max(epochs_list)

    # Create learning schedules
    alpha_schedule = _create_alpha_schedule(
        optimizer, n_epochs, initial_alpha, good_initialization
    )
    momentum_schedule = _create_momentum_schedule(
        optimizer, n_epochs, good_initialization
    )

    # Adam-specific schedules
    (
        beta1_schedule,
        beta2_schedule,
        gamma_schedule,
        negative_selection_range_schedule,
    ) = _create_adam_schedules(
        optimizer,
        n_epochs,
        good_initialization,
        gamma,
        n_vertices,
        negative_selection_range,
    )

    # Adjust negative sampling rates for non-compatibility optimizers
    if optimizer != "compatibility":
        epochs_per_negative_sample *= 1.5
        epoch_of_next_negative_sample *= 1.5

    # Initialize optimizer-specific variables
    if optimizer == "compatibility":
        rng_state_per_sample = np.full(
            (head_embedding.shape[0], len(rng_state)), rng_state, dtype=np.int64
        ) + head_embedding[:, 0].astype(np.float64).view(np.int64).reshape(-1, 1)

        # DensMAP setup for compatibility mode
        if densmap_kwds is not None and "mu_sum" in densmap_kwds:
            dens_init_fn = _optimize_layout_euclidean_densmap_epoch_init_coo
            dens_mu_tot = np.sum(densmap_kwds["mu_sum"]) / 2
            dens_lambda = densmap_kwds["lambda"]
            dens_R = densmap_kwds["R"]
            dens_mu = densmap_kwds["mu"]
            dens_phi_sum = np.zeros(n_vertices, dtype=np.float32)
            dens_re_sum = np.zeros(n_vertices, dtype=np.float32)
            dens_var_shift = densmap_kwds["var_shift"]
            densmap = True
        else:
            dens_init_fn = None
            dens_mu_tot = 0
            dens_lambda = 0
            dens_R = np.zeros(1, dtype=np.float32)
            dens_mu = np.zeros(1, dtype=np.float32)
            dens_phi_sum = np.zeros(1, dtype=np.float32)
            dens_re_sum = np.zeros(1, dtype=np.float32)
            dens_var_shift = 0.0
            densmap = False

        optimize_fn = _get_optimize_layout_euclidean_single_epoch_fn(parallel=True)

    elif optimizer.startswith("densmap"):
        # DensMAP setup
        dens_init_fn = _optimize_layout_euclidean_densmap_epoch_init_csr
        dens_mu_tot = np.sum(densmap_kwds["mu_sum"]) / 2
        dens_lambda = densmap_kwds["lambda"]
        dens_R = densmap_kwds["R"]
        dens_mu = densmap_kwds["mu"]
        dens_phi_sum = np.zeros(n_vertices, dtype=np.float32)
        dens_re_sum = np.zeros(n_vertices, dtype=np.float32)
        dens_var_shift = densmap_kwds["var_shift"]
        densmap = True

        if optimizer.endswith("_adam"):
            adam_m = np.zeros_like(updates)
            adam_v = np.zeros_like(updates)

    elif optimizer in ["adam", "standard"]:
        densmap = False
        if optimizer == "adam":
            adam_m = np.zeros_like(updates)
            adam_v = np.zeros_like(updates)

    if densmap_kwds is None:
        densmap_kwds = {}
    if tqdm_kwds is None:
        tqdm_kwds = {}

    if "disable" not in tqdm_kwds:
        tqdm_kwds["disable"] = not verbose

    print(f"Using {negative_selection_range=}")

    for n in tqdm(range(n_epochs), **tqdm_kwds):
        if (
            densmap
            and (densmap_kwds["lambda"] > 0)
            and (((n + 1) / float(n_epochs)) > (1 - densmap_kwds["frac"]))
        ):
            if csr_indptr is not None and csr_indices is not None:
                dens_init_fn(
                    head_embedding,
                    tail_embedding,
                    csr_indptr,
                    csr_indices,
                    a,
                    b,
                    dens_re_sum,
                    dens_phi_sum,
                )
            else:
                dens_init_fn(
                    head_embedding,
                    tail_embedding,
                    head,
                    tail,
                    a,
                    b,
                    dens_re_sum,
                    dens_phi_sum,
                )

            # FIXME: dens_var_shift might be referenced before assignment
            dens_re_std = np.sqrt(np.var(dens_re_sum) + dens_var_shift)
            dens_re_mean = np.mean(dens_re_sum)
            dens_re_cov = np.dot(dens_re_sum, dens_R) / (n_vertices - 1)
            densmap_flag = True
        else:
            dens_re_std = 0
            dens_re_mean = 0
            dens_re_cov = 0
            densmap_flag = False

        if optimizer == "standard":
            optimize_layout_euclidean_single_epoch_fast(
                head_embedding,
                tail_embedding,
                csr_indptr,
                csr_indices,
                n_vertices,
                epochs_per_sample,
                a,
                b,
                gamma,
                dim,
                alpha_schedule[n],
                epochs_per_negative_sample,
                epoch_of_next_negative_sample,
                epoch_of_next_sample,
                n,
                updates,
                node_order,
                to_node_order,
                block_size=block_size,
            )
            updates *= momentum_schedule[n]
            random_state.shuffle(node_order)
            if tail_embedding.shape[0] != head_embedding.shape[0]:
                random_state.shuffle(to_node_order)
        elif optimizer == "adam":
            optimize_layout_euclidean_single_epoch_adam(
                head_embedding,
                tail_embedding,
                csr_indptr,
                csr_indices,
                n_vertices,
                epochs_per_sample,
                a,
                b,
                gamma_schedule[n],
                dim,
                alpha_schedule[n],
                epochs_per_negative_sample,
                epoch_of_next_negative_sample,
                epoch_of_next_sample,
                n,
                updates,
                adam_m,
                adam_v,
                beta1_schedule[n],
                beta2_schedule[n],
                node_order,
                to_node_order,
                block_size=n_vertices,  # block_size,
                negative_selection_range=negative_selection_range_schedule[
                    n
                ],  # negative_selection_range,
            )
            updates[:] = 0.0
            projection_direction = random_state.randn(dim)
            projection_direction /= np.linalg.norm(projection_direction)
            projection = np.dot(head_embedding, projection_direction)
            node_order = np.argsort(projection).astype(np.int32)
            # node_order = adaptive_bucket_sort(
            #     projection, negative_selection_range // 16
            # )

            # norms = (
            #     np.linalg.norm(head_embedding, axis=1)
            #     + random_state.randn(head_embedding.shape[0]) * 1e-1
            # )
            # node_order = np.argsort(norms)[::-1].astype(np.int32, order="C")

            # if n % 10 == 0:
            #     # Occasionally reshuffle the negative sampling order
            #     projection_direction = random_state.randn(dim)
            #     projection_direction /= np.linalg.norm(projection_direction)
            #     projection = np.dot(head_embedding, projection_direction)
            #     to_node_order = np.argsort(projection).astype(np.int32)
            # random_state.shuffle(node_order)
            # if tail_embedding.shape[0] != head_embedding.shape[0]:
            # random_state.shuffle(to_node_order)
            random_state.shuffle(to_node_order)
        elif optimizer == "densmap_standard":
            optimize_layout_euclidean_single_epoch_fast_densmap(
                head_embedding,
                tail_embedding,
                csr_indptr,
                csr_indices,
                n_vertices,
                epochs_per_sample,
                a,
                b,
                gamma,
                dim,
                alpha_schedule[n],
                epochs_per_negative_sample,
                epoch_of_next_negative_sample,
                epoch_of_next_sample,
                n,
                dens_phi_sum,
                dens_re_sum,
                dens_re_cov,
                dens_re_std,
                dens_re_mean,
                dens_lambda,
                dens_R,
                dens_mu,
                dens_mu_tot,
                updates,
                node_order,
                block_size,
                densmap_flag=densmap_flag,
            )
            updates *= momentum_schedule[n]
            random_state.shuffle(node_order)
        elif optimizer == "densmap_adam":
            optimize_layout_euclidean_single_epoch_adam_densmap(
                head_embedding,
                tail_embedding,
                csr_indptr,
                csr_indices,
                n_vertices,
                epochs_per_sample,
                a,
                b,
                gamma_schedule[n],
                dim,
                alpha_schedule[n],
                epochs_per_negative_sample,
                epoch_of_next_negative_sample,
                epoch_of_next_sample,
                n,
                dens_phi_sum,
                dens_re_sum,
                dens_re_cov,
                dens_re_std,
                dens_re_mean,
                dens_lambda,
                dens_R,
                dens_mu,
                dens_mu_tot,
                updates,
                adam_m,
                adam_v,
                beta1_schedule[n],
                beta2_schedule[n],
                node_order,
                block_size=block_size,
                densmap_flag=densmap_flag,
            )
            updates[:] = 0.0
            random_state.shuffle(node_order)
        elif optimizer == "compatibility":
            optimize_fn(
                head_embedding,
                tail_embedding,
                head,
                tail,
                n_vertices,
                epochs_per_sample,
                a,
                b,
                rng_state_per_sample,
                gamma,
                dim,
                move_other,
                alpha_schedule[n],
                epochs_per_negative_sample,
                epoch_of_next_negative_sample,
                epoch_of_next_sample,
                n,
                densmap_flag,
                dens_phi_sum,
                dens_re_sum,
                dens_re_cov,
                dens_re_std,
                dens_re_mean,
                dens_lambda,
                dens_R,
                dens_mu,
                dens_mu_tot,
            )

        if epochs_list is not None and n in epochs_list:
            embedding_list.append(head_embedding.copy())

    # Add the last embedding to the list as well
    if epochs_list is not None:
        embedding_list.append(head_embedding.copy())

    return head_embedding if epochs_list is None else embedding_list


def _optimize_layout_generic_single_epoch(
    epochs_per_sample,
    epoch_of_next_sample,
    head,
    tail,
    head_embedding,
    tail_embedding,
    output_metric,
    output_metric_kwds,
    dim,
    alpha,
    move_other,
    n,
    epoch_of_next_negative_sample,
    epochs_per_negative_sample,
    rng_state_per_sample,
    n_vertices,
    a,
    b,
    gamma,
):
    for i in range(epochs_per_sample.shape[0]):
        if epoch_of_next_sample[i] <= n:
            j = head[i]
            k = tail[i]

            current = head_embedding[j]
            other = tail_embedding[k]

            dist_output, grad_dist_output = output_metric(
                current, other, *output_metric_kwds
            )
            _, rev_grad_dist_output = output_metric(other, current, *output_metric_kwds)

            if dist_output > 0.0:
                w_l = pow((1 + a * pow(dist_output, 2 * b)), -1)
            else:
                w_l = 1.0
            grad_coeff = 2 * b * (w_l - 1) / (dist_output + 1e-6)

            for d in range(dim):
                grad_d = clip(grad_coeff * grad_dist_output[d])

                current[d] += grad_d * alpha
                if move_other:
                    grad_d = clip(grad_coeff * rev_grad_dist_output[d])
                    other[d] += grad_d * alpha

            epoch_of_next_sample[i] += epochs_per_sample[i]

            n_neg_samples = int(
                (n - epoch_of_next_negative_sample[i]) / epochs_per_negative_sample[i]
            )

            for p in range(n_neg_samples):
                k = tau_rand_int(rng_state_per_sample[j]) % n_vertices

                other = tail_embedding[k]

                dist_output, grad_dist_output = output_metric(
                    current, other, *output_metric_kwds
                )

                if dist_output > 0.0:
                    w_l = pow((1 + a * pow(dist_output, 2 * b)), -1)
                elif j == k:
                    continue
                else:
                    w_l = 1.0

                grad_coeff = gamma * 2 * b * w_l / (dist_output + 1e-6)

                for d in range(dim):
                    grad_d = clip(grad_coeff * grad_dist_output[d])
                    current[d] += grad_d * alpha

            epoch_of_next_negative_sample[i] += (
                n_neg_samples * epochs_per_negative_sample[i]
            )
    return epoch_of_next_sample, epoch_of_next_negative_sample


@numba.njit(
    fastmath=True,
    parallel=True,
    locals={
        "grad_d": numba.types.float32,
        "dist_output": numba.types.float32,
        "grad_dist_output": numba.types.float64[::1],
        "grad_coeff": numba.types.float32,
        "w_l": numba.types.float32,
        "from_node": numba.types.intp,
        "to_node": numba.types.intp,
        "raw_index": numba.types.intp,
        "n_neg_samples": numba.types.intp,
        "p": numba.types.intp,
        "block_start": numba.types.intp,
        "block_end": numba.types.intp,
        "node_idx": numba.types.intp,
        "current": numba.types.float32[:],
        "other": numba.types.float32[:],
        "updates": numba.types.float32[:, ::1],
    },
)
def _optimize_layout_generic_single_epoch_fast(
    epochs_per_sample,
    epoch_of_next_sample,
    csr_indptr,
    csr_indices,
    head_embedding,
    tail_embedding,
    output_metric,
    output_metric_kwds,
    dim,
    alpha,
    n,
    epoch_of_next_negative_sample,
    epochs_per_negative_sample,
    n_vertices,
    a,
    b,
    gamma,
    updates,
    node_order,
    block_size=4096,
):
    for block_start in range(0, n_vertices, block_size):
        block_end = min(block_start + block_size, n_vertices)
        for node_idx in numba.prange(block_start, block_end):
            from_node = node_order[node_idx]
            current = head_embedding[from_node]

            for raw_index in range(csr_indptr[from_node], csr_indptr[from_node + 1]):
                if epoch_of_next_sample[raw_index] <= n:
                    to_node = csr_indices[raw_index]
                    other = tail_embedding[to_node]

                    dist_output, grad_dist_output = output_metric(
                        current, other, *output_metric_kwds
                    )
                    # _, rev_grad_dist_output = output_metric(other, current, *output_metric_kwds)

                    if dist_output > 0.0:
                        w_l = pow((1 + a * pow(dist_output, 2 * b)), -1)
                        grad_coeff = 2 * b * (w_l - 1) / (dist_output + 1e-6)

                        for d in range(dim):
                            grad_d = clip(grad_coeff * grad_dist_output[d])
                            updates[from_node, d] += grad_d * alpha
                            # if move_other:
                            #     grad_d = clip(grad_coeff * rev_grad_dist_output[d])
                            #     other[d] += grad_d * alpha

                        epoch_of_next_sample[raw_index] += epochs_per_sample[raw_index]

                    n_neg_samples = int(
                        (n - epoch_of_next_negative_sample[raw_index])
                        / epochs_per_negative_sample[raw_index]
                    )

                    for p in range(n_neg_samples):
                        to_node = node_order[(raw_index * (n + p + 1)) % n_vertices]
                        other = tail_embedding[to_node]

                        dist_output, grad_dist_output = output_metric(
                            current, other, *output_metric_kwds
                        )

                        if dist_output > 0.0:
                            w_l = pow((1 + a * pow(dist_output, 2 * b)), -1)
                            grad_coeff = gamma * 2 * b * w_l / (dist_output + 1e-6)

                            for d in range(dim):
                                grad_d = clip(grad_coeff * grad_dist_output[d])
                                updates[from_node, d] += grad_d * alpha

                    epoch_of_next_negative_sample[raw_index] += (
                        n_neg_samples * epochs_per_negative_sample[raw_index]
                    )

        for node_idx in numba.prange(block_start, block_end):
            from_node = node_order[node_idx]
            for d in range(dim):
                head_embedding[from_node, d] += updates[from_node, d]

    return epoch_of_next_sample, epoch_of_next_negative_sample


@numba.njit(
    fastmath=True,
    parallel=True,
    locals={
        "grad_d": numba.types.float32,
        "dist_output": numba.types.float32,
        "grad_dist_output": numba.types.float64[::1],
        "grad_coeff": numba.types.float32,
        "w_l": numba.types.float32,
        "from_node": numba.types.intp,
        "to_node": numba.types.intp,
        "raw_index": numba.types.intp,
        "n_neg_samples": numba.types.intp,
        "p": numba.types.intp,
        "block_start": numba.types.intp,
        "block_end": numba.types.intp,
        "node_idx": numba.types.intp,
        "current": numba.types.float32[:],
        "other": numba.types.float32[:],
        "updates": numba.types.float32[:, ::1],
    },
)
def _optimize_layout_generic_single_epoch_adam(
    epochs_per_sample,
    epoch_of_next_sample,
    csr_indptr,
    csr_indices,
    head_embedding,
    tail_embedding,
    output_metric,
    output_metric_kwds,
    dim,
    alpha,
    n,
    epoch_of_next_negative_sample,
    epochs_per_negative_sample,
    n_vertices,
    a,
    b,
    gamma,
    updates,
    adam_m,
    adam_v,
    beta1,
    beta2,
    node_order,
    block_size=4096,
):
    for block_start in range(0, n_vertices, block_size):
        block_end = min(block_start + block_size, n_vertices)
        for node_idx in numba.prange(block_start, block_end):
            from_node = node_order[node_idx]
            current = head_embedding[from_node]

            for raw_index in range(csr_indptr[from_node], csr_indptr[from_node + 1]):
                if epoch_of_next_sample[raw_index] <= n:
                    to_node = csr_indices[raw_index]
                    other = tail_embedding[to_node]

                    dist_output, grad_dist_output = output_metric(
                        current, other, *output_metric_kwds
                    )
                    # _, rev_grad_dist_output = output_metric(other, current, *output_metric_kwds)

                    if dist_output > 0.0:
                        w_l = pow((1 + a * pow(dist_output, 2 * b)), -1)
                        grad_coeff = 2 * b * (w_l - 1) / (dist_output + 1e-6)

                        for d in range(dim):
                            grad_d = clip(grad_coeff * grad_dist_output[d])
                            updates[from_node, d] += grad_d
                            # if move_other:
                            #     grad_d = clip(grad_coeff * rev_grad_dist_output[d])
                            #     other[d] += grad_d * alpha

                        epoch_of_next_sample[raw_index] += epochs_per_sample[raw_index]

                    n_neg_samples = int(
                        (n - epoch_of_next_negative_sample[raw_index])
                        / epochs_per_negative_sample[raw_index]
                    )

                    for p in range(n_neg_samples):
                        to_node = node_order[(raw_index * (n + p + 1)) % n_vertices]
                        other = tail_embedding[to_node]

                        dist_output, grad_dist_output = output_metric(
                            current, other, *output_metric_kwds
                        )

                        if dist_output > 0.0:
                            w_l = pow((1 + a * pow(dist_output, 2 * b)), -1)
                            grad_coeff = gamma * 2 * b * w_l / (dist_output + 1e-6)

                            for d in range(dim):
                                grad_d = clip(grad_coeff * grad_dist_output[d])
                                updates[from_node, d] += grad_d

                    epoch_of_next_negative_sample[raw_index] += (
                        n_neg_samples * epochs_per_negative_sample[raw_index]
                    )

        for node_idx in numba.prange(block_start, block_end):
            from_node = node_order[node_idx]
            for d in range(dim):
                if updates[from_node, d] != 0.0:
                    adam_m[from_node, d] = (
                        beta1 * adam_m[from_node, d]
                        + (1.0 - beta1) * updates[from_node, d]
                    )
                    adam_v[from_node, d] = (
                        beta2 * adam_v[from_node, d]
                        + (1.0 - beta2) * updates[from_node, d] ** 2
                    )
                    m_est = adam_m[from_node, d] / (1.0 - pow(beta1, n))
                    v_est = adam_v[from_node, d] / (1.0 - pow(beta2, n))
                    head_embedding[from_node, d] += (
                        alpha * m_est / (np.sqrt(v_est) + 1e-4)
                    )

    return epoch_of_next_sample, epoch_of_next_negative_sample


def optimize_layout_generic(
    head_embedding,
    tail_embedding,
    head,
    tail,
    n_epochs,
    n_vertices,
    epochs_per_sample,
    a,
    b,
    rng_state,
    gamma=1.0,
    initial_alpha=1.0,
    negative_sample_rate=5.0,
    output_metric=dist.euclidean,
    output_metric_kwds=(),
    verbose=False,
    tqdm_kwds=None,
    move_other=False,
    optimizer="standard",
    csr_indptr=None,
    csr_indices=None,
    good_initialization=False,
    random_state=None,
):
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

    epochs_per_sample: array of shape (n_1_simplices)
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

    tqdm_kwds: dict (optional, default None)
        Keyword arguments for tqdm progress bar.

    move_other: bool (optional, default False)
        Whether to adjust tail_embedding alongside head_embedding

    optimizer: str (optional, default "standard")
        The optimizer to use for the optimization. Can be one of "standard", "adam",
        or "compatibility".

    csr_indptr: array of int (optional, default None)
        CSR indptr array for the graph of 1-simplices.
        If provided, the optimization will use a faster version
        of the optimization code that does not require the head and tail arrays.

    csr_indices: array of int (optional, default None)
        CSR indices array for the graph of 1-simplices.

    good_initialization: bool (optional, default False)
        Whether the initial embedding is already a good representation of the data.
        If True, the optimization will use different learning rate schedules etc.
        This is only used if optimizer is "standard" or "adam".

    random_state: np.random.RandomState (optional, default None)
        Random state to use for the optimization. If None, a new random state will be created
        using np.random.RandomState.

    output_metric: callable (optional, default dist.euclidean)
        The metric to use for the optimization. Should be a callable that takes two
        arrays of shape (n_components,) and returns a float distance and an array of
        gradients of the distance with respect to the two arrays.

    output_metric_kwds: tuple (optional, default ())
        Additional keyword arguments to pass to the output_metric function.

    Returns
    -------
    embedding: array of shape (n_samples, n_components)
        The optimized embedding.
    """

    dim = head_embedding.shape[1]

    if random_state is None:
        random_state = np.random.RandomState()

    if optimizer not in [
        "standard",
        "adam",
        "compatibility",
    ]:
        raise ValueError(
            f"Unknown optimizer {optimizer}. Must be one of 'standard', 'adam', 'compatibility'."
        )

    if optimizer != "compatibility" and (csr_indptr is None or csr_indices is None):
        raise ValueError(
            "When using an optimizer other than 'compatibility', csr_indptr and csr_indices must be provided."
        )

    epochs_per_negative_sample = epochs_per_sample / negative_sample_rate
    epoch_of_next_negative_sample = epochs_per_negative_sample.copy()
    epoch_of_next_sample = epochs_per_sample.copy()
    updates = np.zeros((head_embedding.shape[0], dim), dtype=np.float32)
    node_order = np.arange(head_embedding.shape[0], dtype=np.int32)
    block_size = 4096

    # Create learning schedules using the shared helper functions
    alpha_schedule = _create_alpha_schedule(
        optimizer, n_epochs, initial_alpha, good_initialization
    )
    momentum_schedule = _create_momentum_schedule(
        optimizer, n_epochs, good_initialization
    )

    # Adam-specific schedules
    beta1_schedule, beta2_schedule, gamma_schedule = _create_adam_schedules(
        optimizer, n_epochs, good_initialization
    )

    # Adjust negative sampling rates for non-compatibility optimizers
    if optimizer != "compatibility":
        epochs_per_negative_sample *= 1.5
        epoch_of_next_negative_sample *= 1.5

    # Initialize optimizer-specific variables
    if optimizer == "compatibility":
        optimize_fn = numba.njit(
            _optimize_layout_generic_single_epoch,
            fastmath=True,
        )

        rng_state_per_sample = np.full(
            (head_embedding.shape[0], len(rng_state)), rng_state, dtype=np.int64
        ) + head_embedding[:, 0].astype(np.float64).view(np.int64).reshape(-1, 1)

    elif optimizer == "adam":
        adam_m = np.zeros_like(updates)
        adam_v = np.zeros_like(updates)

    if tqdm_kwds is None:
        tqdm_kwds = {}

    if "disable" not in tqdm_kwds:
        tqdm_kwds["disable"] = not verbose

    epochs_list = None
    embedding_list = []
    if isinstance(n_epochs, list):
        epochs_list = n_epochs
        n_epochs = max(epochs_list)

    for n in tqdm(range(n_epochs), **tqdm_kwds):
        if optimizer == "compatibility":
            optimize_fn(
                epochs_per_sample,
                epoch_of_next_sample,
                head,
                tail,
                head_embedding,
                tail_embedding,
                output_metric,
                output_metric_kwds,
                dim,
                alpha_schedule[n],
                move_other,
                n,
                epoch_of_next_negative_sample,
                epochs_per_negative_sample,
                rng_state_per_sample,
                n_vertices,
                a,
                b,
                gamma,
            )
        elif optimizer == "standard":
            _optimize_layout_generic_single_epoch_fast(
                epochs_per_sample,
                epoch_of_next_sample,
                csr_indptr,
                csr_indices,
                head_embedding,
                tail_embedding,
                output_metric,
                output_metric_kwds,
                dim,
                alpha_schedule[n],
                n,
                epoch_of_next_negative_sample,
                epochs_per_negative_sample,
                n_vertices,
                a,
                b,
                gamma,
                updates,
                node_order,
                block_size=block_size,
            )
            updates *= momentum_schedule[n]
            random_state.shuffle(node_order)
        elif optimizer == "adam":
            _optimize_layout_generic_single_epoch_adam(
                epochs_per_sample,
                epoch_of_next_sample,
                csr_indptr,
                csr_indices,
                head_embedding,
                tail_embedding,
                output_metric,
                output_metric_kwds,
                dim,
                alpha_schedule[n],
                n,
                epoch_of_next_negative_sample,
                epochs_per_negative_sample,
                n_vertices,
                a,
                b,
                gamma_schedule[n],
                updates,
                adam_m,
                adam_v,
                beta1_schedule[n],
                beta2_schedule[n],
                node_order,
                block_size=block_size,
            )
            updates[:] = 0.0
            random_state.shuffle(node_order)
        else:
            raise ValueError(
                f"Unknown optimizer {optimizer}. Must be one of 'standard', 'adam', or 'compatibility'."
            )

        if epochs_list is not None and n in epochs_list:
            embedding_list.append(head_embedding.copy())

    return head_embedding if epochs_list is None else embedding_list


def _optimize_layout_inverse_single_epoch(
    epochs_per_sample,
    epoch_of_next_sample,
    head,
    tail,
    head_embedding,
    tail_embedding,
    output_metric,
    output_metric_kwds,
    weight,
    sigmas,
    dim,
    alpha,
    move_other,
    n,
    epoch_of_next_negative_sample,
    epochs_per_negative_sample,
    rng_state,
    n_vertices,
    rhos,
    gamma,
):
    for i in range(epochs_per_sample.shape[0]):
        if epoch_of_next_sample[i] <= n:
            j = head[i]
            k = tail[i]

            current = head_embedding[j]
            other = tail_embedding[k]

            dist_output, grad_dist_output = output_metric(
                current, other, *output_metric_kwds
            )

            w_l = weight[i]
            grad_coeff = -(1 / (w_l * sigmas[k] + 1e-6))

            for d in range(dim):
                grad_d = clip(grad_coeff * grad_dist_output[d])

                current[d] += grad_d * alpha
                if move_other:
                    other[d] += -grad_d * alpha

            epoch_of_next_sample[i] += epochs_per_sample[i]

            n_neg_samples = int(
                (n - epoch_of_next_negative_sample[i]) / epochs_per_negative_sample[i]
            )

            for p in range(n_neg_samples):
                k = tau_rand_int(rng_state) % n_vertices

                other = tail_embedding[k]

                dist_output, grad_dist_output = output_metric(
                    current, other, *output_metric_kwds
                )

                # w_l = 0.0 # for negative samples, the edge does not exist
                w_h = np.exp(-max(dist_output - rhos[k], 1e-6) / (sigmas[k] + 1e-6))
                grad_coeff = -gamma * ((0 - w_h) / ((1 - w_h) * sigmas[k] + 1e-6))

                for d in range(dim):
                    grad_d = clip(grad_coeff * grad_dist_output[d])
                    current[d] += grad_d * alpha

            epoch_of_next_negative_sample[i] += (
                n_neg_samples * epochs_per_negative_sample[i]
            )


def optimize_layout_inverse(
    head_embedding,
    tail_embedding,
    head,
    tail,
    weight,
    sigmas,
    rhos,
    n_epochs,
    n_vertices,
    epochs_per_sample,
    a,
    b,
    rng_state,
    gamma=1.0,
    initial_alpha=1.0,
    negative_sample_rate=5.0,
    output_metric=dist.euclidean,
    output_metric_kwds=(),
    verbose=False,
    tqdm_kwds=None,
    move_other=False,
):
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

    weight: array of shape (n_1_simplices)
        The membership weights of the 1-simplices.

    sigmas:

    rhos:

    n_epochs: int
        The number of training epochs to use in optimization.

    n_vertices: int
        The number of vertices (0-simplices) in the dataset.

    epochs_per_sample: array of shape (n_1_simplices)
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

    tqdm_kwds: dict (optional, default None)
        Keyword arguments for tqdm progress bar.

    move_other: bool (optional, default False)
        Whether to adjust tail_embedding alongside head_embedding

    Returns
    -------
    embedding: array of shape (n_samples, n_components)
        The optimized embedding.
    """

    dim = head_embedding.shape[1]
    alpha = initial_alpha

    epochs_per_negative_sample = epochs_per_sample / negative_sample_rate
    epoch_of_next_negative_sample = epochs_per_negative_sample.copy()
    epoch_of_next_sample = epochs_per_sample.copy()

    optimize_fn = numba.njit(
        _optimize_layout_inverse_single_epoch,
        fastmath=True,
    )

    if tqdm_kwds is None:
        tqdm_kwds = {}

    if "disable" not in tqdm_kwds:
        tqdm_kwds["disable"] = not verbose

    for n in tqdm(range(n_epochs), **tqdm_kwds):
        optimize_fn(
            epochs_per_sample,
            epoch_of_next_sample,
            head,
            tail,
            head_embedding,
            tail_embedding,
            output_metric,
            output_metric_kwds,
            weight,
            sigmas,
            dim,
            alpha,
            move_other,
            n,
            epoch_of_next_negative_sample,
            epochs_per_negative_sample,
            rng_state,
            n_vertices,
            rhos,
            gamma,
        )
        alpha = initial_alpha * (1.0 - (float(n) / float(n_epochs)))

    return head_embedding


def _optimize_layout_aligned_euclidean_single_epoch(
    head_embeddings,
    tail_embeddings,
    heads,
    tails,
    epochs_per_sample,
    a,
    b,
    regularisation_weights,
    relations,
    rng_state,
    gamma,
    lambda_,
    dim,
    move_other,
    alpha,
    epochs_per_negative_sample,
    epoch_of_next_negative_sample,
    epoch_of_next_sample,
    n,
):
    n_embeddings = len(heads)
    window_size = (relations.shape[1] - 1) // 2

    max_n_edges = 0
    for e_p_s in epochs_per_sample:
        if e_p_s.shape[0] >= max_n_edges:
            max_n_edges = e_p_s.shape[0]

    embedding_order = np.arange(n_embeddings).astype(np.int32)
    np.random.seed(abs(rng_state[0]))
    np.random.shuffle(embedding_order)

    for i in range(max_n_edges):
        for m in embedding_order:
            if i < epoch_of_next_sample[m].shape[0] and epoch_of_next_sample[m][i] <= n:
                j = heads[m][i]
                k = tails[m][i]

                current = head_embeddings[m][j]
                other = tail_embeddings[m][k]

                dist_squared = rdist(current, other)

                if dist_squared > 0.0:
                    grad_coeff = -2.0 * a * b * pow(dist_squared, b - 1.0)
                    grad_coeff /= a * pow(dist_squared, b) + 1.0
                else:
                    grad_coeff = 0.0

                for d in range(dim):
                    grad_d = clip(grad_coeff * (current[d] - other[d]))

                    for offset in range(-window_size, window_size):
                        neighbor_m = m + offset
                        if n_embeddings > neighbor_m >= 0 != offset:
                            identified_index = relations[m, offset + window_size, j]
                            if identified_index >= 0:
                                grad_d -= clip(
                                    (lambda_ * np.exp(-(np.abs(offset) - 1)))
                                    * regularisation_weights[m, offset + window_size, j]
                                    * (
                                        current[d]
                                        - head_embeddings[neighbor_m][
                                            identified_index, d
                                        ]
                                    )
                                )

                    current[d] += clip(grad_d) * alpha
                    if move_other:
                        other_grad_d = clip(grad_coeff * (other[d] - current[d]))

                        for offset in range(-window_size, window_size):
                            neighbor_m = m + offset
                            if n_embeddings > neighbor_m >= 0 != offset:
                                identified_index = relations[m, offset + window_size, k]
                                if identified_index >= 0:
                                    other_grad_d -= clip(
                                        (lambda_ * np.exp(-(np.abs(offset) - 1)))
                                        * regularisation_weights[
                                            m, offset + window_size, k
                                        ]
                                        * (
                                            other[d]
                                            - head_embeddings[neighbor_m][
                                                identified_index, d
                                            ]
                                        )
                                    )

                        other[d] += clip(other_grad_d) * alpha

                epoch_of_next_sample[m][i] += epochs_per_sample[m][i]

                if epochs_per_negative_sample[m][i] > 0:
                    n_neg_samples = int(
                        (n - epoch_of_next_negative_sample[m][i])
                        / epochs_per_negative_sample[m][i]
                    )
                else:
                    n_neg_samples = 0

                for p in range(n_neg_samples):
                    k = tau_rand_int(rng_state) % tail_embeddings[m].shape[0]

                    other = tail_embeddings[m][k]

                    dist_squared = rdist(current, other)

                    if dist_squared > 0.0:
                        grad_coeff = 2.0 * gamma * b
                        grad_coeff /= (0.001 + dist_squared) * (
                            a * pow(dist_squared, b) + 1
                        )
                    elif j == k:
                        continue
                    else:
                        grad_coeff = 0.0

                    for d in range(dim):
                        if grad_coeff > 0.0:
                            grad_d = clip(grad_coeff * (current[d] - other[d]))
                        else:
                            grad_d = 0.0

                        for offset in range(-window_size, window_size):
                            neighbor_m = m + offset
                            if n_embeddings > neighbor_m >= 0 != offset:
                                identified_index = relations[m, offset + window_size, j]
                                if identified_index >= 0:
                                    grad_d -= clip(
                                        (lambda_ * np.exp(-(np.abs(offset) - 1)))
                                        * regularisation_weights[
                                            m, offset + window_size, j
                                        ]
                                        * (
                                            current[d]
                                            - head_embeddings[neighbor_m][
                                                identified_index, d
                                            ]
                                        )
                                    )

                        current[d] += clip(grad_d) * alpha

                epoch_of_next_negative_sample[m][i] += (
                    n_neg_samples * epochs_per_negative_sample[m][i]
                )


@numba.njit(
    fastmath=True,
    parallel=True,
    locals={
        "from_node": numba.types.intp,
        "to_node": numba.types.intp,
        "raw_index": numba.types.intp,
        "dist_squared": numba.types.float32,
        "grad_coeff": numba.types.float32,
        "grad_d": numba.types.float32,
        "other_grad_d": numba.types.float32,
        "offset": numba.types.intp,
        "neighbor_m": numba.types.intp,
        "identified_index": numba.types.intp,
        "m": numba.types.intp,
        "i": numba.types.intp,
        "n_neg_samples": numba.types.intp,
        "p": numba.types.intp,
        "d": numba.types.intp,
        "embedding_idx": numba.types.intp,
    },
)
def optimize_layout_aligned_euclidean_single_epoch_fast(
    head_embeddings,
    tail_embeddings,
    csr_indptrs,
    csr_indices,
    epochs_per_sample,
    a,
    b,
    regularisation_weights,
    relations,
    rng_state,
    gamma,
    lambda_,
    dim,
    alpha,
    epochs_per_negative_sample,
    epoch_of_next_negative_sample,
    epoch_of_next_sample,
    n,
    updates,
    node_orders,
    block_size=256,
):
    n_embeddings = len(head_embeddings)
    window_size = (relations.shape[1] - 1) // 2

    max_n_edges = 0
    for e_p_s in epochs_per_sample:
        if e_p_s.shape[0] >= max_n_edges:
            max_n_edges = e_p_s.shape[0]

    embedding_order = np.arange(n_embeddings).astype(np.int32)
    np.random.seed(abs(rng_state[0]))
    np.random.shuffle(embedding_order)

    # Process edges for each embedding
    for m in embedding_order:
        n_vertices = head_embeddings[m].shape[0]
        for block_start in range(0, n_vertices, block_size):
            block_end = min(block_start + block_size, n_vertices)

            for node_idx in numba.prange(block_start, block_end):
                from_node = node_orders[m][node_idx]
                current = head_embeddings[m][from_node]

                for raw_index in range(
                    csr_indptrs[m][from_node], csr_indptrs[m][from_node + 1]
                ):
                    if (
                        raw_index < epoch_of_next_sample[m].shape[0]
                        and epoch_of_next_sample[m][raw_index] <= n
                    ):
                        to_node = csr_indices[m][raw_index]
                        other = tail_embeddings[m][to_node]

                        dist_squared = rdist(current, other)

                        if dist_squared > 0.0:
                            grad_coeff = -2.0 * a * b * pow(dist_squared, b - 1.0)
                            grad_coeff /= a * pow(dist_squared, b) + 1.0

                            for d in range(dim):
                                grad_d = grad_coeff * (current[d] - other[d])

                                for offset in range(-window_size, window_size):
                                    neighbor_m = m + offset
                                    if n_embeddings > neighbor_m >= 0 != offset:
                                        identified_index = relations[
                                            m, offset + window_size, j
                                        ]
                                        if identified_index >= 0:
                                            grad_d -= (
                                                (
                                                    lambda_
                                                    * np.exp(-(np.abs(offset) - 1))
                                                )
                                                * regularisation_weights[
                                                    m, offset + window_size, j
                                                ]
                                                * (
                                                    current[d]
                                                    - head_embeddings[neighbor_m][
                                                        identified_index, d
                                                    ]
                                                )
                                            )

                                updates[m][from_node, d] += grad_d * alpha

                        epoch_of_next_sample[m][node_idx] += epochs_per_sample[m][
                            node_idx
                        ]

                        if epochs_per_negative_sample[m][node_idx] > 0:
                            n_neg_samples = int(
                                (n - epoch_of_next_negative_sample[m][node_idx])
                                / epochs_per_negative_sample[m][node_idx]
                            )
                        else:
                            n_neg_samples = 0

                        for p in range(n_neg_samples):
                            to_node = node_orders[m][
                                (raw_index * (n + p + 1)) % n_vertices
                            ]
                            other = tail_embeddings[m][to_node]
                            dist_squared = rdist(current, other)

                            if dist_squared > 0.0:
                                grad_coeff = 2.0 * gamma * b
                                grad_coeff /= (0.001 + dist_squared) * (
                                    a * pow(dist_squared, b) + 1
                                )

                                for d in range(dim):
                                    if grad_coeff > 0.0:
                                        grad_d = grad_coeff * (current[d] - other[d])
                                    else:
                                        grad_d = 0.0

                                    for offset in range(-window_size, window_size):
                                        neighbor_m = m + offset
                                        if n_embeddings > neighbor_m >= 0 != offset:
                                            identified_index = relations[
                                                m, offset + window_size, j
                                            ]
                                            if identified_index >= 0:
                                                grad_d -= (
                                                    (
                                                        lambda_
                                                        * np.exp(-(np.abs(offset) - 1))
                                                    )
                                                    * regularisation_weights[
                                                        m, offset + window_size, j
                                                    ]
                                                    * (
                                                        current[d]
                                                        - head_embeddings[neighbor_m][
                                                            identified_index, d
                                                        ]
                                                    )
                                                )

                                    updates[m][from_node, d] += grad_d * alpha

                        epoch_of_next_negative_sample[m][node_idx] += (
                            n_neg_samples * epochs_per_negative_sample[m][node_idx]
                        )

            # Apply updates
            for node_idx in numba.prange(block_start, block_end):
                from_node = node_orders[m][node_idx]
                for d in range(dim):
                    head_embeddings[m][from_node, d] += updates[m][from_node, d]

    return epoch_of_next_sample, epoch_of_next_negative_sample


@numba.njit(
    fastmath=True,
    parallel=True,
    locals={
        "from_node": numba.types.intp,
        "to_node": numba.types.intp,
        "raw_index": numba.types.intp,
        "dist_squared": numba.types.float32,
        "grad_coeff": numba.types.float32,
        "grad_d": numba.types.float32,
        "other_grad_d": numba.types.float32,
        "offset": numba.types.intp,
        "neighbor_m": numba.types.intp,
        "identified_index": numba.types.intp,
        "m": numba.types.intp,
        "i": numba.types.intp,
        "n_neg_samples": numba.types.intp,
        "p": numba.types.intp,
        "d": numba.types.intp,
        "embedding_idx": numba.types.intp,
        "m_est": numba.types.float32,
        "v_est": numba.types.float32,
    },
)
def optimize_layout_aligned_euclidean_single_epoch_adam(
    head_embeddings,
    tail_embeddings,
    heads,
    tails,
    epochs_per_sample,
    a,
    b,
    regularisation_weights,
    relations,
    rng_state,
    gamma,
    lambda_,
    dim,
    alpha,
    epochs_per_negative_sample,
    epoch_of_next_negative_sample,
    epoch_of_next_sample,
    n,
    updates,
    adam_m,
    adam_v,
    beta1,
    beta2,
    node_orders,
    block_size=256,
):
    n_embeddings = len(heads)
    window_size = (relations.shape[1] - 1) // 2

    max_n_edges = 0
    for e_p_s in epochs_per_sample:
        if e_p_s.shape[0] >= max_n_edges:
            max_n_edges = e_p_s.shape[0]

    embedding_order = np.arange(n_embeddings).astype(np.int32)
    np.random.seed(abs(rng_state[0]))
    np.random.shuffle(embedding_order)

    # Process edges for each embedding
    for m in embedding_order:
        n_vertices = head_embeddings[m].shape[0]
        for block_start in range(0, max_n_edges, block_size):
            block_end = min(block_start + block_size, max_n_edges)

            for i in numba.prange(block_start, block_end):
                if (
                    i < epoch_of_next_sample[m].shape[0]
                    and epoch_of_next_sample[m][i] <= n
                ):
                    j = heads[m][i]
                    k = tails[m][i]

                    current = head_embeddings[m][j]
                    other = tail_embeddings[m][k]

                    dist_squared = (
                        rdist(current, other) / 2
                    )  # Adam uses scaled distance

                    if dist_squared > 0.0:
                        grad_coeff = -2.0 * a * b * pow(dist_squared, b - 1.0)
                        grad_coeff /= a * pow(dist_squared, b) + 1.0
                    else:
                        grad_coeff = 0.0

                    for d in range(dim):
                        grad_d = grad_coeff * (current[d] - other[d])

                        for offset in range(-window_size, window_size):
                            neighbor_m = m + offset
                            if n_embeddings > neighbor_m >= 0 != offset:
                                identified_index = relations[m, offset + window_size, j]
                                if identified_index >= 0:
                                    grad_d -= (
                                        (lambda_ * np.exp(-(np.abs(offset) - 1)))
                                        * regularisation_weights[
                                            m, offset + window_size, j
                                        ]
                                        * (
                                            current[d]
                                            - head_embeddings[neighbor_m][
                                                identified_index, d
                                            ]
                                        )
                                    )

                        updates[m][j, d] += grad_d

                        if True:  # move_other equivalent - always true for adam version
                            other_grad_d = grad_coeff * (other[d] - current[d])

                            for offset in range(-window_size, window_size):
                                neighbor_m = m + offset
                                if n_embeddings > neighbor_m >= 0 != offset:
                                    identified_index = relations[
                                        m, offset + window_size, k
                                    ]
                                    if identified_index >= 0:
                                        other_grad_d -= (
                                            (lambda_ * np.exp(-(np.abs(offset) - 1)))
                                            * regularisation_weights[
                                                m, offset + window_size, k
                                            ]
                                            * (
                                                other[d]
                                                - head_embeddings[neighbor_m][
                                                    identified_index, d
                                                ]
                                            )
                                        )

                            updates[m][k, d] += other_grad_d

                    epoch_of_next_sample[m][i] += epochs_per_sample[m][i]

                    if epochs_per_negative_sample[m][i] > 0:
                        n_neg_samples = int(
                            (n - epoch_of_next_negative_sample[m][i])
                            / epochs_per_negative_sample[m][i]
                        )
                    else:
                        n_neg_samples = 0

                    for p in range(n_neg_samples):
                        k = tau_rand_int(rng_state) % tail_embeddings[m].shape[0]
                        other = tail_embeddings[m][k]
                        dist_squared = (
                            rdist(current, other) / 4
                        )  # Adam uses scaled distance for negative samples

                        if dist_squared > 0.0:
                            grad_coeff = 2.0 * gamma * b
                            grad_coeff /= (0.001 + dist_squared) * (
                                a * pow(dist_squared, b) + 1
                            )
                        elif j == k:
                            continue
                        else:
                            grad_coeff = 0.0

                        for d in range(dim):
                            if grad_coeff > 0.0:
                                grad_d = grad_coeff * (current[d] - other[d])
                            else:
                                grad_d = 0.0

                            for offset in range(-window_size, window_size):
                                neighbor_m = m + offset
                                if n_embeddings > neighbor_m >= 0 != offset:
                                    identified_index = relations[
                                        m, offset + window_size, j
                                    ]
                                    if identified_index >= 0:
                                        grad_d -= (
                                            (lambda_ * np.exp(-(np.abs(offset) - 1)))
                                            * regularisation_weights[
                                                m, offset + window_size, j
                                            ]
                                            * (
                                                current[d]
                                                - head_embeddings[neighbor_m][
                                                    identified_index, d
                                                ]
                                            )
                                        )

                            updates[m][j, d] += clip(grad_d)

                    epoch_of_next_negative_sample[m][i] += (
                        n_neg_samples * epochs_per_negative_sample[m][i]
                    )

    # Apply Adam updates
    for m in range(n_embeddings):
        for j in numba.prange(head_embeddings[m].shape[0]):
            for d in range(dim):
                if updates[m][j, d] != 0.0:
                    adam_m[m][j, d] = (
                        beta1 * adam_m[m][j, d] + (1.0 - beta1) * updates[m][j, d]
                    )
                    adam_v[m][j, d] = (
                        beta2 * adam_v[m][j, d] + (1.0 - beta2) * updates[m][j, d] ** 2
                    )
                    m_est = adam_m[m][j, d] / (1.0 - pow(beta1, n))
                    v_est = adam_v[m][j, d] / (1.0 - pow(beta2, n))
                    head_embeddings[m][j, d] += alpha * m_est / (np.sqrt(v_est) + 1e-4)

                updates[m][j, d] = 0.0  # Reset for next iteration


def optimize_layout_aligned_euclidean(
    head_embeddings,
    tail_embeddings,
    heads,
    tails,
    n_epochs,
    epochs_per_sample,
    regularisation_weights,
    relations,
    rng_state,
    a=1.576943460405378,
    b=0.8950608781227859,
    gamma=1.0,
    lambda_=5e-3,
    initial_alpha=1.0,
    negative_sample_rate=5.0,
    parallel=True,
    verbose=False,
    tqdm_kwds=None,
    move_other=False,
):
    dim = head_embeddings[0].shape[1]
    alpha = initial_alpha

    epochs_per_negative_sample = numba.typed.List.empty_list(numba.types.float32[::1])
    epoch_of_next_negative_sample = numba.typed.List.empty_list(
        numba.types.float32[::1]
    )
    epoch_of_next_sample = numba.typed.List.empty_list(numba.types.float32[::1])

    for m in range(len(heads)):
        epochs_per_negative_sample.append(
            epochs_per_sample[m].astype(np.float32) / negative_sample_rate
        )
        epoch_of_next_negative_sample.append(
            epochs_per_negative_sample[m].astype(np.float32)
        )
        epoch_of_next_sample.append(epochs_per_sample[m].astype(np.float32))

    optimize_fn = numba.njit(
        _optimize_layout_aligned_euclidean_single_epoch,
        fastmath=True,
        parallel=parallel,
    )

    if tqdm_kwds is None:
        tqdm_kwds = {}

    if "disable" not in tqdm_kwds:
        tqdm_kwds["disable"] = not verbose

    for n in tqdm(range(n_epochs), **tqdm_kwds):
        optimize_fn(
            head_embeddings,
            tail_embeddings,
            heads,
            tails,
            epochs_per_sample,
            a,
            b,
            regularisation_weights,
            relations,
            rng_state,
            gamma,
            lambda_,
            dim,
            move_other,
            alpha,
            epochs_per_negative_sample,
            epoch_of_next_negative_sample,
            epoch_of_next_sample,
            n,
        )

        alpha = initial_alpha * (1.0 - (float(n) / float(n_epochs)))

    return head_embeddings
