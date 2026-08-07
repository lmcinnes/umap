import numba
import numpy as np
from tqdm.auto import tqdm

import umap.distances as dist
from umap.utils import adaptive_bucket_sort, tau_rand_int

PUBLIC_OPTIMIZERS = ("adam", "momentum", "compatibility")
GENERIC_NEGATIVE_SEARCH_MAX_CANDIDATES = 4

# Small kernels accumulate all active incident-edge forces for a source before
# applying one optimizer update. Empirical Iris/Digits force-balance sweeps show
# that half-strength positive forces avoid contraction under this accumulation
# policy while retaining neighborhood quality.
SMALL_LAYOUT_ATTRACTION_SCALE = 0.5


def validate_optimizer(optimizer):
    """Validate an optimizer name and return it unchanged."""
    if optimizer not in PUBLIC_OPTIMIZERS:
        choices = ", ".join(repr(value) for value in PUBLIC_OPTIMIZERS)
        raise ValueError(f"Unknown optimizer {optimizer!r}. Must be one of {choices}.")
    return optimizer


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
    "void(f4[:, ::1], f4[:, ::1], i4[::1], i4[::1], i8, f8[::1], f8, f8, f8, i8, f8, f8[::1], f8[::1], f8[::1], i8, f4[:, ::1], i4[::1], i4[::1], i4[::1], i8, i8, f8, b1)",
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
    source_ranks,
    block_size=4096,
    negative_selection_range=200_000,
    negative_sample_scale=-1.0,
    exclude_graph_neighbors=False,
):
    n_from_vertices = csr_indptr.shape[0] - 1
    transform_mode = n_from_vertices != n_vertices
    negative_selection_range = max(200, min(n_vertices, negative_selection_range))
    if negative_sample_scale < 0.0:
        negative_sample_scaling = negative_selection_range / n_vertices
    else:
        negative_sample_scaling = negative_sample_scale
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
                        to_node_raw_selection = (
                            raw_index * (n + p + 1)
                        ) % negative_selection_range
                        range_start = (
                            source_ranks[from_node] - negative_selection_range // 2
                        )
                        if range_start < 0:
                            range_start = 0
                        elif range_start + negative_selection_range > n_vertices:
                            range_start = n_vertices - negative_selection_range
                        to_node = to_node_order[range_start + to_node_raw_selection]
                        if exclude_graph_neighbors:
                            if not transform_mode and to_node == from_node:
                                continue
                            edge_start = csr_indptr[from_node]
                            edge_end = csr_indptr[from_node + 1]
                            edge_position = np.searchsorted(
                                csr_indices[edge_start:edge_end], to_node
                            )
                            if (
                                edge_position < edge_end - edge_start
                                and csr_indices[edge_start + edge_position] == to_node
                            ):
                                continue
                        other = tail_embedding[to_node]

                        dist_squared = rdist(current, other)

                        if dist_squared > 0.0:
                            grad_coeff = negative_sample_scaling * 2.0 * gamma * b
                            grad_coeff /= (0.001 + dist_squared) * (
                                a * pow(dist_squared, b) + 1
                            )

                            if grad_coeff > 0.0:
                                grad_norm = np.sqrt(
                                    grad_coeff * grad_coeff * dist_squared
                                )
                                scale = gamma * np.tanh(grad_norm / gamma) / grad_norm
                                for d in range(dim):
                                    updates[from_node, d] += (
                                        alpha
                                        * grad_coeff
                                        * (current[d] - other[d])
                                        * scale
                                    )

                    epoch_of_next_negative_sample[raw_index] += (
                        n_neg_samples * epochs_per_negative_sample[raw_index]
                    )

        for node_idx in numba.prange(block_start, block_end):
            from_node = from_node_order[node_idx]
            for d in range(dim):
                head_embedding[from_node, d] += updates[from_node, d]


@numba.njit(
    "void(f4[:, ::1], f4[:, ::1], i4[::1], i4[::1], f4[::1], i8, f8[::1], f8, f8, f8, i8, f8, f8[::1], f8[::1], f8[::1], i8, f4[:, ::1], i4[::1], i4[::1], b1)",
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
def optimize_small_layout_euclidean_single_epoch_fast(
    head_embedding,
    tail_embedding,
    csr_indptr,
    csr_indices,
    csr_data,
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
    exclude_graph_neighbors=False,
):
    n_from_vertices = csr_indptr.shape[0] - 1
    transform_mode = n_from_vertices != n_vertices
    for node_idx in numba.prange(n_from_vertices):
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
                        updates[from_node, d] += (
                            SMALL_LAYOUT_ATTRACTION_SCALE
                            * alpha
                            * grad_coeff
                            * (current[d] - other[d])
                        )

                epoch_of_next_sample[raw_index] += epochs_per_sample[raw_index]

                n_neg_samples = int(
                    (n - epoch_of_next_negative_sample[raw_index])
                    / epochs_per_negative_sample[raw_index]
                )

                accepted_negatives = 0
                candidate_attempts = 0
                max_candidate_attempts = max(1, n_neg_samples) * n_vertices
                while (
                    accepted_negatives < n_neg_samples
                    and candidate_attempts < max_candidate_attempts
                ):
                    if exclude_graph_neighbors:
                        candidate_rank = (
                            raw_index * (n + 1) + candidate_attempts
                        ) % n_vertices
                    else:
                        candidate_rank = (
                            raw_index * (n + accepted_negatives + 1)
                        ) % n_vertices
                    candidate_attempts += 1
                    to_node = to_node_order[candidate_rank]
                    if exclude_graph_neighbors:
                        if not transform_mode and to_node == from_node:
                            continue
                        edge_start = csr_indptr[from_node]
                        edge_end = csr_indptr[from_node + 1]
                        edge_position = np.searchsorted(
                            csr_indices[edge_start:edge_end], to_node
                        )
                        if (
                            edge_position < edge_end - edge_start
                            and csr_indices[edge_start + edge_position] == to_node
                        ):
                            continue
                    accepted_negatives += 1
                    other = tail_embedding[to_node]

                    dist_squared = rdist(current, other)

                    if dist_squared > 0.0:
                        grad_coeff = 2.0 * gamma * b
                        grad_coeff /= (0.001 + dist_squared) * (
                            a * pow(dist_squared, b) + 1
                        )

                        if grad_coeff > 0.0:
                            grad_norm = np.sqrt(grad_coeff * grad_coeff * dist_squared)
                            scale = gamma * np.tanh(grad_norm / gamma) / grad_norm
                            for d in range(dim):
                                updates[from_node, d] += (
                                    alpha * grad_coeff * (current[d] - other[d]) * scale
                                )

                epoch_of_next_negative_sample[raw_index] += (
                    n_neg_samples * epochs_per_negative_sample[raw_index]
                )

    for node_idx in numba.prange(n_from_vertices):
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


def _sample_negative_force_ratio(
    embedding,
    to_node_order,
    csr_indptr,
    csr_indices,
    selection_range,
    a,
    b,
    gamma,
    random_state,
    n_sources,
    negatives_per_source,
    exclude_graph_neighbors,
):
    """Robustly estimate global/local unscaled resultant-force ratio."""
    n_vertices = embedding.shape[0]
    n_sources = min(n_sources, n_vertices)
    source_ranks = random_state.randint(0, n_vertices, size=n_sources)
    sources = to_node_order[source_ranks]
    starts = np.clip(
        source_ranks - selection_range // 2, 0, n_vertices - selection_range
    )
    local_ranks = starts[:, None] + random_state.randint(
        0, selection_range, size=(n_sources, negatives_per_source)
    )
    global_ranks = random_state.randint(
        0, n_vertices, size=(n_sources, negatives_per_source)
    )

    def resultants(candidate_ranks):
        candidates = to_node_order[candidate_ranks]
        delta = embedding[sources, None, :] - embedding[candidates]
        distance_squared = np.sum(delta * delta, axis=2)
        coefficient = 2.0 * gamma * b
        coefficient /= (0.001 + distance_squared) * (
            a * np.power(distance_squared, b) + 1.0
        )
        gradient_norm = coefficient * np.sqrt(distance_squared)
        clipping = np.ones_like(gradient_norm)
        nonzero = gradient_norm > 0.0
        clipping[nonzero] = (
            gamma * np.tanh(gradient_norm[nonzero] / gamma) / gradient_norm[nonzero]
        )
        vectors = coefficient[:, :, None] * delta * clipping[:, :, None]
        if exclude_graph_neighbors:
            for source_index, source in enumerate(sources):
                neighbors = csr_indices[csr_indptr[source] : csr_indptr[source + 1]]
                invalid = np.isin(candidates[source_index], neighbors)
                invalid |= candidates[source_index] == source
                vectors[source_index, invalid] = 0.0
        return np.linalg.norm(np.sum(vectors, axis=1), axis=1)

    global_resultants = resultants(global_ranks)
    local_resultants = resultants(local_ranks)
    valid = (global_resultants > 0.0) & (local_resultants > 0.0)
    if not np.any(valid):
        return 1.0
    log_ratios = np.log(global_resultants[valid]) - np.log(local_resultants[valid])
    lower, upper = np.quantile(log_ratios, [0.1, 0.9])
    return float(np.exp(np.mean(np.clip(log_ratios, lower, upper))))


@numba.njit(
    "void(f4[:, ::1], f4[:, ::1], i4[::1], i4[::1], i8, f8[::1], f8, f8, f8, i8, f8, f8[::1], f8[::1], f8[::1], i8, f4[:, ::1], f4[:, ::1], f4[:, ::1], f8, f8, i4[::1], i4[::1], i4[::1], i8, i8, f8, b1)",
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
    source_ranks,
    block_size=256,
    negative_selection_range=200_000,
    negative_sample_scale=-1.0,
    exclude_graph_neighbors=False,
):
    n_from_vertices = csr_indptr.shape[0] - 1
    negative_selection_range = max(200, min(n_vertices, negative_selection_range))
    if negative_sample_scale < 0.0:
        negative_sample_scaling = negative_selection_range / n_vertices
    else:
        negative_sample_scaling = negative_sample_scale
    transform_mode = from_node_order.shape[0] != to_node_order.shape[0]
    for block_start in range(0, n_from_vertices, block_size):
        block_end = min(block_start + block_size, n_from_vertices)
        for raw_idx in numba.prange(block_start, block_end):
            node_idx = from_node_order[raw_idx]
            if transform_mode:
                from_node = node_idx
            else:
                from_node = to_node_order[node_idx]
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
                            source_ranks[from_node],
                            negative_selection_range,
                            n_vertices,
                        )
                        to_node = to_node_order[
                            (range_start + to_node_raw_selection) % n_vertices
                        ]
                        if exclude_graph_neighbors:
                            if not transform_mode and to_node == from_node:
                                continue
                            edge_start = csr_indptr[from_node]
                            edge_end = csr_indptr[from_node + 1]
                            edge_position = np.searchsorted(
                                csr_indices[edge_start:edge_end], to_node
                            )
                            if (
                                edge_position < edge_end - edge_start
                                and csr_indices[edge_start + edge_position] == to_node
                            ):
                                continue
                        other = tail_embedding[to_node]

                        dist_squared = rdist(current, other)

                        if dist_squared > 0.0:
                            grad_coeff = negative_sample_scaling * 2.0 * gamma * b
                            grad_coeff /= (0.001 + dist_squared) * (
                                a * pow(dist_squared, b) + 1
                            )

                            if grad_coeff > 0.0:
                                grad_norm = np.sqrt(
                                    grad_coeff * grad_coeff * dist_squared
                                )
                                scale = gamma * np.tanh(grad_norm / gamma) / grad_norm
                                for d in range(dim):
                                    updates[from_node, d] += (
                                        grad_coeff * (current[d] - other[d]) * scale
                                    )

                    epoch_of_next_negative_sample[raw_index] += (
                        n_neg_samples * epochs_per_negative_sample[raw_index]
                    )

        for raw_idx in numba.prange(block_start, block_end):
            node_idx = from_node_order[raw_idx]
            if transform_mode:
                from_node = node_idx
            else:
                from_node = to_node_order[node_idx]
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
                    m_est = adam_m[from_node, d] / (1.0 - pow(beta1, n + 1))
                    v_est = adam_v[from_node, d] / (1.0 - pow(beta2, n + 1))
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
    "void(f4[:, ::1], f4[:, ::1], i4[::1], i4[::1], f4[::1], i8, f8[::1], f8, f8, f8, i8, f8, f8[::1], f8[::1], f8[::1], i8, f4[:, ::1], f4[:, ::1], f4[:, ::1], f8, f8, i4[::1], i4[::1], b1)",
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
def optimize_small_layout_euclidean_single_epoch_adam(
    head_embedding,
    tail_embedding,
    csr_indptr,
    csr_indices,
    csr_data,
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
    exclude_graph_neighbors=False,
):
    n_from_vertices = csr_indptr.shape[0] - 1
    transform_mode = from_node_order.shape[0] != to_node_order.shape[0]
    for raw_idx in numba.prange(n_from_vertices):
        node_idx = from_node_order[raw_idx]
        if transform_mode:
            from_node = node_idx
        else:
            from_node = to_node_order[node_idx]
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
                        updates[from_node, d] += (
                            SMALL_LAYOUT_ATTRACTION_SCALE
                            * grad_coeff
                            * (current[d] - other[d])
                        )

                epoch_of_next_sample[raw_index] += epochs_per_sample[raw_index]
                n_neg_samples = int(
                    (n - epoch_of_next_negative_sample[raw_index])
                    / epochs_per_negative_sample[raw_index]
                )

                accepted_negatives = 0
                candidate_attempts = 0
                max_candidate_attempts = max(1, n_neg_samples) * n_vertices
                while (
                    accepted_negatives < n_neg_samples
                    and candidate_attempts < max_candidate_attempts
                ):
                    if exclude_graph_neighbors:
                        candidate_rank = (
                            raw_index * (n + 1) + candidate_attempts
                        ) % n_vertices
                    else:
                        candidate_rank = (
                            raw_index * (n + accepted_negatives + 1)
                        ) % n_vertices
                    candidate_attempts += 1
                    to_node = to_node_order[candidate_rank]
                    if exclude_graph_neighbors:
                        if not transform_mode and to_node == from_node:
                            continue
                        edge_start = csr_indptr[from_node]
                        edge_end = csr_indptr[from_node + 1]
                        edge_position = np.searchsorted(
                            csr_indices[edge_start:edge_end], to_node
                        )
                        if (
                            edge_position < edge_end - edge_start
                            and csr_indices[edge_start + edge_position] == to_node
                        ):
                            continue
                    accepted_negatives += 1
                    other = tail_embedding[to_node]
                    dist_squared = rdist(current, other)

                    if dist_squared > 0.0:
                        grad_coeff = 2.0 * gamma * b
                        grad_coeff /= (0.001 + dist_squared) * (
                            a * pow(dist_squared, b) + 1
                        )
                        if grad_coeff > 0.0:
                            grad_norm = np.sqrt(grad_coeff * grad_coeff * dist_squared)
                            scale = gamma * np.tanh(grad_norm / gamma) / grad_norm
                            for d in range(dim):
                                updates[from_node, d] += (
                                    grad_coeff * (current[d] - other[d]) * scale
                                )

                epoch_of_next_negative_sample[raw_index] += (
                    n_neg_samples * epochs_per_negative_sample[raw_index]
                )

    for raw_idx in numba.prange(n_from_vertices):
        node_idx = from_node_order[raw_idx]
        if transform_mode:
            from_node = node_idx
        else:
            from_node = to_node_order[node_idx]
        for d in range(dim):
            if updates[from_node, d] != 0.0:
                adam_m[from_node, d] = (
                    beta1 * adam_m[from_node, d] + (1.0 - beta1) * updates[from_node, d]
                )
                adam_v[from_node, d] = (
                    beta2 * adam_v[from_node, d]
                    + (1.0 - beta2) * updates[from_node, d] ** 2
                )
                m_est = adam_m[from_node, d] / (1.0 - pow(beta1, n + 1))
                v_est = adam_v[from_node, d] / (1.0 - pow(beta2, n + 1))
                head_embedding[from_node, d] += alpha * m_est / (np.sqrt(v_est) + 1e-4)


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

                    # DensMAP's phi and force equations are defined on squared
                    # Euclidean distance; keep this objective independent of
                    # whether Adam or momentum applies the accumulated update.
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
                            updates[from_node, d] += grad_d

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

    elif optimizer == "momentum":
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

    elif optimizer == "adam":
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
    if optimizer == "adam":
        return np.zeros(n_epochs, dtype=np.float32)

    elif optimizer == "momentum":
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


def _finalize_update_buffer(updates, optimizer, momentum=0.0):
    """Apply the optimizer-specific update-buffer lifetime policy."""
    if optimizer == "momentum":
        updates *= momentum
    else:
        updates[:] = 0.0


def _projection_order_and_source_ranks(head_embedding, tail_embedding, random_state):
    """Order reference vertices by projection and rank each source in it."""
    dim = head_embedding.shape[1]
    projection_direction = random_state.randn(dim)
    projection_direction /= np.linalg.norm(projection_direction)
    tail_projection = np.dot(tail_embedding, projection_direction)
    to_node_order = np.argsort(tail_projection).astype(np.int32)

    if head_embedding.shape[0] == tail_embedding.shape[0]:
        source_ranks = np.empty(head_embedding.shape[0], dtype=np.int32)
        source_ranks[to_node_order] = np.arange(tail_embedding.shape[0], dtype=np.int32)
    else:
        sorted_projection = tail_projection[to_node_order]
        source_ranks = np.searchsorted(
            sorted_projection,
            np.dot(head_embedding, projection_direction),
        ).astype(np.int32)
        source_ranks = np.minimum(source_ranks, tail_embedding.shape[0] - 1)

    return to_node_order, source_ranks


def _generic_negative_search_candidates(
    n_vertices,
    negative_selection_range,
    max_candidates=GENERIC_NEGATIVE_SEARCH_MAX_CANDIDATES,
):
    """Map an effective search range to a bounded force-tournament size."""
    effective_range = max(1, min(n_vertices, int(negative_selection_range)))
    return min(max_candidates, max(1, n_vertices // effective_range))


def _create_adam_schedules(
    optimizer,
    n_epochs,
    good_initialization,
    gamma,
    n_vertices,
    negative_selection_range,
):
    """Create beta1, beta2, and gamma schedules for Adam optimizer."""
    if optimizer not in ["adam", "momentum"]:
        return None, None, None, None

    if good_initialization:
        n_warm_up_epochs = int(min(100, n_epochs / 4))
    else:
        n_warm_up_epochs = int(min(n_epochs // 2, 100))  # Use n_epochs/2 but cap at 100

    # beta1_schedule = np.concatenate(
    #     [
    #         [
    #             0.2 + (0.7 * (float(n) / float(n_warm_up_epochs)))
    #             for n in range(n_warm_up_epochs)
    #         ],
    #         np.full(n_epochs - n_warm_up_epochs, 0.9),
    #     ]
    # )
    beta1_schedule = np.full(n_epochs, 0.9, dtype=np.float32)

    # beta2_schedule = np.concatenate(
    #     [
    #         [
    #             0.79 + (0.2 * ((float(n) / float(n_warm_up_epochs))))
    #             for n in range(n_warm_up_epochs)
    #         ],
    #         np.full(n_epochs - n_warm_up_epochs, 0.99),
    #     ]
    # )
    beta2_schedule = np.full(n_epochs, 0.99, dtype=np.float32)

    if good_initialization:
        # gamma_schedule = (
        #     np.concatenate(
        #         [
        #             [
        #                 1.5 * np.sqrt(float(n) / float(n_warm_up_epochs))
        #                 for n in range(n_warm_up_epochs)
        #             ],
        #             [
        #                 0.5
        #                 * (
        #                     1.0
        #                     - (
        #                         float(n - n_warm_up_epochs)
        #                         / float(n_epochs - n_warm_up_epochs)
        #                     )
        #                 )
        #                 + 1.0
        #                 for n in range(n_warm_up_epochs, n_epochs)
        #             ],
        #         ]
        #     )
        #     * gamma
        #     * max(np.sqrt(n_epochs / 100.0), 1.0)
        # )
        # gamma_schedule = np.full(
        #     n_epochs, gamma * max(np.sqrt(n_epochs / 100.0), 1.0), dtype=np.float32
        # )
        gamma_schedule = np.full(n_epochs, gamma, dtype=np.float32)
    else:
        # gamma_schedule = (
        #     np.concatenate(
        #         [
        #             [
        #                 3.0 * np.sqrt(float(n) / float(n_warm_up_epochs))
        #                 for n in range(n_warm_up_epochs)
        #             ],
        #             [
        #                 1.0
        #                 * (
        #                     1.0
        #                     - float(n - n_warm_up_epochs)
        #                     / float(n_epochs - n_warm_up_epochs)
        #                 )
        #                 + 2.0
        #                 for n in range(n_warm_up_epochs, n_epochs)
        #             ],
        #         ]
        #     )
        #     * gamma
        #     * max(np.sqrt(n_epochs / 100.0), 1.0)
        # )
        # gamma_schedule = np.full(
        #     n_epochs, gamma * max(np.sqrt(n_epochs / 100.0), 1.0), dtype=np.float32
        # )
        gamma_schedule = np.full(n_epochs, gamma, dtype=np.float32)

    # negative_selection_range_schedule = np.linspace(
    #     n_vertices,
    #     negative_selection_range,
    #     n_epochs,
    #     dtype=np.int32,
    # )
    negative_selection_range_schedule = np.full(
        n_epochs, negative_selection_range, dtype=np.int32
    )

    return (
        beta1_schedule,
        beta2_schedule,
        gamma_schedule,
        negative_selection_range_schedule,
    )


def _initialize_euclidean_compatibility(
    head_embedding, rng_state, densmap, densmap_kwds, n_vertices
):
    rng_state_per_sample = np.full(
        (head_embedding.shape[0], len(rng_state)), rng_state, dtype=np.int64
    ) + head_embedding[:, 0].astype(np.float64).view(np.int64).reshape(-1, 1)

    if densmap:
        dens_init_fn = _optimize_layout_euclidean_densmap_epoch_init_coo
        dens_mu_tot = np.sum(densmap_kwds["mu_sum"]) / 2
        dens_lambda = densmap_kwds["lambda"]
        dens_R = densmap_kwds["R"]
        dens_mu = densmap_kwds["mu"]
        dens_phi_sum = np.zeros(n_vertices, dtype=np.float32)
        dens_re_sum = np.zeros(n_vertices, dtype=np.float32)
        dens_var_shift = densmap_kwds["var_shift"]
    else:
        dens_init_fn = None
        dens_mu_tot = 0
        dens_lambda = 0
        dens_R = np.zeros(1, dtype=np.float32)
        dens_mu = np.zeros(1, dtype=np.float32)
        dens_phi_sum = np.zeros(1, dtype=np.float32)
        dens_re_sum = np.zeros(1, dtype=np.float32)
        dens_var_shift = 0.0

    return {
        "optimize_fn": _get_optimize_layout_euclidean_single_epoch_fn(parallel=True),
        "rng_state_per_sample": rng_state_per_sample,
        "dens_init_fn": dens_init_fn,
        "dens_mu_tot": dens_mu_tot,
        "dens_lambda": dens_lambda,
        "dens_R": dens_R,
        "dens_mu": dens_mu,
        "dens_phi_sum": dens_phi_sum,
        "dens_re_sum": dens_re_sum,
        "dens_var_shift": dens_var_shift,
    }


def _run_euclidean_compatibility_epoch(
    state,
    head_embedding,
    tail_embedding,
    head,
    tail,
    n_vertices,
    epochs_per_sample,
    a,
    b,
    gamma,
    dim,
    move_other,
    alpha,
    epochs_per_negative_sample,
    epoch_of_next_negative_sample,
    epoch_of_next_sample,
    n,
    densmap_flag,
    dens_re_cov,
    dens_re_std,
    dens_re_mean,
):
    state["optimize_fn"](
        head_embedding,
        tail_embedding,
        head,
        tail,
        n_vertices,
        epochs_per_sample,
        a,
        b,
        state["rng_state_per_sample"],
        gamma,
        dim,
        move_other,
        alpha,
        epochs_per_negative_sample,
        epoch_of_next_negative_sample,
        epoch_of_next_sample,
        n,
        densmap_flag,
        state["dens_phi_sum"],
        state["dens_re_sum"],
        dens_re_cov,
        dens_re_std,
        dens_re_mean,
        state["dens_lambda"],
        state["dens_R"],
        state["dens_mu"],
        state["dens_mu_tot"],
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
    csr_data=None,
    optimizer="adam",
    good_initialization=False,
    random_state=None,
    negative_selection_range=200_000,
    negative_sample_scale=None,
    exclude_graph_neighbors=False,
    negative_sample_scale_adaptation_samples=128,
    densmap=False,
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
    csr_data: array of float (optional, default None)
        CSR data array for the graph of 1-simplices.
    optimizer: str (optional, default "adam")
        The optimizer to use for the optimization. Can be one of "momentum", "adam",
        or "compatibility". DensMAP is selected independently with ``densmap``.
    good_initialization: bool (optional, default False)
        Whether the initial embedding is already a good representation of the data.
        If True, the optimization will use a different learning rate schedules etc.
        This is only used if optimizer is "momentum" or "adam".
    random_state: np.random.RandomState, optional (default None)
        A random number generator instance to use for reproducibility. If None, the global numpy random state is used.
    negative_sample_scale: float, optional (default None)
        Explicit multiplier for repulsive negative-sample gradients in the
        large-input Adam and momentum kernels. If None, use the configured
        range correction.
        This is primarily useful for diagnosing the range correction separately
        from the locality of negative selection.
    exclude_graph_neighbors: bool, optional (default False)
        If True, do not apply negative updates to the source vertex itself or
        to vertices joined to it by an attractive fuzzy-graph edge in the
        large-input kernels. This protects against false negatives in localized
        candidate windows.
    negative_sample_scale_adaptation_samples: int, optional (default 128)
        When hard-negative mining is active (``negative_selection_range <
        n_vertices``), adapt the negative scale every ten epochs through the
        first half of optimization using a sampled, anchored, bounded
        global/local resultant-force ratio. Set to 0 to disable adaptation.
    densmap: bool, optional (default False)
        Whether to include the DensMAP objective terms. This is independent of
        the selected optimizer.
    Returns
    -------
    embedding: array of shape (n_samples, n_components)
        The optimized embedding.
    """

    dim = head_embedding.shape[1]
    if random_state is None:
        random_state = np.random.RandomState()
    resolved_negative_sample_scale = (
        -1.0 if negative_sample_scale is None else float(negative_sample_scale)
    )
    if negative_sample_scale_adaptation_samples > 0:
        if resolved_negative_sample_scale < 0.0:
            resolved_negative_sample_scale = pow(
                min(negative_selection_range, n_vertices) / n_vertices, 0.75
            )
        negative_scale_prior = resolved_negative_sample_scale
        negative_force_ratio_anchor = -1.0
        adaptation_random_state = np.random.RandomState(42)

    validate_optimizer(optimizer)

    if optimizer != "compatibility" and (csr_indptr is None or csr_indices is None):
        raise ValueError(
            "When using an optimizer other than 'compatibility', csr_indptr and csr_indices must be provided."
        )

    epochs_list = None
    embedding_list = []
    if isinstance(n_epochs, list):
        epochs_list = n_epochs
        n_epochs = max(epochs_list)

    epochs_per_negative_sample = epochs_per_sample / negative_sample_rate
    epoch_of_next_negative_sample = epochs_per_negative_sample.copy()
    epoch_of_next_sample = epochs_per_sample.copy()
    updates = np.zeros((head_embedding.shape[0], dim), dtype=np.float32)
    node_order = np.arange(head_embedding.shape[0], dtype=np.int32)
    if tail_embedding.shape[0] != head_embedding.shape[0] or optimizer == "adam":
        to_node_order = np.arange(tail_embedding.shape[0], dtype=np.int32)
    else:
        to_node_order = node_order
    source_ranks = np.arange(head_embedding.shape[0], dtype=np.int32)
    block_size = 4096

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
    b_schedule = np.full(n_epochs, b)  # np.linspace(1.0, b, n_epochs)

    # Adjust negative sampling rates for non-compatibility optimizers
    if optimizer != "compatibility":
        epochs_per_negative_sample *= 1.5
        epoch_of_next_negative_sample *= 1.5

    # Initialize optimizer-specific variables
    if optimizer == "compatibility":
        compatibility_state = _initialize_euclidean_compatibility(
            head_embedding, rng_state, densmap, densmap_kwds, n_vertices
        )
        dens_init_fn = compatibility_state["dens_init_fn"]
        dens_mu_tot = compatibility_state["dens_mu_tot"]
        dens_lambda = compatibility_state["dens_lambda"]
        dens_R = compatibility_state["dens_R"]
        dens_mu = compatibility_state["dens_mu"]
        dens_phi_sum = compatibility_state["dens_phi_sum"]
        dens_re_sum = compatibility_state["dens_re_sum"]
        dens_var_shift = compatibility_state["dens_var_shift"]

    elif densmap:
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

        if optimizer == "adam":
            adam_m = np.zeros_like(updates)
            adam_v = np.zeros_like(updates)

    elif optimizer in ["adam", "momentum"]:
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

    if (
        not densmap
        and head_embedding.shape[0] > 1024
        and optimizer in ("adam", "momentum")
        and negative_selection_range < n_vertices
    ):
        to_node_order, source_ranks = _projection_order_and_source_ranks(
            head_embedding, tail_embedding, random_state
        )

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

        if densmap and optimizer == "momentum":
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
            _finalize_update_buffer(updates, "momentum", momentum_schedule[n])
            random_state.shuffle(node_order)
        elif densmap and optimizer == "adam":
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
            _finalize_update_buffer(updates, "adam")
            random_state.shuffle(node_order)
        elif optimizer == "momentum":
            if head_embedding.shape[0] <= 1024:
                optimize_small_layout_euclidean_single_epoch_fast(
                    head_embedding,
                    tail_embedding,
                    csr_indptr,
                    csr_indices,
                    csr_data,
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
                    exclude_graph_neighbors=exclude_graph_neighbors,
                )
            else:
                optimize_layout_euclidean_single_epoch_fast(
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
                    node_order,
                    to_node_order,
                    source_ranks,
                    block_size=n_vertices // 2,  # block_size,
                    negative_selection_range=negative_selection_range_schedule[n],
                    negative_sample_scale=resolved_negative_sample_scale,
                    exclude_graph_neighbors=exclude_graph_neighbors,
                )
            _finalize_update_buffer(updates, "momentum", momentum_schedule[n])
            random_state.shuffle(node_order)
            if tail_embedding.shape[0] != head_embedding.shape[0]:
                if negative_selection_range_schedule[n] < n_vertices:
                    to_node_order, source_ranks = _projection_order_and_source_ranks(
                        head_embedding, tail_embedding, random_state
                    )
                else:
                    random_state.shuffle(to_node_order)
            elif negative_selection_range_schedule[n] < n_vertices:
                to_node_order, source_ranks = _projection_order_and_source_ranks(
                    head_embedding, tail_embedding, random_state
                )
        elif optimizer == "adam":
            if (
                negative_sample_scale_adaptation_samples > 0
                and head_embedding.shape[0] == tail_embedding.shape[0]
                and negative_selection_range_schedule[n] < n_vertices
                and 10 <= n <= n_epochs // 2
                and n % 10 == 0
            ):
                force_ratio = _sample_negative_force_ratio(
                    head_embedding,
                    to_node_order,
                    csr_indptr,
                    csr_indices,
                    negative_selection_range_schedule[n],
                    a,
                    b_schedule[n],
                    gamma_schedule[n],
                    adaptation_random_state,
                    negative_sample_scale_adaptation_samples,
                    5,
                    exclude_graph_neighbors,
                )
                if negative_force_ratio_anchor < 0.0:
                    negative_force_ratio_anchor = (
                        resolved_negative_sample_scale / force_ratio
                    )
                else:
                    raw_scale = negative_force_ratio_anchor * force_ratio
                    raw_scale = np.clip(
                        raw_scale,
                        0.5 * negative_scale_prior,
                        2.0 * negative_scale_prior,
                    )
                    log_change = np.clip(
                        np.log(raw_scale / resolved_negative_sample_scale),
                        np.log(0.9),
                        np.log(1.1),
                    )
                    resolved_negative_sample_scale *= np.exp(0.1 * log_change)
            if head_embedding.shape[0] <= 1024:
                optimize_small_layout_euclidean_single_epoch_adam(
                    head_embedding,
                    tail_embedding,
                    csr_indptr,
                    csr_indices,
                    csr_data,
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
                    exclude_graph_neighbors=exclude_graph_neighbors,
                )
            else:
                optimize_layout_euclidean_single_epoch_adam(
                    head_embedding,
                    tail_embedding,
                    csr_indptr,
                    csr_indices,
                    n_vertices,
                    epochs_per_sample,
                    a,
                    b_schedule[n],
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
                    source_ranks,
                    block_size=n_vertices,
                    negative_selection_range=negative_selection_range_schedule[n],
                    negative_sample_scale=resolved_negative_sample_scale,
                    exclude_graph_neighbors=exclude_graph_neighbors,
                )
            _finalize_update_buffer(updates, "adam")
            random_state.shuffle(node_order)
            if negative_selection_range_schedule[n] < n_vertices:
                to_node_order, source_ranks = _projection_order_and_source_ranks(
                    head_embedding, tail_embedding, random_state
                )
            else:
                random_state.shuffle(to_node_order)
        elif optimizer == "compatibility":
            _run_euclidean_compatibility_epoch(
                compatibility_state,
                head_embedding,
                tail_embedding,
                head,
                tail,
                n_vertices,
                epochs_per_sample,
                a,
                b,
                gamma,
                dim,
                move_other,
                alpha_schedule[n],
                epochs_per_negative_sample,
                epoch_of_next_negative_sample,
                epoch_of_next_sample,
                n,
                densmap_flag,
                dens_re_cov,
                dens_re_std,
                dens_re_mean,
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
    from_node_order,
    to_node_order,
    negative_search_candidates,
    block_size=4096,
):
    n_from_vertices = csr_indptr.shape[0] - 1
    for block_start in range(0, n_from_vertices, block_size):
        block_end = min(block_start + block_size, n_from_vertices)
        for node_idx in numba.prange(block_start, block_end):
            from_node = from_node_order[node_idx]
            current = head_embedding[from_node]
            best_force = np.empty(dim, dtype=np.float32)

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
                        best_force_squared = -1.0
                        for candidate in range(negative_search_candidates):
                            to_node = to_node_order[
                                (raw_index * (n + p + 1) + candidate) % n_vertices
                            ]
                            other = tail_embedding[to_node]

                            dist_output, grad_dist_output = output_metric(
                                current, other, *output_metric_kwds
                            )

                            force_squared = 0.0
                            if dist_output > 0.0:
                                w_l = pow((1 + a * pow(dist_output, 2 * b)), -1)
                                grad_coeff = gamma * 2 * b * w_l / (dist_output + 1e-6)
                                for d in range(dim):
                                    grad_d = clip(grad_coeff * grad_dist_output[d])
                                    force_squared += grad_d * grad_d

                            if force_squared > best_force_squared:
                                best_force_squared = force_squared
                                for d in range(dim):
                                    if dist_output > 0.0:
                                        best_force[d] = clip(
                                            grad_coeff * grad_dist_output[d]
                                        )
                                    else:
                                        best_force[d] = 0.0

                        for d in range(dim):
                            updates[from_node, d] += best_force[d] * alpha

                    epoch_of_next_negative_sample[raw_index] += (
                        n_neg_samples * epochs_per_negative_sample[raw_index]
                    )

        for node_idx in numba.prange(block_start, block_end):
            from_node = from_node_order[node_idx]
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
    from_node_order,
    to_node_order,
    negative_search_candidates,
    block_size=4096,
):
    n_from_vertices = csr_indptr.shape[0] - 1
    for block_start in range(0, n_from_vertices, block_size):
        block_end = min(block_start + block_size, n_from_vertices)
        for node_idx in numba.prange(block_start, block_end):
            from_node = from_node_order[node_idx]
            current = head_embedding[from_node]
            best_force = np.empty(dim, dtype=np.float32)

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
                        best_force_squared = -1.0
                        for candidate in range(negative_search_candidates):
                            to_node = to_node_order[
                                (raw_index * (n + p + 1) + candidate) % n_vertices
                            ]
                            other = tail_embedding[to_node]

                            dist_output, grad_dist_output = output_metric(
                                current, other, *output_metric_kwds
                            )

                            force_squared = 0.0
                            if dist_output > 0.0:
                                w_l = pow((1 + a * pow(dist_output, 2 * b)), -1)
                                grad_coeff = gamma * 2 * b * w_l / (dist_output + 1e-6)
                                for d in range(dim):
                                    grad_d = clip(grad_coeff * grad_dist_output[d])
                                    force_squared += grad_d * grad_d

                            if force_squared > best_force_squared:
                                best_force_squared = force_squared
                                for d in range(dim):
                                    if dist_output > 0.0:
                                        best_force[d] = clip(
                                            grad_coeff * grad_dist_output[d]
                                        )
                                    else:
                                        best_force[d] = 0.0

                        for d in range(dim):
                            updates[from_node, d] += best_force[d]

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
                    m_est = adam_m[from_node, d] / (1.0 - pow(beta1, n + 1))
                    v_est = adam_v[from_node, d] / (1.0 - pow(beta2, n + 1))
                    head_embedding[from_node, d] += (
                        alpha * m_est / (np.sqrt(v_est) + 1e-4)
                    )

    return epoch_of_next_sample, epoch_of_next_negative_sample


def _initialize_generic_compatibility(head_embedding, rng_state):
    optimize_fn = numba.njit(
        _optimize_layout_generic_single_epoch,
        fastmath=True,
    )
    rng_state_per_sample = np.full(
        (head_embedding.shape[0], len(rng_state)), rng_state, dtype=np.int64
    ) + head_embedding[:, 0].astype(np.float64).view(np.int64).reshape(-1, 1)
    return optimize_fn, rng_state_per_sample


def _run_generic_compatibility_epoch(
    optimize_fn,
    rng_state_per_sample,
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
    n_vertices,
    a,
    b,
    gamma,
):
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
    )


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
    optimizer="adam",
    csr_indptr=None,
    csr_indices=None,
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

    optimizer: str (optional, default "adam")
        The optimizer to use for the optimization. Can be one of "momentum", "adam",
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
        This is only used if optimizer is "momentum" or "adam".

    random_state: np.random.RandomState (optional, default None)
        Random state to use for the optimization. If None, a new random state will be created
        using np.random.RandomState.

    negative_selection_range: int (optional, default 200000)
        Effective reference set size used to derive the number of force-ranked
        candidates for each negative sample. The tournament size is
        ``min(4, max(1, n_vertices // negative_selection_range))``.

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

    validate_optimizer(optimizer)

    if optimizer != "compatibility" and (csr_indptr is None or csr_indices is None):
        raise ValueError(
            "When using an optimizer other than 'compatibility', csr_indptr and csr_indices must be provided."
        )

    epochs_list = None
    embedding_list = []
    if isinstance(n_epochs, list):
        epochs_list = n_epochs
        n_epochs = max(epochs_list)

    epochs_per_negative_sample = epochs_per_sample / negative_sample_rate
    epoch_of_next_negative_sample = epochs_per_negative_sample.copy()
    epoch_of_next_sample = epochs_per_sample.copy()
    updates = np.zeros((head_embedding.shape[0], dim), dtype=np.float32)
    # In transform mode CSR rows index new/source points while negative samples
    # index the existing reference embedding, so the two permutations differ.
    from_node_order = np.arange(head_embedding.shape[0], dtype=np.int32)
    if head_embedding.shape[0] == n_vertices:
        to_node_order = from_node_order
    else:
        to_node_order = np.arange(n_vertices, dtype=np.int32)
    negative_search_candidates = _generic_negative_search_candidates(
        n_vertices, negative_selection_range
    )
    if optimizer != "compatibility":
        random_state.shuffle(from_node_order)
        if to_node_order is not from_node_order:
            random_state.shuffle(to_node_order)
    block_size = 4096

    # Create learning schedules using the shared helper functions
    alpha_schedule = _create_alpha_schedule(
        optimizer, n_epochs, initial_alpha, good_initialization
    )
    momentum_schedule = _create_momentum_schedule(
        optimizer, n_epochs, good_initialization
    )

    # Adam-specific schedules
    beta1_schedule, beta2_schedule, gamma_schedule, _ = _create_adam_schedules(
        optimizer,
        n_epochs,
        good_initialization,
        gamma,
        n_vertices,
        negative_selection_range=n_vertices,
    )

    # Adjust negative sampling rates for non-compatibility optimizers
    if optimizer != "compatibility":
        epochs_per_negative_sample *= 1.5
        epoch_of_next_negative_sample *= 1.5

    # Initialize optimizer-specific variables
    if optimizer == "compatibility":
        optimize_fn, rng_state_per_sample = _initialize_generic_compatibility(
            head_embedding, rng_state
        )

    elif optimizer == "adam":
        adam_m = np.zeros_like(updates)
        adam_v = np.zeros_like(updates)

    if tqdm_kwds is None:
        tqdm_kwds = {}

    if "disable" not in tqdm_kwds:
        tqdm_kwds["disable"] = not verbose

    for n in tqdm(range(n_epochs), **tqdm_kwds):
        if optimizer == "compatibility":
            _run_generic_compatibility_epoch(
                optimize_fn,
                rng_state_per_sample,
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
                n_vertices,
                a,
                b,
                gamma,
            )
        elif optimizer == "momentum":
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
                from_node_order,
                to_node_order,
                negative_search_candidates,
                block_size=block_size,
            )
            _finalize_update_buffer(updates, "momentum", momentum_schedule[n])
            random_state.shuffle(from_node_order)
            if to_node_order is not from_node_order:
                random_state.shuffle(to_node_order)
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
                from_node_order,
                to_node_order,
                negative_search_candidates,
                block_size=block_size,
            )
            _finalize_update_buffer(updates, "adam")
            random_state.shuffle(from_node_order)
            if to_node_order is not from_node_order:
                random_state.shuffle(to_node_order)
        else:
            validate_optimizer(optimizer)

        if epochs_list is not None and n in epochs_list:
            embedding_list.append(head_embedding.copy())

    if epochs_list is not None:
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
