import numba
import numpy as np
from tqdm.auto import tqdm

import umap.distances as dist
from umap.utils import tau_rand_int


@numba.njit(inline='always')
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
    "void(f4[:, ::1], f4[:, ::1], i4[::1], i4[::1], i8, f8[::1], f8, f8, f8, i8, f8, f8[::1], f8[::1], f8[::1], i8, f4[:, ::1], f4[:, ::1], f4[:, ::1], f8, f8, i4[::1], i8)",
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
    adam_m,
    adam_v,
    beta1,
    beta2,
    node_order,
    block_size=256,
):
    # gamma = 3.0
    for block_start in range(0, n_vertices, block_size):
        block_end = min(block_start + block_size, n_vertices)
        for node_idx in numba.prange(block_start, block_end):
            from_node = node_order[node_idx]
            current = head_embedding[from_node]

            for raw_index in range(csr_indptr[from_node], csr_indptr[from_node+1]):
                if epoch_of_next_sample[raw_index] <= n:
                    to_node = csr_indices[raw_index]
                    other = tail_embedding[to_node]

                    dist_squared = rdist(current, other) / 2

                    if dist_squared > 0.0:
                        grad_coeff = -2.0 * a * b * pow(dist_squared, b - 1.0)
                        grad_coeff /= a * pow(dist_squared, b) + 1.0
                        for d in range(dim):
                            grad_d = clip(grad_coeff * (current[d] - other[d]))
                            updates[from_node, d] += grad_d

                    epoch_of_next_sample[raw_index] += epochs_per_sample[raw_index]

                    n_neg_samples = int(
                        (n - epoch_of_next_negative_sample[raw_index]) / epochs_per_negative_sample[raw_index]
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
                    adam_m[from_node, d] = beta1 * adam_m[from_node, d] + (1.0 - beta1) * updates[from_node, d]
                    adam_v[from_node, d] = beta2 * adam_v[from_node, d] + (1.0 - beta2) * updates[from_node, d]**2
                    m_est = adam_m[from_node, d] / (1.0 - pow(beta1, n))
                    v_est = adam_v[from_node, d] / (1.0 - pow(beta2, n))
                    head_embedding[from_node, d] += alpha * m_est / (np.sqrt(v_est) + 1e-4)
            # if from_node == 0 or from_node == 1:
            #     print(adam_m[from_node], updates[from_node])



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
):
    for block_start in range(0, head_embedding.shape[0], block_size):
        block_end = min(block_start + block_size, head_embedding.shape[0])
        for node_idx in numba.prange(block_start, block_end):
            from_node = node_order[node_idx]
            current = head_embedding[from_node]

            for raw_index in range(csr_indptr[from_node], csr_indptr[from_node+1]):
                if epoch_of_next_sample[raw_index] <= n:
                    to_node = csr_indices[raw_index]
                    other = tail_embedding[to_node]

                    dist_squared = rdist(current, other)

                    phi = 1.0 / (1.0 + a * pow(dist_squared, b))
                    dphi_term = (
                        a * b * pow(dist_squared, b - 1) / (1.0 + a * pow(dist_squared, b))
                    )

                    q_jk = phi / dens_phi_sum[to_node]
                    q_kj = phi / dens_phi_sum[from_node]

                    drk = q_jk * (
                        (1.0 - b * (1 - phi)) / np.exp(dens_re_sum[to_node]) + dphi_term
                    )
                    drj = q_kj * (
                        (1.0 - b * (1 - phi)) / np.exp(dens_re_sum[from_node]) + dphi_term
                    )

                    re_std_sq = dens_re_std * dens_re_std
                    weight_k = (
                        dens_R[to_node]
                        - dens_re_cov * (dens_re_sum[to_node] - dens_re_mean) / re_std_sq
                    )
                    weight_j = (
                        dens_R[from_node]
                        - dens_re_cov * (dens_re_sum[from_node] - dens_re_mean) / re_std_sq
                    )

                    grad_cor_coeff = (
                        dens_lambda
                        * dens_mu_tot
                        * (weight_k * drk + weight_j * drj)
                        / (dens_mu[raw_index] * dens_re_std)
                        / n_vertices
                    )

                    if dist_squared > 0.0:
                        grad_coeff = -2.0 * a * b * pow(dist_squared, b - 1.0)
                        grad_coeff /= a * pow(dist_squared, b) + 1.0
                        for d in range(dim):
                            grad_d = clip(grad_coeff * (current[d] - other[d]))
                            grad_d += clip(2 * grad_cor_coeff * (current[d] - other[d]))

                            updates[from_node, d] += grad_d * alpha

                    epoch_of_next_sample[raw_index] += epochs_per_sample[raw_index]

                    n_neg_samples = int(
                        (n - epoch_of_next_negative_sample[raw_index]) / epochs_per_negative_sample[raw_index]
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


def _optimize_layout_euclidean_densmap_epoch_init(
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

    for i in numba.prange(head.size):
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
    for i in range(re_sum.size):
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
    densmap=False,
    densmap_kwds=None,
    tqdm_kwds=None,
    move_other=False,
    csr_indptr=None,
    csr_indices=None,
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
    densmap: bool (optional, default False)
        Whether to use the density-augmented densMAP objective
    densmap_kwds: dict (optional, default None)
        Auxiliary data for densMAP
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
    if random_state is None:
        random_state = np.random.RandomState()

    epochs_per_negative_sample = epochs_per_sample / negative_sample_rate
    epoch_of_next_negative_sample = epochs_per_negative_sample.copy()
    epoch_of_next_sample = epochs_per_sample.copy()

    # Fix for calling UMAP many times for small datasets, otherwise we spend here
    # a lot of time in compilation step (first call to numba function)
    if csr_indptr is not None and csr_indices is not None:
        optimize_fn = optimize_layout_euclidean_single_epoch_fast
        epochs_per_negative_sample *= 2.0 # to account for the fact that we are using a fast version
        epoch_of_next_negative_sample *= 2.0
        initial_alpha = 2.0
        epochs_per_sample /= 1.0
    else:
        optimize_fn = _get_optimize_layout_euclidean_single_epoch_fn(parallel=True)

    if densmap_kwds is None:
        densmap_kwds = {}
    if tqdm_kwds is None:
        tqdm_kwds = {}

    if densmap:
        dens_init_fn = numba.njit(
            _optimize_layout_euclidean_densmap_epoch_init,
            fastmath=True,
            parallel=parallel,
        )

        dens_mu_tot = np.sum(densmap_kwds["mu_sum"]) / 2
        dens_lambda = densmap_kwds["lambda"]
        dens_R = densmap_kwds["R"]
        dens_mu = densmap_kwds["mu"]
        dens_phi_sum = np.zeros(n_vertices, dtype=np.float32)
        dens_re_sum = np.zeros(n_vertices, dtype=np.float32)
        dens_var_shift = densmap_kwds["var_shift"]
    else:
        dens_mu_tot = 0
        dens_lambda = 0
        dens_R = np.zeros(1, dtype=np.float32)
        dens_mu = np.zeros(1, dtype=np.float32)
        dens_phi_sum = np.zeros(1, dtype=np.float32)
        dens_re_sum = np.zeros(1, dtype=np.float32)

    epochs_list = None
    embedding_list = []
    if isinstance(n_epochs, list):
        epochs_list = n_epochs
        n_epochs = max(epochs_list)

    if "disable" not in tqdm_kwds:
        tqdm_kwds["disable"] = not verbose

    n_threads = numba.get_num_threads()
    print(f"Using {n_threads=}")
    updates = np.zeros((head_embedding.shape[0], dim), dtype=np.float32)
    adam_m = np.zeros_like(updates)
    adam_v = np.zeros_like(updates)
    node_order = np.arange(head_embedding.shape[0], dtype=np.int32)
    block_size = head_embedding.shape[0]

    rng_state_per_sample = np.full(
        (head_embedding.shape[0], len(rng_state)), rng_state, dtype=np.int64
    ) + head_embedding[:, 0].astype(np.float64).view(np.int64).reshape(-1, 1)
    rng_state_per_thread = np.full(
        (n_threads, len(rng_state)), rng_state, dtype=np.int64
    ) + np.arange(n_threads).reshape(-1, 1)

    for n in tqdm(range(n_epochs), **tqdm_kwds):
        densmap_flag = (
            densmap
            and (densmap_kwds["lambda"] > 0)
            and (((n + 1) / float(n_epochs)) > (1 - densmap_kwds["frac"]))
        )

        if densmap_flag:
            # FIXME: dens_init_fn might be referenced before assignment

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
        else:
            dens_re_std = 0
            dens_re_mean = 0
            dens_re_cov = 0

        if csr_indptr is not None and csr_indices is not None:
            if n <= 100:
                alpha = (initial_alpha - 0.1) * (1.0 - (float(n) / float(100)))**2 + 0.2
                beta1 = 0.2 + (0.7 * ((float(n) / float(100))))
                beta2 = 0.79 + (0.2 * ((float(n) / float(100))))
                gamma = 1.0 * (float(n) / float(100))
            else:
                alpha = 0.15 * (1.0 - (float(n - 100) / float(n_epochs - 100))) + 0.05
                beta1 = 0.9
                beta2 = 0.99
                gamma = 0.25 * (1.0 - float(n - 100) / float(n_epochs - 100)) + 0.75
            if densmap_flag:
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
                    block_size,
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
                    node_order,
                    block_size=block_size,
                )
            block_size = 4096
            momentum = 0.0# (1.0 - alpha) * 0.5
            updates *= momentum
            random_state.shuffle(node_order)  # Shuffle the order of nodes for the next epoch
        else:
            # use the old version of the function for compatibility
            alpha = initial_alpha * (1.0 - (float(n) / float(n_epochs)))
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
        _optimize_layout_generic_single_epoch,
        fastmath=True,
    )

    if tqdm_kwds is None:
        tqdm_kwds = {}

    if "disable" not in tqdm_kwds:
        tqdm_kwds["disable"] = not verbose

    rng_state_per_sample = np.full(
        (head_embedding.shape[0], len(rng_state)), rng_state, dtype=np.int64
    ) + head_embedding[:, 0].astype(np.float64).view(np.int64).reshape(-1, 1)

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
        alpha = initial_alpha * (1.0 - (float(n) / float(n_epochs)))

    return head_embedding


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
