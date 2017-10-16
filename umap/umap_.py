from scipy.optimize import curve_fit
from sklearn.base import BaseEstimator
from collections import deque, namedtuple

import numpy as np
import numba
import cffi

ffi = cffi.FFI()
ffi.cdef('int rand();')
ffi.cdef('double erf(double);')
c = ffi.dlopen(None)
c_rand = c.rand
c_erf = c.erf


@numba.njit('f8[:, :, :](i8,i8)')
def make_heap(n_points, size):
    result = np.zeros((3, n_points, size))
    result[0] = -1
    result[1] = np.infty
    result[2] = 0

    return result


@numba.jit('i8(f8[:,:,:],i8,f8,i8,i8)')
def heap_push(heap, row, weight, index, flag):
    indices = heap[0, row]
    weights = heap[1, row]
    is_new = heap[2, row]

    if weight > weights[0]:
        return 0

    for i in range(indices.shape[0]):
        if index == indices[i]:
            return 0

    # insert val at position zero
    weights[0] = weight
    indices[0] = index
    is_new[0] = flag

    # descend the heap, swapping values until the max heap criterion is met
    i = 0
    while True:
        ic1 = 2 * i + 1
        ic2 = ic1 + 1

        if ic1 >= heap.shape[2]:
            break
        elif ic2 >= heap.shape[2]:
            if weights[ic1] > weight:
                i_swap = ic1
            else:
                break
        elif weights[ic1] >= weights[ic2]:
            if weight < weights[ic1]:
                i_swap = ic1
            else:
                break
        else:
            if weight < weights[ic2]:
                i_swap = ic2
            else:
                break

        weights[i] = weights[i_swap]
        indices[i] = indices[i_swap]
        is_new[i] = is_new[i_swap]

        i = i_swap

    weights[i] = weight
    indices[i] = index
    is_new[i] = flag

    return 1


@numba.njit('f8(f8[:],f8[:])')
def rdist(x, y):
    result = 0.0
    for i in range(x.shape[0]):
        result += (x[i] - y[i]) ** 2

    return result


@numba.njit(parallel=True)
def build_candidates(current_graph, n_vertices, n_neighbors, K):
    candidate_neighbors = make_heap(n_vertices, K)
    for i in range(n_vertices):
        for j in range(n_neighbors):
            if current_graph[0, i, j] < 0:
                continue
            idx = current_graph[0, i, j]
            isn = current_graph[2, i, j]
            d = np.random.random()
            heap_push(candidate_neighbors, i, d, idx, isn)
            heap_push(candidate_neighbors, idx, d, i, isn)
            current_graph[2, i, j] = 0

    return candidate_neighbors


@numba.njit(parallel=True)
def nn_descent(data, n_neighbors, K=50, n_iters=10, delta=0.001, rho=0.5):
    n_vertices = data.shape[0]
    current_graph = make_heap(data.shape[0], n_neighbors)

    for i in range(data.shape[0]):
        indices = np.random.choice(data.shape[0], size=n_neighbors,
                                   replace=False)
        for j in range(indices.shape[0]):
            d = rdist(data[i], data[indices[j]])
            heap_push(current_graph, i, d, indices[j], 1)
            heap_push(current_graph, indices[j], d, i, 1)

    for n in range(n_iters):

        candidate_neighbors = build_candidates(current_graph, n_vertices,
                                               n_neighbors, K)

        c = 0
        for i in range(n_vertices):
            for j in range(K):
                p = int(candidate_neighbors[0, i, j])
                if p < 0 or np.random.random() < rho:
                    continue
                for k in range(K):
                    q = int(candidate_neighbors[0, i, k])
                    if q < 0 or not (
                        candidate_neighbors[2, i, j] or candidate_neighbors[
                        2, i, k]):
                        continue

                    d = rdist(data[p], data[q])
                    c += heap_push(current_graph, p, d, q, 1)
                    c += heap_push(current_graph, q, d, p, 1)

        if c <= delta * n_neighbors * data.shape[0]:
            break

    return current_graph[:2]


@numba.njit('f8(f8[:],f8[:],f8[:])')
def seuclidean(x, y, v):
    result = 0.0
    for i in range(x.shape[0]):
        result += ((x[i] - y[i]) ** 2) / v[i]

    return result


SMOOTH_K_TOLERANCE = 1e-5
MIN_K_DIST_SCALE = 1e-3
NPY_INFINITY = np.inf


@numba.njit()
def smooth_knn_dist(distances, k, n_iter=128):
    target = np.log(k)
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

            val = 0.0
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

import scipy.sparse

from sklearn.neighbors import KDTree

@numba.jit(parallel=True)
def fuzzy_simplicial_set(X, n_neighbors, oversampling=3):

        n_oversampled_neighbors = (n_neighbors * oversampling)
        dim = X.shape[1]

        rows = np.zeros((X.shape[0] * n_oversampled_neighbors), dtype=np.int64)
        cols = np.zeros((X.shape[0] * n_oversampled_neighbors), dtype=np.int64)
        vals = np.zeros((X.shape[0] * n_oversampled_neighbors), dtype=np.float64)

        tmp_dists = np.empty(n_oversampled_neighbors, dtype=np.float64)
        densities = np.zeros(X.shape[0])

        tmp_indices, knn_dists = nn_descent(X,
                                            n_neighbors=n_oversampled_neighbors,
                                            K=60)
        knn_indices = tmp_indices.astype(np.int64)
        for i in range(knn_indices.shape[0]):
            order = np.argsort(knn_dists[i])
            knn_dists[i]  = knn_dists[i][order]
            knn_indices[i] = knn_indices[i][order]

#         for i in range(knn_indices.shape[0]):
#             v = np.sqrt(X[knn_indices[i, 1:n_neighbors//3]].var(axis=0))
#             if np.mean(v) > 0:
#                 v += 0.1 * np.mean(v)
#             else:
#                 v[:] = 1.0
#             for j in range(knn_dists.shape[1]):
#                 knn_dists[i, j] = seuclidean(X[i], X[knn_indices[i, j]], v)

        sigmas, rhos = smooth_knn_dist(knn_dists, n_neighbors)

        for i in range(knn_indices.shape[0]):

            for j in range(n_oversampled_neighbors):
                if knn_indices[i, j] == i:
                    val = 0.0
                elif knn_dists[i, j] == 0.0:
                    val = 1.0
                else:
                    val = np.exp(-((knn_dists[i, j] - rhos[i]) / sigmas[i]))

#                 if sigmas[i] > sigmas[knn_indices[i, j]]:
#                     val *= sigmas[knn_indices[i, j]] / sigmas[i]
#                 else:
#                     val *= sigmas[i] / sigmas[knn_indices[i, j]]

                rows[i * n_oversampled_neighbors + j] = i
                cols[i * n_oversampled_neighbors + j] = knn_indices[i, j]
                vals[i * n_oversampled_neighbors + j] = val


        result = scipy.sparse.coo_matrix((vals, (rows, cols)))
        result.eliminate_zeros()

        transpose = result.transpose()

        prod_matrix = result.multiply(transpose)

        result = result + transpose - prod_matrix
        result.eliminate_zeros()

        return result


@numba.jit()
def create_sampler(probabilities):
    prob = np.zeros(probabilities.shape[0], dtype=np.float64)
    alias = np.zeros(probabilities.shape[0], dtype=np.int64)

    norm_prob = probabilities.shape[0] * probabilities / probabilities.sum()

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
def sample(prob, alias):
    k = c_rand() % prob.shape[0]
    u = c_rand() / 0x7fffffff

    if u < prob[k]:
        return k
    else:
        return alias[k]


@numba.jit()
def spectral_layout(graph, dim):

    diag_data = np.asarray(graph.sum(axis=0))
    D = scipy.sparse.spdiags(diag_data, 0, graph.shape[0], graph.shape[0])
    L = D - graph

    k = dim + 1
    num_lanczos_vectors = max(2 * k + 1, int(np.sqrt(graph.shape[0])))
    eigenvalues, eigenvectors = scipy.sparse.linalg.eigsh(
        L, k,
        which='SM',
        ncv=num_lanczos_vectors,
        tol=1e-4,
        maxiter=graph.shape[0] * 5)
    order = np.argsort(eigenvalues)[1:k]
    return eigenvectors[:, order]


@numba.njit()
def clip(val):
    return 8.0 * c_erf(val / 8.0)


@numba.njit()
def optimize_layout(embedding, positive_head, positive_tail,
                    n_edge_samples, n_vertices, prob, alias,
                    a, b, gamma=1.0, initial_alpha=1.0):

    dim = embedding.shape[1]
    alpha = initial_alpha

    for i in range(n_edge_samples):

        if c_rand() / 0x7fffffff < 0.8:
            is_negative_sample = True
        else:
            is_negative_sample = False

        if is_negative_sample:
            edge = c_rand() % (n_vertices ** 2)
            j = edge // n_vertices
            k = edge % n_vertices
        else:
            edge = sample(prob, alias)
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
                grad_coeff = 128.0

        else:

            grad_coeff = (-2.0 * a * b * pow(dist_squared, b - 1.0))
            grad_coeff /= (a * pow(dist_squared, b) + 1.0)

        for d in range(dim):
            grad_d = clip(grad_coeff * (current[d] - other[d]))
            current[d] += grad_d * alpha
            other[d] += -grad_d * alpha

        if i % 10000 == 0:
            alpha = np.exp(
                -0.69314718055994529 * ((3 * i) / n_edge_samples) ** 2)
            if alpha < (initial_alpha * 0.0001):
                alpha = initial_alpha * 0.0001;

    return embedding


def simplicial_set_embedding(graph, n_components,
                             initial_alpha, a, b,
                             gamma, n_edge_samples,
                             init):
    graph = graph.tocoo()
    graph.sum_duplicates()
    n_vertices = graph.shape[0]

    prob, alias = create_sampler(graph.data)

    if init == 'random':
        embedding = np.random.uniform(low=-10.0, high=10.0,
                                      size=(graph.shape[0], 2))
    else:
        # We add a little noise to avoid local minima for optimization to come
        embedding = (spectral_layout(graph, n_components) * 100.0) + \
                    np.random.normal(scale=0.05,
                                     size=[graph.shape[0],
                                           n_components])

    if n_edge_samples <= 0:
        n_edge_samples = (graph.shape[0] // 150) * 1000000

    positive_head = graph.row
    positive_tail = graph.col

    embedding = optimize_layout(embedding, positive_head, positive_tail,
                                n_edge_samples, n_vertices,
                                prob, alias, a, b, gamma, initial_alpha)

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


class UMAP (BaseEstimator):
    """Uniform Manifold Approximation and Projection

    Finds a low dimensional embedding of the data that approximates
    an underlying manifold.

    Parameters
    ----------
    n_neighbors: int (optional, default 50)
        The size of local neighborhood (in terms of number of neighboring
        sample points) used for manifold approximation. Larger values
        result in more global views of the manifold, while smaller
        values result in more local data being preserved. In general
        values should be in the range 10 to 100.

    n_components: int (optional, default 2)
        The dimension of the space to embed into. This defaults to 2 to
        provide easy visualization, but can reasonably be set to any
        integer value in the range 2 to 100.

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
            * 'random': assign initial emebdding positions at random.

    min_dist: float (optional, default 0.25)
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

    oversampling: int (optional, default 3)
        The scaling factor for the number of neighbors to sample to attempt
        to find the local neighborhood in manifold distance. If set to be too
        large this can have a significant negative impact on performance.
    """

    def __init__(self,
                 n_neighbors=50,
                 n_components=2,
                 gamma=1.0,
                 n_edge_samples=None,
                 alpha=1.0,
                 init='spectral',
                 spread=1.0,
                 min_dist=0.25,
                 a=None,
                 b=None,
                 oversampling=3
                 ):

        self.n_neighbors = n_neighbors
        self.n_edge_samples = n_edge_samples
        self.init = init
        self.n_components = n_components
        self.gamma = gamma
        self.initial_alpha = alpha
        self.alpha = alpha

        self.spread = spread
        self.min_dist = min_dist

        self.oversampling = 3

        if a is None or b is None:
            self.a, self.b = find_ab_params(self.spread, self.min_dist)
        else:
            self.a = a
            self.b = b

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
        X = X.astype(np.float64)

        graph = fuzzy_simplicial_set(X, self.n_neighbors, self.oversampling)

        if self.n_edge_samples is None:
            n_edge_samples = 0
        else:
            n_edge_samples = self.n_edge_samples

        self.embedding_ = simplicial_set_embedding(
            graph,
            self.n_components,
            self.initial_alpha,
            self.a,
            self.b,
            self.gamma,
            n_edge_samples,
            self.init
        )

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
