import cython

from cython cimport view

import numpy as np
cimport numpy as np

from collections import deque
from libc.math cimport exp, pow, erf, sqrt, isfinite, ceil, log, fabs
# from libcpp.unordered_set cimport unordered_set

import scipy.sparse

cdef extern from "numpy/npy_math.h":
    float NPY_INFINITY

from sklearn .neighbors import KDTree

cdef extern from "gsl/gsl_rng.h":
    ctypedef struct gsl_rng_type:
        pass
    ctypedef struct gsl_rng:
        pass
    const gsl_rng_type *gsl_rng_taus
    gsl_rng *gsl_rng_alloc(const gsl_rng_type *T)
    void gsl_rng_free(gsl_rng *r)
    double gsl_rng_uniform(const gsl_rng *r) nogil
    unsigned long int gsl_rng_uniform_int(const gsl_rng *r, unsigned long int n) nogil

cdef float SMOOTH_K_TOLERANCE = 1e-5

cpdef tuple smooth_knn_dist(
    np.ndarray[np.float64_t, ndim=2] distances,
    np.float64_t k,
    np.int64_t n_iter=128
):

    cdef np.int64_t i, j
    cdef np.int64_t n
    cdef np.float64_t lo, hi, mid, val, psum
    cdef np.ndarray[np.float64_t, ndim=1] result = np.empty(distances.shape[0],
                                                            dtype=np.float64)
    cdef np.ndarray[np.float64_t, ndim=1] rho = np.empty(distances.shape[0],
                                                            dtype=np.float64)
    cdef np.float64_t target = np.log(k)

    for i in range(distances.shape[0]):
        lo = 0.0
        hi = NPY_INFINITY
        mid = 1.0

        rho[i] = np.partition(distances[i], 1)[1]

        for n in range(n_iter):

            val = 0.0
            psum = 0.0
            for j in range(1, distances.shape[1]):
                psum += exp(-((distances[i, j] - rho[i]) / mid))
            val = psum

            if fabs(val - target) < SMOOTH_K_TOLERANCE:
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

    return result, rho

@cython.boundscheck(False)
@cython.cdivision(True)
cdef np.float64_t seuclidean(np.float64_t *a, np.float64_t *b, np.float64_t *v, np.int64_t size) nogil except -1:

    cdef np.int64_t i
    cdef np.float64_t diff
    cdef np.float64_t result

    result = 0.0
    for i in range(size):
        diff = a[i] - b[i]
        result += (diff * diff) / v[i]

    return sqrt(result)

cpdef object fuzzy_simplicial_set(
    np.ndarray[np.float64_t, ndim=2] X,
    np.int64_t n_neighbors,
    np.int64_t oversampling=3
):

        cdef np.float64_t[:, :] x_view
        cdef np.ndarray[np.float64_t, ndim=2] knn_dists
        cdef np.ndarray[np.int64_t, ndim=2] knn_indices

        cdef np.int64_t dim

        cdef np.float64_t *center
        cdef np.float64_t *neighbor
        cdef np.float64_t[:] v

        cdef np.ndarray[np.float64_t, ndim=1] vals
        cdef np.ndarray[np.int64_t, ndim=1] rows
        cdef np.ndarray[np.int64_t, ndim=1] cols

        cdef np.ndarray[np.float64_t, ndim=1] new_dists
        cdef np.ndarray[np.int64_t, ndim=1] new_neighbor_indices
        cdef np.int64_t[:] idxs, tmp_indices

        cdef np.float64_t val, v_prod, v_normalize, r, normalizing_exponent
        cdef np.int64_t i, j, k, col_index
        cdef np.float64_t[:] tmp_dists_view
        cdef np.float64_t[:] densities
        cdef np.ndarray[np.float64_t, ndim=1] sigmas

        n_oversampled_neighbors = <np.int64_t> (n_neighbors * oversampling)
        x_view = X
        dim = X.shape[1]

        rows = np.zeros((X.shape[0] * n_oversampled_neighbors), dtype=np.int64)
        cols = np.zeros((X.shape[0] * n_oversampled_neighbors), dtype=np.int64)
        vals = np.zeros((X.shape[0] * n_oversampled_neighbors), dtype=np.float64)

        tmp_dists = np.empty(n_oversampled_neighbors, dtype=np.float64)
        tmp_dists_view = tmp_dists
        densities = np.zeros(X.shape[0])

        tree = KDTree(X)
        knn_dists, knn_indices = tree.query(X, k=n_oversampled_neighbors, dualtree=True)

        for i in range(knn_indices.shape[0]):
            v = np.sqrt(X[knn_indices[i, 1:n_neighbors//3]].var(axis=0))
            v += 0.1 * np.mean(v)
            for j in range(knn_dists.shape[1]):
                knn_dists[i, j] = seuclidean(&x_view[i, 0],
                                             &x_view[knn_indices[i, j], 0],
                                             &v[0],
                                             dim)

        sigmas, rhos = smooth_knn_dist(knn_dists, n_neighbors)

        for i in range(knn_indices.shape[0]):

            for j in range(n_oversampled_neighbors):
                if j == 0:
                    val = 0.0
                else:
                    val = exp(-((knn_dists[i, j] - rhos[i]) / sigmas[i]))

                if sigmas[i] > sigmas[knn_indices[i, j]]:
                    val *= sigmas[knn_indices[i, j]] / sigmas[i]
                else:
                    val *= sigmas[i] / sigmas[knn_indices[i, j]]

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

cdef class WalkerAliasSampler (object):

    cdef np.int64_t size
    cdef np.ndarray prob
    cdef np.ndarray alias
    cdef gsl_rng *rng

    def __init__(self, probabilities):

        self.size = probabilities.shape[0]
        self.prob = np.zeros(probabilities.shape[0], dtype=np.float64)
        self.alias = np.zeros(probabilities.shape[0], dtype=np.int64)
        self.rng = gsl_rng_alloc(gsl_rng_taus)

        norm_prob = self.size * probabilities / probabilities.sum()

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

            self.prob[j] = norm_prob[j]
            self.alias[j] = k

            norm_prob[k] -= (1.0 - norm_prob[j])

            if norm_prob[k] < 1.0:
                small.append(k)
            else:
                large.append(k)

        while small:
            self.prob[small.pop()] = 1.0

        while large:
            self.prob[large.pop()] = 1.0

    def __dealloc__(self):
        gsl_rng_free(self.rng)

    cdef np.int64_t sample(self) nogil except -1:

        cdef np.int64_t k
        cdef np.double_t u
        cdef np.int64_t size = self.size
        cdef np.float64_t * prob = <np.float64_t *> self.prob.data
        cdef np.int64_t * alias = <np.int64_t *> self.alias.data

        k = gsl_rng_uniform_int(self.rng, size)
        u = gsl_rng_uniform(self.rng)

        if u < prob[k]:
            return k
        else:
            return alias[k]

cdef inline np.float64_t clip(np.float64_t value) nogil except -1:
    return 8.0 * erf(value / 8.0)

cdef inline np.float64_t rdist(np.float64_t *a, np.float64_t *b, int dim) nogil except -1:
    cdef int d
    cdef np.float64_t dist_squared

    dist_squared = 0
    for d in range(dim):
        dist_squared += ((a[d] - b[d])**2)

    return dist_squared

cdef np.ndarray spectral_layout(object graph, int dim):

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
        max_iter=graph.shape[0] * 5)
    order = np.argsort(eigenvalues)[1:k]
    return eigenvectors[:, order]


cdef class SimpleBloomFilter (object):

    cdef np.float64_t load_factor
    cdef np.int64_t size
    cdef np.int8_t n_hashes
    cdef np.int8_t[:] table

    def __init__(self, size):
        self.load_factor = 5.0
        self.size = <np.int64_t> 2**(ceil(log((size * self.load_factor))
                                             /log(2)))
        self.n_hashes = 3
        self.table = view.array(shape=(self.size,),
                                       itemsize=sizeof(np.int8_t),
                                       format='c')
        self.table[:] = 0


    @cython.cdivision(True)
    cdef long add_key(self, np.int64_t key) except -1:
        cdef np.int64_t hash_val
        cdef np.int8_t i

        hash_val = key % self.size
        for i in range(self.n_hashes):
            self.table[hash_val] = 1
            hash_val = ((5 * hash_val) + 1) % self.size

        return 0

    @cython.boundscheck(False)
    @cython.initializedcheck(False)
    @cython.cdivision(True)
    cdef int has_key(self, np.int64_t key) nogil except -1:
        cdef np.int64_t hash_val
        cdef np.int8_t i

        hash_val = key % self.size
        for i in range(self.n_hashes):
            if self.table[hash_val] == 0:
                return 0
            hash_val = ((5 * hash_val) + 1) % self.size

        return 1


cdef class NegativeSampler (object):

    cdef object graph
    cdef np.int64_t n_non_zeros
    cdef np.int64_t n_zeros
    cdef np.int64_t n_vertices
    cdef np.int64_t n_edges
    cdef WalkerAliasSampler non_zero_sampler
    cdef np.float64_t sample_from_zeros_prob
    cdef int[:] head
    cdef int[:] tail
#     cdef unordered_set[np.int64_t] non_zero_set
    cdef SimpleBloomFilter non_zero_set
    cdef gsl_rng *rng

    def __init__(self, graph):

        cdef np.int64_t i
        cdef np.int64_t index

        self.graph = graph
        self.n_non_zeros = self.graph.nnz
        self.n_zeros  = self.graph.shape[0]**2 - self.graph.nnz
        self.n_vertices = self.graph.shape[0]
        self.n_edges = self.n_vertices**2
        self.sample_from_zeros_prob = 1.0 - ((1.0 - self.graph.data).sum()
                                             / (self.n_zeros))
        self.non_zero_sampler = WalkerAliasSampler(1.0 - self.graph.data)
        self.head = self.graph.row
        self.tail = self.graph.col

#         self.non_zero_set = self.graph.row * self.n_vertices + self.graph.col
        self.non_zero_set = SimpleBloomFilter(self.n_non_zeros)
        for i in range(self.n_non_zeros):
            self.non_zero_set.add_key(self.graph.row[i] *
                                      self.n_vertices +
                                      self.graph.col[i])

        self.rng = gsl_rng_alloc(gsl_rng_taus)

    def __dealloc__(self):
        gsl_rng_free(self.rng)

    @cython.boundscheck(False)
    @cython.initializedcheck(False)
    @cython.cdivision(True)
    cdef np.int64_t sample(self) nogil except -1:

        cdef np.int64_t row, col, index
        cdef np.int64_t edge

        if gsl_rng_uniform(self.rng) < self.sample_from_zeros_prob:
            row = gsl_rng_uniform_int(self.rng, self.n_vertices)
            col = gsl_rng_uniform_int(self.rng, self.n_vertices)
            index = row * self.n_vertices + col
    #             while self.non_zero_set.count(index) > 0:
            while self.non_zero_set.has_key(index):
                index = gsl_rng_uniform_int(self.rng, self.n_edges)
            return index
        else:
            edge = self.non_zero_sampler.sample()
            return self.head[edge] * self.n_vertices +  self.tail[edge]

@cython.boundscheck(False)
@cython.cdivision(True)
cpdef np.ndarray[np.float64_t, ndim=2] simplicial_set_embedding(
    object graph,
    np.int64_t n_components,
    np.float64_t initial_alpha,
    np.float64_t a,
    np.float64_t b,
    np.float64_t gamma,
    np.int64_t n_edge_samples,
    str init
):

    cdef np.int64_t i, j, k, d, edge, n_sampled
    cdef np.int64_t n_vertices
    cdef np.int64_t n_edges
    cdef np.float64_t[:, :] embedding_view
    cdef np.float64_t* current
    cdef np.float64_t* other
    cdef np.float64_t dist_squared, grad_coeff, grad_d
    cdef int dim = <int> n_components
    cdef int[:] positive_head, positive_tail, negative_head, negative_tail
    cdef int is_negative_sample
    cdef gsl_rng *rng
    cdef np.float64_t alpha = initial_alpha
    cdef WalkerAliasSampler edge_sampler
    cdef NegativeSampler negative_sampler
    cdef np.float64_t negative_sample_prob

    rng = gsl_rng_alloc(gsl_rng_taus)

    graph = graph.tocoo()
    graph.sum_duplicates()
    n_edges = graph.nnz
    n_vertices = graph.shape[0]

    edge_sampler = WalkerAliasSampler(graph.data)
    negative_sampler = NegativeSampler(graph)

    negative_edge_weight = (1.0 - graph.data).sum() + (graph.shape[0]**2 - n_edges)
    negative_sample_prob = 1.0 - (graph.sum() / (0.5 * negative_edge_weight))

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

    embedding_view = embedding
    positive_head = graph.row
    positive_tail = graph.col

    for i in range(n_edge_samples):

#         if gsl_rng_uniform(rng) < negative_sample_prob:
        if gsl_rng_uniform(rng) < 0.8:
            is_negative_sample = True
        else:
            is_negative_sample = False


        if is_negative_sample:
            edge = negative_sampler.sample()
            j = edge // n_vertices
            k = edge % n_vertices
        else:
            edge = edge_sampler.sample()
            j = positive_head[edge]
            k = positive_tail[edge]

        current = &embedding_view[j, 0]
        other = &embedding_view[k, 0]

        dist_squared = rdist(current, other, dim)

        if is_negative_sample:

            grad_coeff = (2.0 * gamma * b)
            grad_coeff /= (0.001 + dist_squared) * (a * pow(dist_squared, b) + 1)

            if not isfinite(grad_coeff):
                grad_coeff = 128.0

        else:

            grad_coeff = (-2.0 * a * b * pow(dist_squared, b - 1.0))
            grad_coeff /= (a * pow(dist_squared, b) + 1.0)


        for d in range(dim):
            grad_d = clip(grad_coeff * (current[d] - other[d]))
            current[d] += grad_d * alpha
            other[d] += -grad_d * alpha

        if i % 10000 == 0:
#             alpha = initial_alpha * (1.0 - ((i + 1.0) / n_edge_samples));
            alpha = exp( -0.69314718055994529 * ((3 * i) / n_edge_samples)**2 )
            if alpha < (initial_alpha * 0.0001):
                alpha = initial_alpha * 0.0001;

    gsl_rng_free(rng)

    return embedding

