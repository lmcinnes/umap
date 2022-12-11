import numpy as np
import numba

from sklearn.neighbors import KDTree
from umap.distances import named_distances


@numba.njit()
def trustworthiness_vector_bulk(
    indices_source, indices_embedded, max_k
):  # pragma: no cover

    n_samples = indices_embedded.shape[0]
    trustworthiness = np.zeros(max_k + 1, dtype=np.float64)

    for i in range(n_samples):
        for j in range(max_k):

            rank = 0
            while indices_source[i, rank] != indices_embedded[i, j]:
                rank += 1

            for k in range(j + 1, max_k + 1):
                if rank > k:
                    trustworthiness[k] += rank - k

    for k in range(1, max_k + 1):
        trustworthiness[k] = 1.0 - trustworthiness[k] * (
            2.0 / (n_samples * k * (2.0 * n_samples - 3.0 * k - 1.0))
        )

    return trustworthiness


def make_trustworthiness_calculator(metric):  # pragma: no cover
    @numba.njit(parallel=True)
    def trustworthiness_vector_lowmem(source, indices_embedded, max_k):

        n_samples = indices_embedded.shape[0]
        trustworthiness = np.zeros(max_k + 1, dtype=np.float64)
        dist_vector = np.zeros(n_samples, dtype=np.float64)

        for i in range(n_samples):

            for j in numba.prange(n_samples):
                dist_vector[j] = metric(source[i], source[j])

            indices_source = np.argsort(dist_vector)

            for j in range(max_k):

                rank = 0
                while indices_source[rank] != indices_embedded[i, j]:
                    rank += 1

                for k in range(j + 1, max_k + 1):
                    if rank > k:
                        trustworthiness[k] += rank - k

        for k in range(1, max_k + 1):
            trustworthiness[k] = 1.0 - trustworthiness[k] * (
                2.0 / (n_samples * k * (2.0 * n_samples - 3.0 * k - 1.0))
            )

        trustworthiness[0] = 1.0

        return trustworthiness

    return trustworthiness_vector_lowmem


def trustworthiness_vector(
    source, embedding, max_k, metric="euclidean"
):  # pragma: no cover
    tree = KDTree(embedding, metric=metric)
    indices_embedded = tree.query(embedding, k=max_k, return_distance=False)
    # Drop the actual point itself
    indices_embedded = indices_embedded[:, 1:]

    dist = named_distances[metric]

    vec_calculator = make_trustworthiness_calculator(dist)

    result = vec_calculator(source, indices_embedded, max_k)

    return result
