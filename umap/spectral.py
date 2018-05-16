import numpy as np

import scipy.sparse
import scipy.sparse.csgraph

from sklearn.manifold import SpectralEmbedding
from sklearn.metrics import pairwise_distances
from warnings import warn

def component_layout(data, n_components, component_labels, dim, metric='euclidean', metric_kwds={}):

    component_centroids = np.empty((n_components, data.shape[1]), dtype=np.float64)

    for label in range(n_components):
        component_centroids[label] = data[component_labels == label].mean(axis=0)

    distance_matrix = pairwise_distances(component_centroids, metric=metric, **metric_kwds)
    affinity_matrix = np.exp(-distance_matrix**2)

    component_embedding = SpectralEmbedding(n_components=dim,
                                            affinity='precomputed').fit_transform(affinity_matrix)

    return component_embedding


def multi_component_layout(data, graph, n_components, component_labels, dim, random_state, metric='euclidean', metric_kwds={}):

    result = np.empty((graph.shape[0], dim), dtype=np.float32)

    if n_components > 2 * dim:
        meta_embedding = component_layout(data, n_components, component_labels, dim,
                                          metric=metric, metric_kwds=metric_kwds)
    else:
        k = int(np.ceil(n_components / 2.0))
        base = np.hstack([np.eye(k), np.zeros((k, dim - k))])
        meta_embedding = np.vstack([base, -base])[:n_components]

    for label in range(n_components):
        component_graph = graph.tocsr()[component_labels == label, :].tocsc()
        component_graph = component_graph[:, component_labels == label].tocoo()

        distances = pairwise_distances([meta_embedding[label]], meta_embedding)
        data_range = distances[distances > 0.0].min() / 2.0

        if component_graph.shape[0] < 2 * dim:
            result[component_labels == label] = \
                random_state.uniform(low=-data_range, high=data_range, size=(component_graph.shape[0], dim)) + \
                    meta_embedding[label]
            continue

        diag_data = np.asarray(component_graph.sum(axis=0))
        # standard Laplacian
        # D = scipy.sparse.spdiags(diag_data, 0, graph.shape[0], graph.shape[0])
        # L = D - graph
        # Normalized Laplacian
        I = scipy.sparse.identity(component_graph.shape[0], dtype=np.float64)
        D = scipy.sparse.spdiags(1.0 / np.sqrt(diag_data), 0, component_graph.shape[0],
                                 component_graph.shape[0])
        L = I - D * component_graph * D

        k = dim + 1
        num_lanczos_vectors = max(2 * k + 1, int(np.sqrt(component_graph.shape[0])))
        try:
            eigenvalues, eigenvectors = scipy.sparse.linalg.eigsh(
                L, k,
                which='SM',
                ncv=num_lanczos_vectors,
                tol=1e-4,
                v0=np.ones(L.shape[0]),
                maxiter=graph.shape[0] * 5)
            order = np.argsort(eigenvalues)[1:k]
            component_embedding = eigenvectors[:, order]
            expansion = data_range / np.max(np.abs(component_embedding))
            component_embedding *= expansion
            result[component_labels == label] = component_embedding + meta_embedding[label]
        except scipy.sparse.linalg.ArpackError:
            warn('WARNING: spectral initialisation failed! The eigenvector solver\n'
                 'failed. This is likely due to too small an eigengap. Consider\n'
                 'adding some noise or jitter to your data.\n\n'
                 'Falling back to random initialisation!')
            result[component_labels == label] = \
                random_state.uniform(low=-data_range, high=data_range, size=(component_graph.shape[0], dim)) + \
                    meta_embedding[label]

    return result

def spectral_layout(data, graph, dim, random_state, metric='euclidean', metric_kwds={}):
    """Given a graph compute the spectral embedding of the graph. This is
    simply the eigenvectors of the laplacian of the graph. Here we use the
    normalized laplacian.

    Parameters
    ----------
    data: array of shape (n_samples, n_features)
        The source data

    graph: sparse matrix
        The (weighted) adjacency matrix of the graph as a sparse matrix.

    dim: int
        The dimension of the space into which to embed.

    random_state: numpy RandomState or equivalent
        A state capable being used as a numpy random state.

    Returns
    -------
    embedding: array of shape (n_vertices, dim)
        The spectral embedding of the graph.
    """
    n_samples = graph.shape[0]
    n_components, labels = scipy.sparse.csgraph.connected_components(graph)

    if n_components > 1:
        warn('Embedding {} connected components using meta-embedding (experimental)'.format(n_components))
        return multi_component_layout(data, graph, n_components, labels, dim, random_state,
                                      metric=metric, metric_kwds=metric_kwds)

    diag_data = np.asarray(graph.sum(axis=0))
    # standard Laplacian
    # D = scipy.sparse.spdiags(diag_data, 0, graph.shape[0], graph.shape[0])
    # L = D - graph
    # Normalized Laplacian
    I = scipy.sparse.identity(graph.shape[0], dtype=np.float64)
    D = scipy.sparse.spdiags(1.0 / np.sqrt(diag_data), 0, graph.shape[0],
                             graph.shape[0])
    L = I - D * graph * D

    k = dim + 1
    num_lanczos_vectors = max(2 * k + 1, int(np.sqrt(graph.shape[0])))
    try:
        eigenvalues, eigenvectors = scipy.sparse.linalg.eigsh(
            L, k,
            which='SM',
            ncv=num_lanczos_vectors,
            tol=1e-4,
            v0=np.ones(L.shape[0]),
            maxiter=graph.shape[0] * 5)
        order = np.argsort(eigenvalues)[1:k]
        if n_components == 1:
            return eigenvectors[:, order]
        else:
            init = random_state.uniform(low=-10.0, high=10.0,
                                        size=(n_samples, dim))
            init[labels == largest_component] = eigenvectors[:, order]
            return init
    except scipy.sparse.linalg.ArpackError:
        warn('WARNING: spectral initialisation failed! The eigenvector solver\n'
             'failed. This is likely due to too small an eigengap. Consider\n'
             'adding some noise or jitter to your data.\n\n'
             'Falling back to random initialisation!')
        return random_state.uniform(low=-10.0, high=10.0,
                                    size=(graph.shape[0], dim))
