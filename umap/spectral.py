import warnings

from warnings import warn

import numpy as np

import scipy.sparse
import scipy.sparse.csgraph
from sklearn.decomposition import TruncatedSVD
from sklearn.manifold import SpectralEmbedding
from sklearn.metrics import pairwise_distances
from sklearn.metrics.pairwise import _VALID_METRICS as SKLEARN_PAIRWISE_VALID_METRICS

from umap.distances import pairwise_special_metric, SPECIAL_METRICS
from umap.sparse import SPARSE_SPECIAL_METRICS, sparse_named_distances


def component_layout(
    data,
    n_components,
    component_labels,
    dim,
    random_state,
    metric="euclidean",
    metric_kwds={},
):
    """Provide a layout relating the separate connected components. This is done
    by taking the centroid of each component and then performing a spectral embedding
    of the centroids.

    Parameters
    ----------
    data: array of shape (n_samples, n_features)
        The source data -- required so we can generate centroids for each
        connected component of the graph.

    n_components: int
        The number of distinct components to be layed out.

    component_labels: array of shape (n_samples)
        For each vertex in the graph the label of the component to
        which the vertex belongs.

    dim: int
        The chosen embedding dimension.

    metric: string or callable (optional, default 'euclidean')
        The metric used to measure distances among the source data points.

    metric_kwds: dict (optional, default {})
        Keyword arguments to be passed to the metric function.
        If metric is 'precomputed', 'linkage' keyword can be used to specify
        'average', 'complete', or 'single' linkage. Default is 'average'

    Returns
    -------
    component_embedding: array of shape (n_components, dim)
        The ``dim``-dimensional embedding of the ``n_components``-many
        connected components.
    """
    if data is None:
        # We don't have data to work with; just guess
        return np.random.random(size=(n_components, dim)) * 10.0

    component_centroids = np.empty((n_components, data.shape[1]), dtype=np.float64)

    if metric == "precomputed":
        # cannot compute centroids from precomputed distances
        # instead, compute centroid distances using linkage
        distance_matrix = np.zeros((n_components, n_components), dtype=np.float64)
        linkage = metric_kwds.get("linkage", "average")
        if linkage == "average":
            linkage = np.mean
        elif linkage == "complete":
            linkage = np.max
        elif linkage == "single":
            linkage = np.min
        else:
            raise ValueError(
                "Unrecognized linkage '%s'. Please choose from "
                "'average', 'complete', or 'single'" % linkage
            )
        for c_i in range(n_components):
            dm_i = data[component_labels == c_i]
            for c_j in range(c_i + 1, n_components):
                dist = linkage(dm_i[:, component_labels == c_j])
                distance_matrix[c_i, c_j] = dist
                distance_matrix[c_j, c_i] = dist
    else:
        for label in range(n_components):
            component_centroids[label] = data[component_labels == label].mean(axis=0)

        if scipy.sparse.isspmatrix(component_centroids):
            warn(
                "Forcing component centroids to dense; if you are running out of "
                "memory then consider increasing n_neighbors."
            )
            component_centroids = component_centroids.toarray()

        if metric in SPECIAL_METRICS:
            distance_matrix = pairwise_special_metric(
                component_centroids,
                metric=metric,
                kwds=metric_kwds,
            )
        elif metric in SPARSE_SPECIAL_METRICS:
            distance_matrix = pairwise_special_metric(
                component_centroids,
                metric=SPARSE_SPECIAL_METRICS[metric],
                kwds=metric_kwds,
            )
        else:
            if callable(metric) and scipy.sparse.isspmatrix(data):
                function_to_name_mapping = {
                    sparse_named_distances[k]: k
                    for k in set(SKLEARN_PAIRWISE_VALID_METRICS)
                    & set(sparse_named_distances.keys())
                }
                try:
                    metric_name = function_to_name_mapping[metric]
                except KeyError:
                    raise NotImplementedError(
                        "Multicomponent layout for custom "
                        "sparse metrics is not implemented at "
                        "this time."
                    )
                distance_matrix = pairwise_distances(
                    component_centroids, metric=metric_name, **metric_kwds
                )
            else:
                distance_matrix = pairwise_distances(
                    component_centroids, metric=metric, **metric_kwds
                )

    affinity_matrix = np.exp(-(distance_matrix**2))

    component_embedding = SpectralEmbedding(
        n_components=dim, affinity="precomputed", random_state=random_state
    ).fit_transform(affinity_matrix)
    component_embedding /= component_embedding.max()

    return component_embedding


def multi_component_layout(
    data,
    graph,
    n_components,
    component_labels,
    dim,
    random_state,
    metric="euclidean",
    metric_kwds={},
    init="random",
    tol=0.0,
    maxiter=0
):
    """Specialised layout algorithm for dealing with graphs with many connected components.
    This will first find relative positions for the components by spectrally embedding
    their centroids, then spectrally embed each individual connected component positioning
    them according to the centroid embeddings. This provides a decent embedding of each
    component while placing the components in good relative positions to one another.

    Parameters
    ----------
    data: array of shape (n_samples, n_features)
        The source data -- required so we can generate centroids for each
        connected component of the graph.

    graph: sparse matrix
        The adjacency matrix of the graph to be embedded.

    n_components: int
        The number of distinct components to be layed out.

    component_labels: array of shape (n_samples)
        For each vertex in the graph the label of the component to
        which the vertex belongs.

    dim: int
        The chosen embedding dimension.

    metric: string or callable (optional, default 'euclidean')
        The metric used to measure distances among the source data points.

    metric_kwds: dict (optional, default {})
        Keyword arguments to be passed to the metric function.

    init: string, either "random" or "tsvd"
        Indicates to initialize the eigensolver. Use "random" (the default) to
        use uniformly distributed random initialization; use "tsvd" to warm-start the
        eigensolver with singular vectors of the Laplacian associated to the largest
        singular values. This latter option also forces usage of the LOBPCG eigensolver;
        with the former, ARPACK's solver ``eigsh`` will be used for smaller Laplacians.

    tol: float, default chosen by implementation
        Stopping tolerance for the numerical algorithm computing the embedding.

    maxiter: int, default chosen by implementation
        Number of iterations the numerical algorithm will go through at most as it
        attempts to compute the embedding.

    Returns
    -------
    embedding: array of shape (n_samples, dim)
        The initial embedding of ``graph``.
    """

    result = np.empty((graph.shape[0], dim), dtype=np.float32)

    if n_components > 2 * dim:
        meta_embedding = component_layout(
            data,
            n_components,
            component_labels,
            dim,
            random_state,
            metric=metric,
            metric_kwds=metric_kwds,
        )
    else:
        k = int(np.ceil(n_components / 2.0))
        base = np.hstack([np.eye(k), np.zeros((k, dim - k))])
        meta_embedding = np.vstack([base, -base])[:n_components]

    for label in range(n_components):
        component_graph = graph.tocsr()[component_labels == label, :].tocsc()
        component_graph = component_graph[:, component_labels == label].tocoo()

        distances = pairwise_distances([meta_embedding[label]], meta_embedding)
        data_range = distances[distances > 0.0].min() / 2.0

        if component_graph.shape[0] < 2 * dim or component_graph.shape[0] <= dim + 1:
            result[component_labels == label] = (
                random_state.uniform(
                    low=-data_range,
                    high=data_range,
                    size=(component_graph.shape[0], dim),
                )
                + meta_embedding[label]
            )
        else:
            component_embedding = _spectral_layout(
                data=None,
                graph=component_graph,
                dim=dim,
                random_state=random_state,
                metric=metric,
                metric_kwds=metric_kwds,
                init=init,
                tol=tol,
                maxiter=maxiter
            )
            expansion = data_range / np.max(np.abs(component_embedding))
            component_embedding *= expansion
            result[component_labels == label] = (
                component_embedding + meta_embedding[label]
            )

    return result


def spectral_layout(
    data,
    graph,
    dim,
    random_state,
    metric="euclidean",
    metric_kwds={},
    tol=0.0,
    maxiter=0
):
    """
    Given a graph compute the spectral embedding of the graph. This is
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

    tol: float, default chosen by implementation
        Stopping tolerance for the numerical algorithm computing the embedding.

    maxiter: int, default chosen by implementation
        Number of iterations the numerical algorithm will go through at most as it
        attempts to compute the embedding.

    Returns
    -------
    embedding: array of shape (n_vertices, dim)
        The spectral embedding of the graph.
    """
    return _spectral_layout(
        data=data,
        graph=graph,
        dim=dim,
        random_state=random_state,
        metric=metric,
        metric_kwds=metric_kwds,
        init="random",
        tol=tol,
        maxiter=maxiter
    )


def tswspectral_layout(
    data,
    graph,
    dim,
    random_state,
    metric="euclidean",
    metric_kwds={},
    method=None,
    tol=0.0,
    maxiter=0
):
    """Given a graph, compute the spectral embedding of the graph. This is
    simply the eigenvectors of the Laplacian of the graph. Here we use the
    normalized laplacian and a truncated SVD-based guess of the
    eigenvectors to "warm" up the eigensolver. This function should
    give results of similar accuracy to the spectral_layout function, but
    may converge more quickly for graph Laplacians that cause
    spectral_layout to take an excessive amount of time to complete.

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

    metric: string or callable (optional, default 'euclidean')
        The metric used to measure distances among the source data points.
        Used only if the multiple connected components are found in the
        graph.

    metric_kwds: dict (optional, default {})
        Keyword arguments to be passed to the metric function.
        If metric is 'precomputed', 'linkage' keyword can be used to specify
        'average', 'complete', or 'single' linkage. Default is 'average'.
        Used only if the multiple connected components are found in the
        graph.

    method: str (optional, default None, values either 'eigsh' or 'lobpcg')
        Name of the eigenvalue computation method to use to compute the spectral
        embedding. If left to None (or empty string), as by default, the method is
        chosen from the number of vectors in play: larger vector collections are
        handled with lobpcg, smaller collections with eigsh. Method names correspond
        to SciPy routines in scipy.sparse.linalg.

    tol: float, default chosen by implementation
        Stopping tolerance for the numerical algorithm computing the embedding.

    maxiter: int, default chosen by implementation
        Number of iterations the numerical algorithm will go through at most as it
        attempts to compute the embedding.

    Returns
    -------
    embedding: array of shape (n_vertices, dim)
        The spectral embedding of the graph.
    """
    return _spectral_layout(
        data=data,
        graph=graph,
        dim=dim,
        random_state=random_state,
        metric=metric,
        metric_kwds=metric_kwds,
        init="tsvd",
        method=method,
        tol=tol,
        maxiter=maxiter
    )


def _spectral_layout(
    data,
    graph,
    dim,
    random_state,
    metric="euclidean",
    metric_kwds={},
    init="random",
    method=None,
    tol=0.0,
    maxiter=0
):
    """General implementation of the spectral embedding of the graph, derived as
    a subset of the eigenvectors of the normalized Laplacian of the graph. The numerical
    method for computing the eigendecomposition is chosen through heuristics.

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

    metric: string or callable (optional, default 'euclidean')
        The metric used to measure distances among the source data points.
        Used only if the multiple connected components are found in the
        graph.

    metric_kwds: dict (optional, default {})
        Keyword arguments to be passed to the metric function.
        If metric is 'precomputed', 'linkage' keyword can be used to specify
        'average', 'complete', or 'single' linkage. Default is 'average'.
        Used only if the multiple connected components are found in the
        graph.

    init: string, either "random" or "tsvd"
        Indicates to initialize the eigensolver. Use "random" (the default) to
        use uniformly distributed random initialization; use "tsvd" to warm-start the
        eigensolver with singular vectors of the Laplacian associated to the largest
        singular values. This latter option also forces usage of the LOBPCG eigensolver;
        with the former, ARPACK's solver ``eigsh`` will be used for smaller Laplacians.

    method: string -- either "eigsh" or "lobpcg" -- or None
        Name of the eigenvalue computation method to use to compute the spectral
        embedding. If left to None (or empty string), as by default, the method is
        chosen from the number of vectors in play: larger vector collections are
        handled with lobpcg, smaller collections with eigsh. Method names correspond
        to SciPy routines in scipy.sparse.linalg.

    tol: float, default chosen by implementation
        Stopping tolerance for the numerical algorithm computing the embedding.

    maxiter: int, default chosen by implementation
        Number of iterations the numerical algorithm will go through at most as it
        attempts to compute the embedding.

    Returns
    -------
    embedding: array of shape (n_vertices, dim)
        The spectral embedding of the graph.
    """
    n_samples = graph.shape[0]
    n_components, labels = scipy.sparse.csgraph.connected_components(graph)

    if n_components > 1:
        return multi_component_layout(
            data,
            graph,
            n_components,
            labels,
            dim,
            random_state,
            metric=metric,
            metric_kwds=metric_kwds,
        )

    sqrt_deg = np.sqrt(np.asarray(graph.sum(axis=0)).squeeze())
    # standard Laplacian
    # D = scipy.sparse.spdiags(diag_data, 0, graph.shape[0], graph.shape[0])
    # L = D - graph
    # Normalized Laplacian
    I = scipy.sparse.identity(graph.shape[0], dtype=np.float64)
    D = scipy.sparse.spdiags(
        1.0 / sqrt_deg, 0, graph.shape[0], graph.shape[0]
    )
    L = I - D * graph * D
    if not scipy.sparse.issparse(L):
        L = np.asarray(L)

    k = dim + 1
    num_lanczos_vectors = max(2 * k + 1, int(np.sqrt(graph.shape[0])))
    gen = (
        random_state
        if isinstance(random_state, (np.random.Generator, np.random.RandomState))
        else np.random.default_rng(seed=random_state)
    )
    if not method:
        method = "eigsh" if L.shape[0] < 2000000 else "lobpcg"

    try:
        if init == "random":
            X = gen.normal(size=(L.shape[0], k))
        elif init == "tsvd":
            X = TruncatedSVD(
                n_components=k,
                random_state=random_state,
                # algorithm="arpack"
            ).fit_transform(L)
        else:
            raise ValueError(
                "The init parameter must be either 'random' or 'tsvd': "
                f"{init} is invalid."
            )
        # For such a normalized Laplacian, the first eigenvector is always
        # proportional to sqrt(degrees). We thus replace the first t-SVD guess
        # with the exact value.
        X[:, 0] = sqrt_deg / np.linalg.norm(sqrt_deg)

        if method == "eigsh":
            eigenvalues, eigenvectors = scipy.sparse.linalg.eigsh(
                L,
                k,
                which="SM",
                ncv=num_lanczos_vectors,
                tol=tol or 1e-4,
                v0=np.ones(L.shape[0]),
                maxiter=maxiter or graph.shape[0] * 5,
            )
        elif method == "lobpcg":
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    category=UserWarning,
                    message=r"(?ms).*not reaching the requested tolerance",
                    action="error"
                )
                eigenvalues, eigenvectors = scipy.sparse.linalg.lobpcg(
                    L,
                    np.asarray(X),
                    largest=False,
                    tol=tol or 1e-4,
                    maxiter=maxiter or 5 * graph.shape[0]
                )
        else:
            raise ValueError("Method should either be None, 'eigsh' or 'lobpcg'")

        order = np.argsort(eigenvalues)[1:k]
        return eigenvectors[:, order]
    except (scipy.sparse.linalg.ArpackError, UserWarning):
        warn(
            "Spectral initialisation failed! The eigenvector solver\n"
            "failed. This is likely due to too small an eigengap. Consider\n"
            "adding some noise or jitter to your data.\n\n"
            "Falling back to random initialisation!"
        )
        return gen.uniform(low=-10.0, high=10.0, size=(graph.shape[0], dim))
