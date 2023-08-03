import warnings

from warnings import warn

import numpy as np

import scipy.sparse
import scipy.sparse.csgraph
import sklearn.decomposition
import os
import mkl
import numpy.ctypeslib as npct
import ctypes
import psutil
from sklearn.manifold import SpectralEmbedding
from sklearn.metrics import pairwise_distances
from sklearn.metrics.pairwise import _VALID_METRICS as SKLEARN_PAIRWISE_VALID_METRICS

from umap.distances import pairwise_special_metric, SPECIAL_METRICS
from umap.sparse import SPARSE_SPECIAL_METRICS, sparse_named_distances

proc = psutil.Process(os.getpid())
mkl_rtc = [lib.path for lib in proc.memory_maps() if 'mkl_rt' in lib.path][0]


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
            continue

        diag_data = np.asarray(component_graph.sum(axis=0))
        # standard Laplacian
        # D = scipy.sparse.spdiags(diag_data, 0, graph.shape[0], graph.shape[0])
        # L = D - graph
        # Normalized Laplacian
        I = scipy.sparse.identity(component_graph.shape[0], dtype=np.float64)
        D = scipy.sparse.spdiags(
            1.0 / np.sqrt(diag_data),
            0,
            component_graph.shape[0],
            component_graph.shape[0],
        )
        L = I - D * component_graph * D

        k = dim + 1
        num_lanczos_vectors = max(2 * k + 1, int(np.sqrt(component_graph.shape[0])))
        try:
            eigenvalues, eigenvectors = scipy.sparse.linalg.eigsh(
                L,
                k,
                which="SM",
                ncv=num_lanczos_vectors,
                tol=1e-4,
                v0=np.ones(L.shape[0]),
                maxiter=graph.shape[0] * 5,
            )
            order = np.argsort(eigenvalues)[1:k]
            component_embedding = eigenvectors[:, order]
            expansion = data_range / np.max(np.abs(component_embedding))
            component_embedding *= expansion
            result[component_labels == label] = (
                component_embedding + meta_embedding[label]
            )
        except scipy.sparse.linalg.ArpackError:
            warn(
                "WARNING: spectral initialisation failed! The eigenvector solver\n"
                "failed. This is likely due to too small an eigengap. Consider\n"
                "adding some noise or jitter to your data.\n\n"
                "Falling back to random initialisation!"
            )
            result[component_labels == label] = (
                random_state.uniform(
                    low=-data_range,
                    high=data_range,
                    size=(component_graph.shape[0], dim),
                )
                + meta_embedding[label]
            )

    return result


def mkl_eigenvalue(*args, **kwargs):
    print("Inside mkl_eigenvalue")
    use_fp32 = os.environ.get("UMAP_MKL_FP32", "1") != "0"
    if use_fp32:
        return mkl_eigenvalue_s(*args, **kwargs)
    return mkl_eigenvalue_d(*args, **kwargs)


def mkl_eigenvalue_d(spM, k0, whichE, ncv, tol, maxiter):
    
    
    #mkl_rt = ctypes.cdll.LoadLibrary(mkl_rtc)
    mkl_rt = ctypes.CDLL(mkl_rtc)
    # types
    # MKL_INT is  np.intc if MKL_INTERFACE is LP64
    array_1d_mkl_int_t = npct.ndpointer(dtype=np.intc, ndim=1, flags="CONTIGUOUS")
    array_1d_double_t = npct.ndpointer(dtype=np.double, ndim=1, flags="CONTIGUOUS")

    # set up function mkl_sparse_ee_init
    py_mkl_sparse_ee_init = mkl_rt.mkl_sparse_ee_init
    py_mkl_sparse_ee_init.restupe = None
    py_mkl_sparse_ee_init.argtypes = [array_1d_mkl_int_t]

    sparse_matrix_t = ctypes.c_void_p
    sparse_status_t = ctypes.c_int  # enum
    sparse_operation_t = ctypes.c_int  # enum
    sparse_matrix_type_t = ctypes.c_int  # enum
    sparse_index_base_t = ctypes.c_int  # enum
    sparse_fill_mode_t = ctypes.c_int  # enum
    sparse_diag_type_t = ctypes.c_int  # enum
    sparse_layout_t = ctypes.c_int  # enum
    verbose_mode_t = ctypes.c_int  # enum
    sparse_memory_usage = ctypes.c_int  # enum
    sparse_request_t = ctypes.c_int  # enum

    class SparseMatrixDescr(ctypes.Structure):
        _fields_ = [
            ("type", sparse_matrix_type_t),
            ("mode", sparse_fill_mode_t),
            ("diag", sparse_diag_type_t),
        ]

    # setup mkl_sparse_d_create_csr
    py_mkl_sparse_d_create_csr = mkl_rt.mkl_sparse_d_create_csr
    py_mkl_sparse_d_create_csr.restype = sparse_status_t
    py_mkl_sparse_d_create_csr.argtypes = [
        ctypes.POINTER(ctypes.c_void_p),  # sparse_matrix_t *A
        sparse_index_base_t,  # enum
        ctypes.c_int,  # MKL_INT rows
        ctypes.c_int,  # MKL_INT cols
        array_1d_mkl_int_t,  # MKL_INT *row_start
        array_1d_mkl_int_t,  # MKL_INT *row_end
        array_1d_mkl_int_t,  # MKL_INT *col_indx
        array_1d_double_t,  # double * vals
    ]

    py_mkl_sparse_d_ev = mkl_rt.mkl_sparse_d_ev
    py_mkl_sparse_d_ev.restype = ctypes.c_int
    py_mkl_sparse_d_ev.argtypes = [
        ctypes.c_char_p,  # char * whichE
        array_1d_mkl_int_t,  # MKL_INT * pm
        sparse_matrix_t,  # sparse_matrix_t A,
        SparseMatrixDescr,  # matrix_desc
        ctypes.c_int,  # k0
        ctypes.POINTER(ctypes.c_int),  # MKL_INT *k,
        array_1d_double_t,  # double * E
        array_1d_double_t,  # double * X
        array_1d_double_t,  # double * res
    ]

    py_mkl_sparse_destroy = mkl_rt.mkl_sparse_destroy
    py_mkl_sparse_destroy.returntype = None
    py_mkl_sparse_destroy.argtypes = [sparse_matrix_t]

    one = ctypes.c_double(1.0)
    zero = ctypes.c_double(0.0)
    tol = ctypes.c_int(int(tol))
    compute_vectors = ctypes.c_int(1)
    xgemmC = ctypes.c_char(b"T")
    xgemmN = ctypes.c_char(b"N")
    # whichE = ctypes.c_char(b'S')
    whichE = ctypes.c_char(whichE)
    whichV = ctypes.c_char(b"L")
    num_lanczos_vectors = ctypes.c_int(int(ncv))
    maxiter = ctypes.c_int(int(maxiter))

    k = ctypes.c_int(0)  # to be computed by MKL
    SPARSE_MATRIX_TYPE_GENERAL = 20
    descr = SparseMatrixDescr(type=SPARSE_MATRIX_TYPE_GENERAL)
    mkl_csrA = sparse_matrix_t()

    pm = np.zeros((128,), dtype=np.intc)
    py_mkl_sparse_ee_init(pm)
    pm[1] = tol.value
    pm[2] = 1
    pm[3] = num_lanczos_vectors.value
    pm[4] = maxiter.value
    pm[6] = compute_vectors.value
    pm[
        7
    ] = 1  # 0 = relative,  1 = Use absolute stopping critertia. Iteration is stopped if norm(Ax - lambda*x) < 10^(-pm[1])

    k0 = ctypes.c_int(int(k0))  # how many eigenvalues to compute

    SPARSE_INDEX_BASE_ZERO = 0
    # mkl_sparse_d_create_csr ( &mkl_csrA, SPARSE_INDEX_BASE_ONE, N, M, ia, ia+1, ja, a );
    csr_create_status = py_mkl_sparse_d_create_csr(
        ctypes.byref(mkl_csrA),
        SPARSE_INDEX_BASE_ZERO,
        spM.shape[0],
        spM.shape[1],
        spM.indptr[:-1],
        spM.indptr[1:],
        spM.indices,
        spM.data,
    )

    eigenvals = np.zeros((np.max(k0.value),), dtype=np.float64)
    resid = np.empty_like(eigenvals)
    X = np.empty((k0.value, spM.shape[0]), dtype=np.float64)
    X_flat = X.reshape((-1))
    # X_flat = np.ones((k0.value * spM.shape[0]), dtype=np.double)

    info = py_mkl_sparse_d_ev(
        ctypes.byref(whichE),  # 1
        pm,  # 3
        mkl_csrA,  # 4
        descr,
        k0,
        ctypes.byref(k),
        eigenvals,
        X_flat,
        resid,
    )

    X = X_flat.reshape((k0.value, int(X_flat.shape[0] / k0.value)))
    X = X.T
    X[:, 0] = -X[:, 0]
    X[:, 1] = -X[:, 1]
    return eigenvals, X


def mkl_eigenvalue_s(spM, k0, whichE, ncv, tol, maxiter):  # single-precision version
    

   # mkl_rt = ctypes.cdll.LoadLibrary(mkl_rtc)
    mkl_rt = ctypes.CDLL(mkl_rtc)
    # types
    # MKL_INT is  np.intc if MKL_INTERFACE is LP64
    array_1d_mkl_int_t = npct.ndpointer(dtype=np.intc, ndim=1, flags="CONTIGUOUS")
    array_1d_float_t = npct.ndpointer(dtype=np.float32, ndim=1, flags="CONTIGUOUS")

    # set up function mkl_sparse_ee_init
    py_mkl_sparse_ee_init = mkl_rt.mkl_sparse_ee_init
    py_mkl_sparse_ee_init.restupe = None
    py_mkl_sparse_ee_init.argtypes = [array_1d_mkl_int_t]

    sparse_matrix_t = ctypes.c_void_p
    sparse_status_t = ctypes.c_int  # enum
    sparse_operation_t = ctypes.c_int  # enum
    sparse_matrix_type_t = ctypes.c_int  # enum
    sparse_index_base_t = ctypes.c_int  # enum
    sparse_fill_mode_t = ctypes.c_int  # enum
    sparse_diag_type_t = ctypes.c_int  # enum
    sparse_layout_t = ctypes.c_int  # enum
    verbose_mode_t = ctypes.c_int  # enum
    sparse_memory_usage = ctypes.c_int  # enum
    sparse_request_t = ctypes.c_int  # enum

    class SparseMatrixDescr(ctypes.Structure):
        _fields_ = [
            ("type", sparse_matrix_type_t),
            ("mode", sparse_fill_mode_t),
            ("diag", sparse_diag_type_t),
        ]

    # setup mkl_sparse_d_create_csr
    py_mkl_sparse_s_create_csr = mkl_rt.mkl_sparse_s_create_csr
    py_mkl_sparse_s_create_csr.restype = sparse_status_t
    py_mkl_sparse_s_create_csr.argtypes = [
        ctypes.POINTER(ctypes.c_void_p),  # sparse_matrix_t *A
        sparse_index_base_t,  # enum
        ctypes.c_int,  # MKL_INT rows
        ctypes.c_int,  # MKL_INT cols
        array_1d_mkl_int_t,  # MKL_INT *row_start
        array_1d_mkl_int_t,  # MKL_INT *row_end
        array_1d_mkl_int_t,  # MKL_INT *col_indx
        array_1d_float_t,  # float * vals
    ]

    py_mkl_sparse_s_ev = mkl_rt.mkl_sparse_s_ev
    py_mkl_sparse_s_ev.restype = ctypes.c_int
    py_mkl_sparse_s_ev.argtypes = [
        ctypes.c_char_p,  # char * whichE
        array_1d_mkl_int_t,  # MKL_INT * pm
        sparse_matrix_t,  # sparse_matrix_t A,
        SparseMatrixDescr,  # matrix_desc
        ctypes.c_int,  # k0
        ctypes.POINTER(ctypes.c_int),  # MKL_INT *k,
        array_1d_float_t,  # float * E
        array_1d_float_t,  # float * X
        array_1d_float_t,  # float * res
    ]

    py_mkl_sparse_destroy = mkl_rt.mkl_sparse_destroy
    py_mkl_sparse_destroy.returntype = None
    py_mkl_sparse_destroy.argtypes = [sparse_matrix_t]

    one = ctypes.c_float(1.0)
    zero = ctypes.c_float(0.0)
    tol = ctypes.c_int(int(tol))
    compute_vectors = ctypes.c_int(1)
    xgemmC = ctypes.c_char(b"T")
    xgemmN = ctypes.c_char(b"N")
    # whichE = ctypes.c_char(b'S')
    whichE = ctypes.c_char(whichE)
    whichV = ctypes.c_char(b"L")
    num_lanczos_vectors = ctypes.c_int(int(ncv))
    maxiter = ctypes.c_int(int(maxiter))

    k = ctypes.c_int(0)  # to be computed by MKL
    SPARSE_MATRIX_TYPE_GENERAL = 20
    descr = SparseMatrixDescr(type=SPARSE_MATRIX_TYPE_GENERAL)
    mkl_csrA = sparse_matrix_t()

    pm = np.zeros((128,), dtype=np.intc)
    py_mkl_sparse_ee_init(pm)
    pm[1] = tol.value
    pm[2] = 1
    pm[3] = num_lanczos_vectors.value
    pm[4] = maxiter.value
    pm[6] = compute_vectors.value
    pm[
        7
    ] = 1  # 0 = relative,  1 = Use absolute stopping critertia. Iteration is stopped if norm(Ax - lambda*x) < 10^(-pm[1])

    k0 = ctypes.c_int(int(k0))  # how many eigenvalues to compute

    tmp = np.empty(spM.data.shape, dtype=np.float32)
    tmp[:] = spM.data.astype(np.float32)

    SPARSE_INDEX_BASE_ZERO = 0
    # mkl_sparse_d_create_csr ( &mkl_csrA, SPARSE_INDEX_BASE_ONE, N, M, ia, ia+1, ja, a );
    csr_create_status = py_mkl_sparse_s_create_csr(
        ctypes.byref(mkl_csrA),
        SPARSE_INDEX_BASE_ZERO,
        spM.shape[0],
        spM.shape[1],
        spM.indptr[:-1],
        spM.indptr[1:],
        spM.indices,
        tmp,  # spM.data.astype(np.float32)
    )

    eigenvals = np.zeros((np.max(k0.value),), dtype=np.float32)
    resid = np.empty_like(eigenvals)
    X = np.empty((k0.value, spM.shape[0]), dtype=np.float32)
    X_flat = X.reshape((-1))
    # X_flat = np.ones((k0.value * spM.shape[0]), dtype=np.double)

    info = py_mkl_sparse_s_ev(
        ctypes.byref(whichE),  # 1
        pm,  # 3
        mkl_csrA,  # 4
        descr,
        k0,
        ctypes.byref(k),
        eigenvals,
        X_flat,
        resid,
    )

    X = X_flat.reshape((k0.value, int(X_flat.shape[0] / k0.value)))
    X = X.T
    X[:, 0] = -X[:, 0]
    X[:, 1] = -X[:, 1]
    return eigenvals, X


def spectral_layout(data, graph, dim, random_state, metric="euclidean", metric_kwds={}):
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

    diag_data = np.asarray(graph.sum(axis=0))
    # standard Laplacian
    # D = scipy.sparse.spdiags(diag_data, 0, graph.shape[0], graph.shape[0])
    # L = D - graph
    # Normalized Laplacian
    I = scipy.sparse.identity(graph.shape[0], dtype=np.float64)
    D = scipy.sparse.spdiags(
        1.0 / np.sqrt(diag_data), 0, graph.shape[0], graph.shape[0]
    )
    L = I - D * graph * D

    k = dim + 1
    num_lanczos_vectors = max(2 * k + 1, int(np.sqrt(graph.shape[0])))
    try:
        if L.shape[0] < 2000000:
            # eigenvalues, eigenvectors = scipy.sparse.linalg.eigsh(
            #    L,
            #    k,
            #    which="SM",
            #    ncv=num_lanczos_vectors,
            #    tol=1e-4,
            #    v0=np.ones(L.shape[0]),
            #    maxiter=graph.shape[0] * 5,
            # )
            eigenvalues, eigenvectors = mkl_eigenvalue(
                L,
                k,
                whichE=b"S",
                ncv=num_lanczos_vectors,
                tol=5,
                maxiter=graph.shape[0] * 5,
            )
        else:
            eigenvalues, eigenvectors = scipy.sparse.linalg.lobpcg(
                L, random_state.normal(size=(L.shape[0], k)), largest=False, tol=1e-8
            )
        order = np.argsort(eigenvalues)[1:k]
        return eigenvectors[:, order]
    except scipy.sparse.linalg.ArpackError:
        warn(
            "WARNING: spectral initialisation failed! The eigenvector solver\n"
            "failed. This is likely due to too small an eigengap. Consider\n"
            "adding some noise or jitter to your data.\n\n"
            "Falling back to random initialisation!"
        )
        return random_state.uniform(low=-10.0, high=10.0, size=(graph.shape[0], dim))


def tswspectral_layout(
    data, graph, dim, random_state, metric="euclidean", metric_kwds={}
):
    """Given a graph compute the spectral embedding of the graph. This is
    simply the eigenvectors of the laplacian of the graph. Here we use the
    normalized laplacian and a truncated SVD-based guess of the
    eigenvectors to "warm" up the lobpcg eigensolver. This function should
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

    diag_data = np.asarray(graph.sum(axis=0))
    D = scipy.sparse.spdiags(1.0 / np.sqrt(diag_data), 0, n_samples, n_samples)
    # L is a shifted version of what we will pass to the eigensolver (I - L)
    # The eigenvectors of I - L coincide with the first few singular vectors
    # of L so we can carry out truncated SVD on L to get a guess to pass to lobpcg
    L = D * graph * D

    k = dim + 1
    tsvd = sklearn.decomposition.TruncatedSVD(
        n_components=k, random_state=random_state, algorithm="arpack", tol=1e-2
    )
    guess = tsvd.fit_transform(L)

    # for a normalized Laplacian, the first eigenvector is always sqrt(D) so replace
    # the tsvd guess with the exact value. Scaling it to length one seems to help.
    guess[:, 0] = np.sqrt(diag_data[0] / np.linalg.norm(diag_data[0]))

    I = scipy.sparse.identity(n_samples, dtype=np.float64)

    # lobpcg emits a UserWarning if convergence was not reached within `maxiter`
    # so we will just have to catch that instead of an Error
    # This will also trigger when lobpcg decides the problem size is too small
    # for it to deal with but there is little chance that this would happen
    # in most real use cases
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        try:
            eigenvalues, eigenvectors = scipy.sparse.linalg.lobpcg(
                I - L,
                guess,
                largest=False,
                tol=1e-4,
                maxiter=graph.shape[0] * 5,
            )
        except UserWarning:
            warn(
                "WARNING: spectral initialisation failed! The eigenvector solver\n"
                "failed. This is likely due to too small an eigengap. Consider\n"
                "adding some noise or jitter to your data.\n\n"
                "Falling back to random initialisation!"
            )
            return random_state.uniform(low=-10.0, high=10.0, size=(n_samples, dim))
    order = np.argsort(eigenvalues)[1:k]
    return eigenvectors[:, order]
