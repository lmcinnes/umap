# Author: Leland McInnes <leland.mcinnes@gmail.com>
# Enough simple sparse operations in numba to enable sparse UMAP
#
# License: BSD 3 clause
from __future__ import print_function

import locale

import numba
import numpy as np
import umap.sparse

from umap.utils import (
    tau_rand,
    norm,
    make_heap,
    heap_push,
    unchecked_heap_push,
    smallest_flagged,
    rejection_sample,
    new_build_candidates,
    deheap_sort,
)

from umap.rp_tree import search_sparse_flat_tree

locale.setlocale(locale.LC_NUMERIC, "C")


@numba.njit(fastmath=True)
def sparse_init_rp_tree(
    inds, indptr, data, sparse_dist, current_graph, leaf_array, tried=None
):
    if tried is None:
        tried = set([(-1, -1)])

    for n in range(leaf_array.shape[0]):
        for i in range(leaf_array.shape[1]):
            p = leaf_array[n, i]
            if p < 0:
                break
            for j in range(i + 1, leaf_array.shape[1]):
                q = leaf_array[n, j]
                if q < 0:
                    break
                if (p, q) in tried:
                    continue

                from_inds = inds[indptr[p] : indptr[p + 1]]
                from_data = data[indptr[p] : indptr[p + 1]]

                to_inds = inds[indptr[q] : indptr[q + 1]]
                to_data = data[indptr[q] : indptr[q + 1]]
                d = sparse_dist(from_inds, from_data, to_inds, to_data)
                heap_push(current_graph, p, d, q, 1)
                tried.add((p, q))
                if p != q:
                    heap_push(current_graph, q, d, p, 1)
                    tried.add((q, p))


@numba.njit(fastmath=True)
def sparse_nn_descent_internal_low_memory(
    current_graph,
    inds,
    indptr,
    data,
    n_vertices,
    n_neighbors,
    rng_state,
    max_candidates=50,
    sparse_dist=umap.sparse.sparse_euclidean,
    n_iters=10,
    delta=0.001,
    rho=0.5,
    verbose=False,
):
    for n in range(n_iters):
        if verbose:
            print("\t", n, " / ", n_iters)

        (new_candidate_neighbors, old_candidate_neighbors) = new_build_candidates(
            current_graph, n_vertices, n_neighbors, max_candidates, rng_state, rho
        )

        c = 0
        for i in range(n_vertices):
            for j in range(max_candidates):
                p = int(new_candidate_neighbors[0, i, j])
                if p < 0:
                    continue
                for k in range(j, max_candidates):
                    q = int(new_candidate_neighbors[0, i, k])
                    if q < 0:
                        continue

                    from_inds = inds[indptr[p] : indptr[p + 1]]
                    from_data = data[indptr[p] : indptr[p + 1]]

                    to_inds = inds[indptr[q] : indptr[q + 1]]
                    to_data = data[indptr[q] : indptr[q + 1]]

                    d = sparse_dist(from_inds, from_data, to_inds, to_data)

                    c += heap_push(current_graph, p, d, q, 1)
                    if p != q:
                        c += heap_push(current_graph, q, d, p, 1)

                for k in range(max_candidates):
                    q = int(old_candidate_neighbors[0, i, k])
                    if q < 0:
                        continue

                    from_inds = inds[indptr[p] : indptr[p + 1]]
                    from_data = data[indptr[p] : indptr[p + 1]]

                    to_inds = inds[indptr[q] : indptr[q + 1]]
                    to_data = data[indptr[q] : indptr[q + 1]]

                    d = sparse_dist(from_inds, from_data, to_inds, to_data)

                    c += heap_push(current_graph, p, d, q, 1)
                    if p != q:
                        c += heap_push(current_graph, q, d, p, 1)

        if c <= delta * n_neighbors * n_vertices:
            return


@numba.njit(fastmath=True)
def sparse_nn_descent_internal_high_memory(
    current_graph,
    inds,
    indptr,
    data,
    n_vertices,
    n_neighbors,
    rng_state,
    tried,
    max_candidates=50,
    sparse_dist=umap.sparse.sparse_euclidean,
    n_iters=10,
    delta=0.001,
    rho=0.5,
    verbose=False,
):
    for n in range(n_iters):
        if verbose:
            print("\t", n, " / ", n_iters)

        (new_candidate_neighbors, old_candidate_neighbors) = new_build_candidates(
            current_graph, n_vertices, n_neighbors, max_candidates, rng_state, rho
        )

        c = 0
        for i in range(n_vertices):
            for j in range(max_candidates):
                p = int(new_candidate_neighbors[0, i, j])
                if p < 0:
                    continue
                for k in range(j, max_candidates):
                    q = int(new_candidate_neighbors[0, i, k])
                    if q < 0 or (p, q) in tried:
                        continue

                    from_inds = inds[indptr[p] : indptr[p + 1]]
                    from_data = data[indptr[p] : indptr[p + 1]]

                    to_inds = inds[indptr[q] : indptr[q + 1]]
                    to_data = data[indptr[q] : indptr[q + 1]]

                    d = sparse_dist(from_inds, from_data, to_inds, to_data)

                    c += unchecked_heap_push(current_graph, p, d, q, 1)
                    tried.add((p, q))
                    if p != q:
                        c += unchecked_heap_push(current_graph, q, d, p, 1)
                        tried.add((q, p))

                for k in range(max_candidates):
                    q = int(old_candidate_neighbors[0, i, k])
                    if q < 0 or (p, q) in tried:
                        continue

                    from_inds = inds[indptr[p] : indptr[p + 1]]
                    from_data = data[indptr[p] : indptr[p + 1]]

                    to_inds = inds[indptr[q] : indptr[q + 1]]
                    to_data = data[indptr[q] : indptr[q + 1]]

                    d = sparse_dist(from_inds, from_data, to_inds, to_data)

                    c += unchecked_heap_push(current_graph, p, d, q, 1)
                    tried.add((p, q))
                    if p != q:
                        c += unchecked_heap_push(current_graph, q, d, p, 1)
                        tried.add((q, p))

        if c <= delta * n_neighbors * n_vertices:
            return


@numba.njit(fastmath=True)
def sparse_nn_descent(
    inds,
    indptr,
    data,
    n_vertices,
    n_neighbors,
    rng_state,
    max_candidates=50,
    sparse_dist=umap.sparse.sparse_euclidean,
    n_iters=10,
    delta=0.001,
    rho=0.5,
    low_memory=False,
    rp_tree_init=True,
    leaf_array=None,
    verbose=False,
):

    tried = set([(-1, -1)])

    current_graph = make_heap(n_vertices, n_neighbors)
    for i in range(n_vertices):
        indices = rejection_sample(n_neighbors, n_vertices, rng_state)
        for j in range(indices.shape[0]):

            from_inds = inds[indptr[i] : indptr[i + 1]]
            from_data = data[indptr[i] : indptr[i + 1]]

            to_inds = inds[indptr[indices[j]] : indptr[indices[j] + 1]]
            to_data = data[indptr[indices[j]] : indptr[indices[j] + 1]]

            d = sparse_dist(from_inds, from_data, to_inds, to_data)

            heap_push(current_graph, i, d, indices[j], 1)
            heap_push(current_graph, indices[j], d, i, 1)
            tried.add((i, indices[j]))
            tried.add((indices[j], i))

    if rp_tree_init:
        sparse_init_rp_tree(
            inds, indptr, data, sparse_dist, current_graph, leaf_array, tried=tried,
        )

    if low_memory:
        sparse_nn_descent_internal_low_memory(
            current_graph,
            inds,
            indptr,
            data,
            n_vertices,
            n_neighbors,
            rng_state,
            max_candidates=max_candidates,
            sparse_dist=sparse_dist,
            n_iters=n_iters,
            delta=delta,
            rho=rho,
            verbose=verbose,
        )
    else:
        sparse_nn_descent_internal_high_memory(
            current_graph,
            inds,
            indptr,
            data,
            n_vertices,
            n_neighbors,
            rng_state,
            tried,
            max_candidates=max_candidates,
            sparse_dist=sparse_dist,
            n_iters=n_iters,
            delta=delta,
            rho=rho,
            verbose=verbose,
        )

    return deheap_sort(current_graph)


@numba.njit()
def sparse_init_from_random(
    n_neighbors,
    inds,
    indptr,
    data,
    query_inds,
    query_indptr,
    query_data,
    heap,
    rng_state,
    sparse_dist,
):
    for i in range(query_indptr.shape[0] - 1):
        indices = rejection_sample(n_neighbors, indptr.shape[0] - 1, rng_state)

        to_inds = query_inds[query_indptr[i] : query_indptr[i + 1]]
        to_data = query_data[query_indptr[i] : query_indptr[i + 1]]

        for j in range(indices.shape[0]):
            if indices[j] < 0:
                continue

            from_inds = inds[indptr[indices[j]] : indptr[indices[j] + 1]]
            from_data = data[indptr[indices[j]] : indptr[indices[j] + 1]]

            d = sparse_dist(from_inds, from_data, to_inds, to_data)
            heap_push(heap, i, d, indices[j], 1)
    return


@numba.njit()
def sparse_init_from_tree(
    tree,
    inds,
    indptr,
    data,
    query_inds,
    query_indptr,
    query_data,
    heap,
    rng_state,
    sparse_dist,
):
    for i in range(query_indptr.shape[0] - 1):

        to_inds = query_inds[query_indptr[i] : query_indptr[i + 1]]
        to_data = query_data[query_indptr[i] : query_indptr[i + 1]]

        indices = search_sparse_flat_tree(
            to_inds,
            to_data,
            tree.hyperplanes,
            tree.offsets,
            tree.children,
            tree.indices,
            rng_state,
        )

        for j in range(indices.shape[0]):
            if indices[j] < 0:
                continue
            from_inds = inds[indptr[indices[j]] : indptr[indices[j] + 1]]
            from_data = data[indptr[indices[j]] : indptr[indices[j] + 1]]

            d = sparse_dist(from_inds, from_data, to_inds, to_data)
            heap_push(heap, i, d, indices[j], 1)

    return


def sparse_initialise_search(
    forest,
    inds,
    indptr,
    data,
    query_inds,
    query_indptr,
    query_data,
    n_neighbors,
    rng_state,
    sparse_dist,
):
    results = make_heap(query_indptr.shape[0] - 1, n_neighbors)
    sparse_init_from_random(
        n_neighbors,
        inds,
        indptr,
        data,
        query_inds,
        query_indptr,
        query_data,
        results,
        rng_state,
        sparse_dist,
    )
    if forest is not None:
        for tree in forest:
            sparse_init_from_tree(
                tree,
                inds,
                indptr,
                data,
                query_inds,
                query_indptr,
                query_data,
                results,
                rng_state,
                sparse_dist,
            )

    return results


@numba.njit(parallel=True)
def sparse_initialized_nnd_search(
    inds,
    indptr,
    data,
    search_indptr,
    search_inds,
    initialization,
    query_inds,
    query_indptr,
    query_data,
    sparse_dist,
):
    for i in numba.prange(query_indptr.shape[0] - 1):

        tried = set(initialization[0, i])

        to_inds = query_inds[query_indptr[i] : query_indptr[i + 1]]
        to_data = query_data[query_indptr[i] : query_indptr[i + 1]]

        while True:

            # Find smallest flagged vertex
            vertex = smallest_flagged(initialization, i)

            if vertex == -1:
                break
            candidates = search_inds[search_indptr[vertex] : search_indptr[vertex + 1]]

            for j in range(candidates.shape[0]):
                if (
                    candidates[j] == vertex
                    or candidates[j] == -1
                    or candidates[j] in tried
                ):
                    continue

                from_inds = inds[indptr[candidates[j]] : indptr[candidates[j] + 1]]
                from_data = data[indptr[candidates[j]] : indptr[candidates[j] + 1]]

                d = sparse_dist(from_inds, from_data, to_inds, to_data)
                unchecked_heap_push(initialization, i, d, candidates[j], 1)
                tried.add(candidates[j])

    return initialization
