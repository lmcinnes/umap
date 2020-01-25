# Author: Leland McInnes <leland.mcinnes@gmail.com>
#
# License: BSD 3 clause
from __future__ import print_function

import numpy as np
import numba

from umap.utils import (
    tau_rand,
    make_heap,
    heap_push,
    unchecked_heap_push,
    smallest_flagged,
    rejection_sample,
    new_build_candidates,
    deheap_sort,
)
import umap.distances as dist

from umap.rp_tree import search_flat_tree


@numba.njit(fastmath=True)
def init_current_graph(data, dist, n_neighbors, rng_state):
    current_graph = make_heap(data.shape[0], n_neighbors)
    for i in range(data.shape[0]):
        indices = rejection_sample(n_neighbors, data.shape[0], rng_state)
        for j in range(indices.shape[0]):
            d = dist(data[i], data[indices[j]])
            heap_push(current_graph, i, d, indices[j], 1)
            heap_push(current_graph, indices[j], d, i, 1)
    return current_graph


@numba.njit(fastmath=True)
def init_rp_tree(data, dist, current_graph, leaf_array, tried=None):
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
                d = dist(data[p], data[q])
                heap_push(current_graph, p, d, q, 1)
                tried.add((p, q))
                if p != q:
                    heap_push(current_graph, q, d, p, 1)
                    tried.add((q, p))


@numba.njit(fastmath=True)
def nn_descent_internal_low_memory(
    current_graph,
    data,
    n_neighbors,
    rng_state,
    max_candidates=50,
    dist=dist.euclidean,
    n_iters=10,
    delta=0.001,
    rho=0.5,
    verbose=False,
):
    n_vertices = data.shape[0]

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

                    d = dist(data[p], data[q])
                    c += heap_push(current_graph, p, d, q, 1)
                    if p != q:
                        c += heap_push(current_graph, q, d, p, 1)

                for k in range(max_candidates):
                    q = int(old_candidate_neighbors[0, i, k])
                    if q < 0:
                        continue

                    d = dist(data[p], data[q])
                    c += heap_push(current_graph, p, d, q, 1)
                    if p != q:
                        c += heap_push(current_graph, q, d, p, 1)

        if c <= delta * n_neighbors * data.shape[0]:
            return


@numba.njit(fastmath=True)
def nn_descent_internal_high_memory(
    current_graph,
    data,
    n_neighbors,
    rng_state,
    tried,
    max_candidates=50,
    dist=dist.euclidean,
    n_iters=10,
    delta=0.001,
    rho=0.5,
    verbose=False,
):
    n_vertices = data.shape[0]

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

                    d = dist(data[p], data[q])
                    c += unchecked_heap_push(current_graph, p, d, q, 1)
                    tried.add((p, q))
                    if p != q:
                        c += unchecked_heap_push(current_graph, q, d, p, 1)
                        tried.add((q, p))

                for k in range(max_candidates):
                    q = int(old_candidate_neighbors[0, i, k])
                    if q < 0 or (p, q) in tried:
                        continue

                    d = dist(data[p], data[q])
                    c += unchecked_heap_push(current_graph, p, d, q, 1)
                    tried.add((p, q))
                    if p != q:
                        c += unchecked_heap_push(current_graph, q, d, p, 1)
                        tried.add((q, p))

        if c <= delta * n_neighbors * data.shape[0]:
            return


@numba.njit(fastmath=True)
def nn_descent(
    data,
    n_neighbors,
    rng_state,
    max_candidates=50,
    dist=dist.euclidean,
    n_iters=10,
    delta=0.001,
    rho=0.5,
    rp_tree_init=True,
    leaf_array=None,
    low_memory=False,
    verbose=False,
):
    tried = set([(-1, -1)])

    current_graph = make_heap(data.shape[0], n_neighbors)
    for i in range(data.shape[0]):
        indices = rejection_sample(n_neighbors, data.shape[0], rng_state)
        for j in range(indices.shape[0]):
            d = dist(data[i], data[indices[j]])
            heap_push(current_graph, i, d, indices[j], 1)
            heap_push(current_graph, indices[j], d, i, 1)
            tried.add((i, indices[j]))
            tried.add((indices[j], i))

    if rp_tree_init:
        init_rp_tree(data, dist, current_graph, leaf_array, tried=tried)

    if low_memory:
        nn_descent_internal_low_memory(
            current_graph,
            data,
            n_neighbors,
            rng_state,
            max_candidates=max_candidates,
            dist=dist,
            n_iters=n_iters,
            delta=delta,
            rho=rho,
            verbose=verbose,
        )
    else:
        nn_descent_internal_high_memory(
            current_graph,
            data,
            n_neighbors,
            rng_state,
            tried,
            max_candidates=max_candidates,
            dist=dist,
            n_iters=n_iters,
            delta=delta,
            rho=rho,
            verbose=verbose,
        )

    return deheap_sort(current_graph)


@numba.njit()
def init_from_random(n_neighbors, data, query_points, heap, rng_state, dist):
    for i in range(query_points.shape[0]):
        indices = rejection_sample(n_neighbors, data.shape[0], rng_state)
        for j in range(indices.shape[0]):
            if indices[j] < 0:
                continue
            d = dist(data[indices[j]], query_points[i])
            heap_push(heap, i, d, indices[j], 1)
    return


@numba.njit()
def init_from_tree(tree, data, query_points, heap, rng_state, dist):
    for i in range(query_points.shape[0]):
        indices = search_flat_tree(
            query_points[i],
            tree.hyperplanes,
            tree.offsets,
            tree.children,
            tree.indices,
            rng_state,
        )

        for j in range(indices.shape[0]):
            if indices[j] < 0:
                continue
            d = dist(data[indices[j]], query_points[i])
            heap_push(heap, i, d, indices[j], 1)

    return


def initialise_search(forest, data, query_points, n_neighbors, rng_state, dist):
    results = make_heap(query_points.shape[0], n_neighbors)
    init_from_random(n_neighbors, data, query_points, results, rng_state, dist)
    if forest is not None:
        for tree in forest:
            init_from_tree(tree, data, query_points, results, rng_state, dist)

    return results


@numba.njit(parallel=True)
def initialized_nnd_search(data, indptr, indices, initialization, query_points, dist):
    for i in numba.prange(query_points.shape[0]):

        tried = set(initialization[0, i])

        while True:

            # Find smallest flagged vertex
            vertex = smallest_flagged(initialization, i)

            if vertex == -1:
                break
            candidates = indices[indptr[vertex] : indptr[vertex + 1]]
            for j in range(candidates.shape[0]):
                if (
                    candidates[j] == vertex
                    or candidates[j] == -1
                    or candidates[j] in tried
                ):
                    continue
                d = dist(data[candidates[j]], query_points[i])
                unchecked_heap_push(initialization, i, d, candidates[j], 1)
                tried.add(candidates[j])

    return initialization
