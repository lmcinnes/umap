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

from umap.rp_tree import search_flat_tree


@numba.njit()
def nn_descent(
        data,
        n_neighbors,
        rng_state,
        dist,
        dist_args=(),
        max_candidates=50,
        n_iters=10,
        delta=0.001,
        rho=0.5,
        rp_tree_init=True,
        leaf_array=None,
        verbose=False,
):
    n_vertices = data.shape[0]
    tried = set([(-1, -1)])

    current_graph = make_heap(data.shape[0], n_neighbors)
    for i in range(data.shape[0]):
        indices = rejection_sample(n_neighbors, data.shape[0], rng_state)
        for j in range(indices.shape[0]):
            d = dist(data[i], data[indices[j]], *dist_args)
            heap_push(current_graph, i, d, indices[j], 1)
            heap_push(current_graph, indices[j], d, i, 1)
            tried.add((i, indices[j]))
            tried.add((indices[j], i))

    if rp_tree_init:
        for n in range(leaf_array.shape[0]):
            for i in range(leaf_array.shape[1]):
                if leaf_array[n, i] < 0:
                    break
                for j in range(i + 1, leaf_array.shape[1]):
                    if leaf_array[n, j] < 0:
                        break
                    if (leaf_array[n, i], leaf_array[n, j]) in tried:
                        continue
                    d = dist(
                        data[leaf_array[n, i]], data[leaf_array[n, j]], *dist_args
                    )
                    unchecked_heap_push(
                        current_graph, leaf_array[n, i], d, leaf_array[n, j], 1
                    )
                    unchecked_heap_push(
                        current_graph, leaf_array[n, j], d, leaf_array[n, i], 1
                    )
                    tried.add((leaf_array[n, i], leaf_array[n, j]))
                    tried.add((leaf_array[n, j], leaf_array[n, i]))


    for n in range(n_iters):

        (new_candidate_neighbors,
         old_candidate_neighbors) = new_build_candidates(current_graph,
                                                         n_vertices,
                                                         n_neighbors,
                                                         max_candidates,
                                                         rng_state, rho)

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

                    d = dist(data[p], data[q], *dist_args)
                    c += unchecked_heap_push(current_graph, p, d, q, 1)
                    c += unchecked_heap_push(current_graph, q, d, p, 1)
                    tried.add((p, q))
                    tried.add((q, p))

                for k in range(max_candidates):
                    q = int(old_candidate_neighbors[0, i, k])
                    if q < 0 or (p, q) in tried:
                        continue

                    d = dist(data[p], data[q], *dist_args)
                    c += unchecked_heap_push(current_graph, p, d, q, 1)
                    c += unchecked_heap_push(current_graph, q, d, p, 1)
                    tried.add((p, q))
                    tried.add((q, p))

        if c <= delta * n_neighbors * data.shape[0]:
            break

    return deheap_sort(current_graph)


@numba.njit(parallel=True)
def init_from_random(n_neighbors, data, query_points, heap, rng_state, dist, dist_args):
    for i in range(query_points.shape[0]):
        indices = rejection_sample(n_neighbors, data.shape[0], rng_state)
        for j in range(indices.shape[0]):
            if indices[j] < 0:
                continue
            d = dist(data[indices[j]], query_points[i], *dist_args)
            heap_push(heap, i, d, indices[j], 1)
    return


@numba.njit(parallel=True)
def init_from_tree(tree, data, query_points, heap, rng_state, dist, dist_args):
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
            d = dist(data[indices[j]], query_points[i], *dist_args)
            heap_push(heap, i, d, indices[j], 1)

    return


def initialise_search(
        forest, data, query_points, n_neighbors, rng_state, dist, dist_args
):
    results = make_heap(query_points.shape[0], n_neighbors)
    init_from_random(n_neighbors, data, query_points, results, rng_state, dist, dist_args)
    if forest is not None:
        for tree in forest:
            init_from_tree(tree, data, query_points, results, rng_state, dist, dist_args)

    return results


@numba.njit(parallel=True)
def initialized_nnd_search(data, indptr, indices, initialization, query_points, dist, dist_args):
    for i in numba.prange(query_points.shape[0]):

        tried = set(initialization[0, i])

        while True:

            # Find smallest flagged vertex
            vertex = smallest_flagged(initialization, i)

            if vertex == -1:
                break
            candidates = indices[indptr[vertex]: indptr[vertex + 1]]
            for j in range(candidates.shape[0]):
                if (
                        candidates[j] == vertex
                        or candidates[j] == -1
                        or candidates[j] in tried
                ):
                    continue
                d = dist(data[candidates[j]], query_points[i], *dist_args)
                unchecked_heap_push(initialization, i, d, candidates[j], 1)
                tried.add(candidates[j])

    return initialization
