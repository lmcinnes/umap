import numpy as np
import numba


def invert_dict(d):
    return {value: key for key, value in d.items()}


def procrustes_align(embedding_base, embedding_to_align, anchors):
    subset1 = embedding_base[anchors[0]]
    subset2 = embedding_to_align[anchors[1]]
    M = subset2.T @ subset1
    U, S, V = np.linalg.svd(M)
    R = U @ V
    return embedding_to_align @ R


def expand_relations(relation_dicts, window_size=3):
    max_n_samples = max(
        [max(d.keys()) for d in relation_dicts] + [max(d.values()) for d in
                                                   relation_dicts]) + 1
    result = np.full((len(relation_dicts), 2 * window_size + 1, max_n_samples), -1,
                     dtype=np.int32)
    reverse_relation_dicts = [invert_dict(d) for d in relation_dicts]
    for i in range(result.shape[0]):
        for j in range(window_size):
            result_index = (window_size) + (j + 1)
            if i + j + 1 >= len(relation_dicts):
                result[i, result_index] = np.full(max_n_samples, -1, dtype=np.int32)
            else:
                mapping = np.arange(max_n_samples)
                for k in range(j + 1):
                    mapping = np.array(
                        [relation_dicts[i + j].get(n, -1) for n in mapping])
                result[i, result_index] = mapping

        for j in range(0, -window_size, -1):
            result_index = (window_size) + (j - 1)
            if i + j - 1 < 0:
                result[i, result_index] = np.full(max_n_samples, -1, dtype=np.int32)
            else:
                mapping = np.arange(max_n_samples)
                for k in range(np.abs(j) + 1):
                    mapping = np.array(
                        [reverse_relation_dicts[i + j - 1].get(n, -1) for n in mapping])
                result[i, result_index] = mapping

    return result


def build_neighborhood_similarities(graphs, relations):
    result = np.zeros(relations.shape, dtype=np.float32)
    center_index = ((relations.shape[1] - 1) // 2)
    for i in range(relations.shape[0]):
        base_graph = graphs[i]
        for j in range(relations.shape[1]):
            if i + j - center_index < 0 or i + j - center_index >= len(graphs):
                continue

            comparison_graph = graphs[i + j - center_index]
            for k in range(relations.shape[2]):
                comparison_index = relations[i, j, k]
                if comparison_index < 0:
                    continue

                base_indices = relations[i, j, base_graph[k].indices]
                base_indices = base_indices[base_indices >= 0]
                comparison_indices = comparison_graph[comparison_index].indices
                comparison_indices = comparison_indices[
                    np.in1d(comparison_indices, relations[i, j])]

                intersection_size = \
                    np.intersect1d(base_indices, comparison_indices).shape[0]
                union_size = np.union1d(base_indices, comparison_indices).shape[0]

                if union_size > 0:
                    result[i, j, k] = intersection_size / union_size
                else:
                    result[i, j, k] = 1.0

    return result
