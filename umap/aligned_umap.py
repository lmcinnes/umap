import numpy as np
import numba
from sklearn.base import BaseEstimator
from sklearn.utils import check_random_state

from umap.sparse import arr_intersect as intersect1d
from umap.sparse import arr_union as union1d
from umap.umap_ import UMAP, make_epochs_per_sample
from umap.spectral import spectral_layout
from umap.layouts import optimize_layout_aligned_euclidean

INT32_MIN = np.iinfo(np.int32).min + 1
INT32_MAX = np.iinfo(np.int32).max - 1


@numba.njit(parallel=True)
def in1d(arr, test_set):
    test_set = set(test_set)
    result = np.empty(arr.shape[0], dtype=np.uint8)
    for i in numba.prange(arr.shape[0]):
        if arr[i] in test_set:
            result[i] = True
        else:
            result[i] = False

    return result


def invert_dict(d):
    return {value: key for key, value in d.items()}


@numba.njit()
def procrustes_align(embedding_base, embedding_to_align, anchors):
    subset1 = embedding_base[anchors[0]]
    subset2 = embedding_to_align[anchors[1]]
    M = subset2.T @ subset1
    U, S, V = np.linalg.svd(M)
    R = U @ V
    return embedding_to_align @ R


def expand_relations(relation_dicts, window_size=3):
    max_n_samples = (
            max(
                [max(d.keys()) for d in relation_dicts]
                + [max(d.values()) for d in relation_dicts]
            )
            + 1
    )
    result = np.full(
        (len(relation_dicts), 2 * window_size + 1, max_n_samples), -1, dtype=np.int32
    )
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
                        [relation_dicts[i + j].get(n, -1) for n in mapping]
                    )
                result[i, result_index] = mapping

        for j in range(0, -window_size, -1):
            result_index = (window_size) + (j - 1)
            if i + j - 1 < 0:
                result[i, result_index] = np.full(max_n_samples, -1, dtype=np.int32)
            else:
                mapping = np.arange(max_n_samples)
                for k in range(np.abs(j) + 1):
                    mapping = np.array(
                        [reverse_relation_dicts[i + j - 1].get(n, -1) for n in mapping]
                    )
                result[i, result_index] = mapping

    return result


@numba.njit()
def build_neighborhood_similarities(graphs_indptr, graphs_indices, relations):
    result = np.zeros(relations.shape, dtype=np.float32)
    center_index = (relations.shape[1] - 1) // 2
    for i in range(relations.shape[0]):
        base_graph_indptr = graphs_indptr[i]
        base_graph_indices = graphs_indices[i]
        for j in range(relations.shape[1]):
            if i + j - center_index < 0 or i + j - center_index >= len(graphs_indptr):
                continue

            comparison_graph_indptr = graphs_indptr[i + j - center_index]
            comparison_graph_indices = graphs_indices[i + j - center_index]
            for k in range(relations.shape[2]):
                comparison_index = relations[i, j, k]
                if comparison_index < 0:
                    continue

                raw_base_graph_indices = base_graph_indices[
                                         base_graph_indptr[k]: base_graph_indptr[k + 1]
                                         ].copy()
                base_indices = relations[i, j][raw_base_graph_indices]
                base_indices = base_indices[base_indices >= 0]
                comparison_indices = comparison_graph_indices[
                                     comparison_graph_indptr[comparison_index]:
                                     comparison_graph_indptr[
                                         comparison_index + 1
                                         ]
                                     ]
                comparison_indices = comparison_indices[
                    in1d(comparison_indices, relations[i, j])
                ]

                intersection_size = intersect1d(base_indices, comparison_indices).shape[
                    0
                ]
                union_size = union1d(base_indices, comparison_indices).shape[0]

                if union_size > 0:
                    result[i, j, k] = intersection_size / union_size
                else:
                    result[i, j, k] = 1.0

    return result


def get_nth_item_or_val(iterable_or_val, n):
    if type(iterable_or_val) in (list, tuple, np.ndarray):
        return iterable_or_val[n]
    elif type(iterable_or_val) in (int, float, bool, None):
        return iterable_or_val
    else:
        raise ValueError("Unrecognized parameter type")


class AlignedUMAP(BaseEstimator):
    def __init__(
            self,
            n_neighbors=15,
            n_components=2,
            metric="euclidean",
            metric_kwds=None,
            n_epochs=None,
            learning_rate=1.0,
            init="spectral",
            alignment_regularisation=1.0e-2,
            min_dist=0.1,
            spread=1.0,
            low_memory=False,
            set_op_mix_ratio=1.0,
            local_connectivity=1.0,
            repulsion_strength=1.0,
            negative_sample_rate=5,
            transform_queue_size=4.0,
            a=None,
            b=None,
            random_state=None,
            angular_rp_forest=False,
            target_n_neighbors=-1,
            target_metric="categorical",
            target_metric_kwds=None,
            target_weight=0.5,
            transform_seed=42,
            force_approximation_algorithm=False,
            verbose=False,
            unique=False,
    ):

        self.n_neighbors = n_neighbors
        self.metric = metric
        self.metric_kwds = metric_kwds

        self.n_epochs = n_epochs
        self.init = init
        self.n_components = n_components
        self.repulsion_strength = repulsion_strength
        self.learning_rate = learning_rate
        self.alignment_regularisation = alignment_regularisation

        self.spread = spread
        self.min_dist = min_dist
        self.low_memory = low_memory
        self.set_op_mix_ratio = set_op_mix_ratio
        self.local_connectivity = local_connectivity
        self.negative_sample_rate = negative_sample_rate
        self.random_state = random_state
        self.angular_rp_forest = angular_rp_forest
        self.transform_queue_size = transform_queue_size
        self.target_n_neighbors = target_n_neighbors
        self.target_metric = target_metric
        self.target_metric_kwds = target_metric_kwds
        self.target_weight = target_weight
        self.transform_seed = transform_seed
        self.force_approximation_algorithm = force_approximation_algorithm
        self.verbose = verbose
        self.unique = unique

        self.a = a
        self.b = b

    def fit(self, X, y=None, **fit_params):
        if "relations" not in fit_params:
            raise ValueError(
                "Aligned UMAP requires relations between data to be " "specified"
            )

        dict_relations = fit_params["relations"]
        assert type(dict_relations) in (list, tuple)
        assert type(X) in (list, tuple, np.ndarray)
        assert len(X) == (len(dict_relations) - 1)

        mappers = [
            UMAP(
                n_neighbors=get_nth_item_or_val(self.n_neighbors, n),
                min_dist=get_nth_item_or_val(self.min_dist, n),
                n_epochs=get_nth_item_or_val(self.n_epochs, n),
                repulsion_strength=get_nth_item_or_val(self.repulsion_strength, n),
                learning_rate=get_nth_item_or_val(self.learning_rate, n),
                spread=get_nth_item_or_val(self.spread, n),
                negative_sample_rate=get_nth_item_or_val(self.negative_sample_rate, n),
                local_connectivity=get_nth_item_or_val(self.local_connectivity, n),
                set_op_mix_ratio=get_nth_item_or_val(self.set_op_mix_ratio, n),
                unique=get_nth_item_or_val(self.unique, n),
            ).fit(X[n])
            for n in range(len(X))
        ]

        window_size = fit_params.get("window_size", 3)
        relations = expand_relations(dict_relations, window_size)
        regularisation_weights = build_neighborhood_similarities(
            [mapper.graph_.indptr for mapper in mappers],
            [mapper.graph_.indices for mapper in mappers],
            relations,
        )
        first_init = spectral_layout(
            mappers[0]._raw_data, mappers[0].graph_, 2, np.random,
        )
        expansion = 10.0 / np.abs(first_init).max()
        first_embedding = (first_init * expansion).astype(np.float32, order="C", )

        embeddings = numba.typed.List.empty_list(numba.types.float32[:, ::1])
        embeddings.append(first_embedding)
        for i in range(1, len(mappers)):
            next_init = spectral_layout(
                mappers[i]._raw_data, mappers[i].graph_, 2, np.random,
            )
            expansion = 10.0 / np.abs(next_init).max()
            next_embedding = (next_init * expansion).astype(np.float32, order="C", )
            anchor_data = relations[i][window_size - 1]
            left_anchors = anchor_data[anchor_data >= 0]
            right_anchors = np.where(anchor_data >= 0)[0]
            embeddings.append(
                procrustes_align(
                    embeddings[-1],
                    next_embedding,
                    np.vstack([left_anchors, right_anchors]),
                )
            )

        heads = numba.typed.List.empty_list(numba.types.int32[::1]).extend(
            [mapper.graph_.tocoo().row for mapper in mappers]
        )
        tails = numba.typed.List.empty_list(numba.types.int32[::1]).extend(
            [mapper.graph_.tocoo().col for mapper in mappers]
        )
        epochs_per_samples = numba.typed.List.empty_list(
            numba.types.float64[::1]
        ).extend(
            [
                make_epochs_per_sample(mapper.graph_.tocoo().data, self.n_epochs)
                for mapper in mappers
            ]
        )

        random_state = check_random_state(self.random_state)
        rng_state = random_state.randint(INT32_MIN, INT32_MAX, 3).astype(np.int64)

        self.embeddings_ = optimize_layout_aligned_euclidean(
            embeddings,
            embeddings,
            heads,
            tails,
            self.n_epochs,
            epochs_per_samples,
            regularisation_weights,
            relations,
            rng_state,
            lambda_=self.alignment_regularisation,
        )

    def update(self, X, y=None, **fit_params):
        pass
