import pytest
from umap import AlignedUMAP
from sklearn.metrics import pairwise_distances
from sklearn.cluster import KMeans
import numpy as np
from sklearn.metrics import adjusted_rand_score

# ===============================
# Test AlignedUMAP on sliced iris
# ===============================


def nn_accuracy(true_nn, embd_nn):
    num_correct = 0.0
    for i in range(true_nn.shape[0]):
        num_correct += np.sum(np.in1d(true_nn[i], embd_nn[i]))
    return num_correct / true_nn.size


def test_neighbor_local_neighbor_accuracy(aligned_iris, aligned_iris_model):
    data, target = aligned_iris
    for i, slice in enumerate(data):
        data_dmat = pairwise_distances(slice)
        true_nn = np.argsort(data_dmat, axis=1)[:, :10]
        embd_dmat = pairwise_distances(aligned_iris_model.embeddings_[i])
        embd_nn = np.argsort(embd_dmat, axis=1)[:, :10]
        assert nn_accuracy(true_nn, embd_nn) >= 0.65


def test_local_clustering(aligned_iris, aligned_iris_model):
    data, target = aligned_iris

    embd = aligned_iris_model.embeddings_[1]
    clusters = KMeans(n_clusters=2).fit_predict(embd)
    ari = adjusted_rand_score(target[1], clusters)
    assert ari >= 0.75

    embd = aligned_iris_model.embeddings_[3]
    clusters = KMeans(n_clusters=2).fit_predict(embd)
    ari = adjusted_rand_score(target[3], clusters)
    assert ari >= 0.40


def test_aligned_update(aligned_iris, aligned_iris_relations):
    data, target = aligned_iris
    small_aligned_model = AlignedUMAP()
    small_aligned_model.fit(data[:3], relations=aligned_iris_relations[:2])
    small_aligned_model.update(data[3], relations=aligned_iris_relations[2])
    for i, slice in enumerate(data[:4]):
        data_dmat = pairwise_distances(slice)
        true_nn = np.argsort(data_dmat, axis=1)[:, :10]
        embd_dmat = pairwise_distances(small_aligned_model.embeddings_[i])
        embd_nn = np.argsort(embd_dmat, axis=1)[:, :10]
        assert nn_accuracy(true_nn, embd_nn) >= 0.45


def test_aligned_update_params(aligned_iris, aligned_iris_relations):
    data, target = aligned_iris
    n_neighbors = [15, 15, 15, 15, 15]
    small_aligned_model = AlignedUMAP(n_neighbors=n_neighbors[:3])
    small_aligned_model.fit(data[:3], relations=aligned_iris_relations[:2])
    small_aligned_model.update(data[3],
                               relations=aligned_iris_relations[2],
                               n_neighbors=n_neighbors[3])
    for i, slice in enumerate(data[:4]):
        data_dmat = pairwise_distances(slice)
        true_nn = np.argsort(data_dmat, axis=1)[:, :10]
        embd_dmat = pairwise_distances(small_aligned_model.embeddings_[i])
        embd_nn = np.argsort(embd_dmat, axis=1)[:, :10]
        assert nn_accuracy(true_nn, embd_nn) >= 0.45

@pytest.mark.skip(reason="Temporarily disable")
def test_aligned_update_array_error(aligned_iris, aligned_iris_relations):
    data, target = aligned_iris
    n_neighbors = [15, 15, 15, 15, 15]
    small_aligned_model = AlignedUMAP(n_neighbors=n_neighbors[:3])
    small_aligned_model.fit(data[:3], relations=aligned_iris_relations[:2])

    with pytest.raises(ValueError):
        small_aligned_model.update(data[3:],
                                   relations=aligned_iris_relations[2:],
                                   n_neighbors=n_neighbors[3:])
