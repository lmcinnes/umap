#!/usr/bin/env python

from sklearn.decomposition import PCA
import umap
from sklearn.datasets import fetch_openml
import matplotlib.pyplot as plt
import seaborn as sns

# install hnswlib with pip install hnswlib
# or update to use any other index (e.g., nndescent)
import hnswlib

sns.set(context="paper", style="white")

mnist = fetch_openml("mnist_784", version=1)

sample_size = 70000

# create a knn index
knn_index = hnswlib.Index(space='l2', dim=mnist.data.shape[1])
knn_index.init_index(max_elements=sample_size, ef_construction=100, M=16)
knn_index.add_items(mnist.data.values[:sample_size])

knn_indices, knn_dists = knn_index.knn_query(mnist.data.values[:sample_size], k=15)


pca_init = PCA(n_components=2).fit_transform(mnist.data.values[:sample_size])

reducer = umap.UMAP(random_state=42, n_neighbors=15, 
                    precomputed_knn=(knn_indices, knn_dists), 
                    verbose=True, n_jobs=-1, init=pca_init, metric="euclidean")
embedding = reducer.fit_transform_index()

fig, ax = plt.subplots(figsize=(12, 10))
color = mnist.target[:sample_size].astype(int)
plt.scatter(embedding[:, 0], embedding[:, 1], c=color, cmap="Spectral", s=0.1)
plt.setp(ax, xticks=[], yticks=[])
plt.title("MNIST data embedded into two dimensions by UMAP", fontsize=18)

plt.show()