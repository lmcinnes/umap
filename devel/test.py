from umap import UMAP
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import pairwise_distances

dense = np.array(
    [
        [1, 0],
        [0, 1],
        [0, 0],
        [-1, 0],
        [0, -1],
    ]
)

colors = ['red', 'green', 'blue', 'pink', 'yellow']


n_plots = 20
n_plots += 1

figsize = plt.rcParams.get("figure.figsize").copy()
figsize = (figsize[0] * 1, figsize[1] * n_plots)

fig, axes = plt.subplots(n_plots, 1, figsize=figsize)
axes[0].scatter(dense[:, 0], dense[:, 1], c=colors)
axes[0].axis('equal')

print(pairwise_distances(dense)[2].sum())
dists = []
for i in range(1, n_plots):
    embedding = UMAP(n_neighbors=3).fit_transform(dense)
    axes[i].scatter(embedding[:, 0], embedding[:, 1], c=colors)
    axes[i].set_xticks([])
    axes[i].set_yticks([])
    axes[i].axis('equal')
    dists.append(pairwise_distances(embedding)[2].sum())

plt.tight_layout()
plt.savefig('devel/original.png', dpi=300)

print(np.array(dists).mean())