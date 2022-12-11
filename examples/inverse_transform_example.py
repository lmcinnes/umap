#!/usr/bin/env python

import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import fetch_openml

import umap

mnist = fetch_openml("Fashion-MNIST", version=1)


trans = umap.UMAP(
    n_neighbors=10,
    random_state=42,
    metric="euclidean",
    output_metric="euclidean",
    init="spectral",
    verbose=True,
).fit(mnist.data)

corners = np.array(
    [
        [-5.1, 2.9],
        [-1.9, 6.4],
        [-5.4, -6.3],
        [8.3, 4.0],
    ]  # 7  # 4  # 1  # 0
)

test_pts = np.array(
    [
        (corners[0] * (1 - x) + corners[1] * x) * (1 - y)
        + (corners[2] * (1 - x) + corners[3] * x) * y
        for y in np.linspace(0, 1, 10)
        for x in np.linspace(0, 1, 10)
    ]
)

inv_transformed_points = trans.inverse_transform(test_pts)

plt.scatter(
    trans.embedding_[:, 0],
    trans.embedding_[:, 1],
    c=mnist.target,
    cmap="Spectral",
    s=0.25,
)
plt.colorbar(boundaries=np.arange(11) - 0.5).set_ticks(np.arange(10))
plt.scatter(test_pts[:, 0], test_pts[:, 1], marker="x", c="k")

fig, ax = plt.subplots(10, 10)
for i in range(10):
    for j in range(10):
        ax[i, j].imshow(
            inv_transformed_points[i * 10 + j].reshape(28, 28), origin="upper"
        )
        ax[i, j].get_xaxis().set_visible(False)
        ax[i, j].get_yaxis().set_visible(False)

plt.show()
