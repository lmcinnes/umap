#!/usr/bin/env python

import matplotlib.pyplot as plt
import numba
import numpy as np
from mayavi import mlab
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

import umap

digits = load_digits()
X_train, X_test, y_train, y_test = train_test_split(
    digits.data, digits.target, stratify=digits.target, random_state=42
)

target_spaces = ["plane", "torus", "sphere"]

if "plane" in target_spaces:
    # embed onto a plane

    trans = umap.UMAP(
        n_neighbors=10,
        random_state=42,
        metric="euclidean",
        output_metric="euclidean",
        init="spectral",
        verbose=True,
    ).fit(X_train)

    plt.scatter(
        trans.embedding_[:, 0], trans.embedding_[:, 1], c=y_train, cmap="Spectral"
    )
    plt.show()

if "torus" in target_spaces:
    # embed onto a torus
    # note: this is a topological torus, not a geometric torus. Think
    # Pacman, not donut.

    @numba.njit(fastmath=True)
    def torus_euclidean_grad(x, y, torus_dimensions=(2 * np.pi, 2 * np.pi)):
        """Standard euclidean distance.

        ..math::
            D(x, y) = \sqrt{\sum_i (x_i - y_i)^2}
        """
        distance_sqr = 0.0
        g = np.zeros_like(x)
        for i in range(x.shape[0]):
            a = abs(x[i] - y[i])
            if 2 * a < torus_dimensions[i]:
                distance_sqr += a ** 2
                g[i] = x[i] - y[i]
            else:
                distance_sqr += (torus_dimensions[i] - a) ** 2
                g[i] = (x[i] - y[i]) * (a - torus_dimensions[i]) / a
        distance = np.sqrt(distance_sqr)
        return distance, g / (1e-6 + distance)

    trans = umap.UMAP(
        n_neighbors=10,
        random_state=42,
        metric="euclidean",
        output_metric=torus_euclidean_grad,
        init="spectral",
        min_dist=0.15,  # requires adjustment since the torus has limited space
        verbose=True,
    ).fit(X_train)

    mlab.clf()
    x, y, z = np.mgrid[-3:3:50j, -3:3:50j, -3:3:50j]

    # Plot a torus
    R = 2
    r = 1
    values = (R - np.sqrt(x ** 2 + y ** 2)) ** 2 + z ** 2 - r ** 2
    mlab.contour3d(x, y, z, values, color=(1.0, 1.0, 1.0), contours=[0])

    # torus angles -> 3D
    x = (R + r * np.cos(trans.embedding_[:, 0])) * np.cos(trans.embedding_[:, 1])
    y = (R + r * np.cos(trans.embedding_[:, 0])) * np.sin(trans.embedding_[:, 1])
    z = r * np.sin(trans.embedding_[:, 0])

    pts = mlab.points3d(
        x, y, z, y_train, colormap="spectral", scale_mode="none", scale_factor=0.1
    )

    mlab.show()

if "sphere" in target_spaces:
    # embed onto a sphere
    trans = umap.UMAP(
        n_neighbors=10,
        random_state=42,
        metric="euclidean",
        output_metric="haversine",
        init="spectral",
        min_dist=0.15,  # requires adjustment since the sphere has limited space
        verbose=True,
    ).fit(X_train)

    mlab.clf()
    x, y, z = np.mgrid[-3:3:50j, -3:3:50j, -3:3:50j]

    # Plot a sphere
    r = 3
    values = x ** 2 + y ** 2 + z ** 2 - r ** 2
    mlab.contour3d(x, y, z, values, color=(1.0, 1.0, 1.0), contours=[0])

    # latitude, longitude -> 3D
    x = r * np.sin(trans.embedding_[:, 0]) * np.cos(trans.embedding_[:, 1])
    y = r * np.sin(trans.embedding_[:, 0]) * np.sin(trans.embedding_[:, 1])
    z = r * np.cos(trans.embedding_[:, 0])

    pts = mlab.points3d(
        x, y, z, y_train, colormap="spectral", scale_mode="none", scale_factor=0.2
    )

    mlab.show()
