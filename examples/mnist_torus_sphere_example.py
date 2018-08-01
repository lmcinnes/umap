#!/usr/bin/env python

import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
import umap
from scipy.optimize import minimize


from mayavi import mlab
import matplotlib.pyplot as plt

digits = load_digits()

X_train, X_test, y_train, y_test = train_test_split(
    digits.data,
    digits.target,
    stratify=digits.target,
    random_state=42
)

if True:
    # embed onto a plane
    # note: this is a topological torus, not a geometric torus. Think
    # Pacman, not donut.

    trans = umap.UMAP(
        n_neighbors=10,
        random_state=42,
        input_metric='euclidean',
        output_metric_grad='euclidean',
        init='spectral',
    ).fit(X_train)

    plt.scatter(trans.embedding_[:, 0], trans.embedding_[:, 1], c=y_train, cmap='Spectral')
    plt.show()
    exit()

if False:
    # embed onto a torus
    # note: this is a topological torus, not a geometric torus. Think
    # Pacman, not donut.

    @numba.njit(fastmath=True)
    def torus_euclidean_grad(x, y, torus_dimensions=(2*np.pi,2*np.pi)):
        """Standard euclidean distance.

        ..math::
            D(x, y) = \sqrt{\sum_i (x_i - y_i)^2}
        """
        distance_sqr = 0.0
        g = np.zeros_like(x)
        for i in range(x.shape[0]):
            a = abs(x[i] - y[i])
            if 2*a < torus_dimensions[i]:
                distance_sqr += a ** 2
                g[i] = (x[i] - y[i])
            else:
                distance_sqr += (torus_dimensions[i]-a) ** 2
                g[i] = (x[i] - y[i]) * (a - torus_dimensions[i]) / a
        distance = np.sqrt(distance_sqr)
        return distance, g/(1e-6 + distance)


    trans = umap.UMAP(
        n_neighbors=10,
        random_state=42,
        input_metric='euclidean',
        output_metric_grad=torus_euclidean_grad,
        init='spectral',
        spread=0.5,
        # min_dist=0.01,
    ).fit(X_train)


    mlab.clf()
    x, y, z = np.mgrid[-3:3:50j, -3:3:50j, -3:3:50j]


    # Plot a torus
    R = 2
    r = 1
    values = (R - np.sqrt(x**2 + y**2))**2 + z**2 - r**2
    mlab.contour3d(x, y, z, values, color=(1.0,1.0,1.0), contours=[0])


    # torus angles -> 3D
    x = (R + r * np.cos(trans.embedding_[:, 0])) * np.cos(trans.embedding_[:, 1])
    y = (R + r * np.cos(trans.embedding_[:, 0])) * np.sin(trans.embedding_[:, 1])
    z = r * np.sin(trans.embedding_[:, 0])

    pts = mlab.points3d(x, y, z, y_train, colormap="spectral", scale_mode='none', scale_factor=.1)

    mlab.show()
    exit()

if False:
    # embed onto a sphere
    trans = umap.UMAP(
        n_neighbors=10,
        random_state=42,
        input_metric='euclidean',
        output_metric_grad='haversine',
        init='spectral',
        spread=0.5,
    ).fit(X_train)

    mlab.clf()
    x, y, z = np.mgrid[-3:3:50j, -3:3:50j, -3:3:50j]


    # Plot a sphere
    r = 1
    values = x**2 + y**2 + z**2 - r
    mlab.contour3d(x, y, z, values, color=(1.0,1.0,1.0), contours=[0])


    # latitude, longitude -> 3D
    x = r * np.sin(trans.embedding_[:, 0]) * np.cos(trans.embedding_[:, 1])
    y = r * np.sin(trans.embedding_[:, 0]) * np.sin(trans.embedding_[:, 1])
    z = r * np.cos(trans.embedding_[:, 0])

    pts = mlab.points3d(x, y, z, y_train, colormap="spectral", scale_mode='none', scale_factor=.1)

    mlab.show()
    exit()
