"""
UMAP on the Galaxy10SDSS dataset
---------------------------------------------------------

This is an simple example of using UMAP on the Galaxy10SDSS
dataset. The goal of this example is largely to demonstrate
the use of supervised learning as an effective tool for
visualizing and reducing complex data.
"""

import numpy as np
import h5py
import matplotlib.pyplot as plt
import umap

# from sklearn.model_selection import train_test_split
import math
import requests

if not os.path.isfile("Galaxy10.h5"):
    url = "http://astro.utoronto.ca/~bovy/Galaxy10/Galaxy10.h5"
    r = requests.get(url, allow_redirects=True)
    open("Galaxy10.h5", "wb").write(r.content)

# To get the images and labels from file
with h5py.File("Galaxy10.h5", "r") as F:
    images = np.array(F["images"])
    labels = np.array(F["ans"])

X_train = np.empty([math.floor(len(labels) / 100), 14283], dtype=np.float64)
y_train = np.empty([math.floor(len(labels) / 100)], dtype=np.float64)
X_test = X_train
y_test = y_train
# Get a subset of the data
for i in range(math.floor(len(labels) / 100)):
    X_train[i, :] = np.array(np.ndarray.flatten(images[i, :, :, :]), dtype=np.float64)
    y_train[i] = labels[i]
    X_test[i, :] = np.array(
        np.ndarray.flatten(images[i + math.floor(len(labels) / 100), :, :, :]),
        dtype=np.float64,
    )
    y_test[i] = labels[i + math.floor(len(labels) / 100)]

# Plot distribution
classes, frequency = np.unique(y_train, return_counts=True)
fig = plt.figure(1, figsize=(4, 4))
plt.clf()
plt.bar(classes, frequency)
plt.xlabel("Class")
plt.ylabel("Frequency")
plt.title("Data Subset")
plt.savefig("galaxy10_subset.jpg")
# 2D Embedding
## UMAP
reducer = umap.UMAP(
    n_components=2, n_neighbors=5, random_state=42, transform_seed=42, verbose=False
)
reducer.fit(X_train)

galaxy10_umap = reducer.transform(X_train)
fig = plt.figure(1, figsize=(4, 4))
plt.clf()
plt.scatter(
    galaxy10_umap[:, 0],
    galaxy10_umap[:, 1],
    c=y_train,
    cmap=plt.cm.nipy_spectral,
    edgecolor="k",
    label=y_train,
)
plt.colorbar(boundaries=np.arange(11) - 0.5).set_ticks(np.arange(10))
plt.savefig("galaxy10_2D_umap.jpg")
### UMAP - Supervised
reducer = umap.UMAP(
    n_components=2, n_neighbors=15, random_state=42, transform_seed=42, verbose=False
)
reducer.fit(X_train, y_train)

galaxy10_umap_supervised = reducer.transform(X_train)
fig = plt.figure(1, figsize=(4, 4))
plt.clf()
plt.scatter(
    galaxy10_umap_supervised[:, 0],
    galaxy10_umap_supervised[:, 1],
    c=y_train,
    cmap=plt.cm.nipy_spectral,
    edgecolor="k",
    label=y_train,
)
plt.colorbar(boundaries=np.arange(11) - 0.5).set_ticks(np.arange(10))
plt.savefig("galaxy10_2D_umap_supervised.jpg")
### UMAP - Supervised prediction
galaxy10_umap_supervised_prediction = reducer.transform(X_test)
fig = plt.figure(1, figsize=(4, 4))
plt.clf()
plt.scatter(
    galaxy10_umap_supervised_prediction[:, 0],
    galaxy10_umap_supervised_prediction[:, 1],
    c=y_test,
    cmap=plt.cm.nipy_spectral,
    edgecolor="k",
    label=y_test,
)
plt.colorbar(boundaries=np.arange(11) - 0.5).set_ticks(np.arange(10))
plt.savefig("galaxy10_2D_umap_supervised_prediction.jpg")

# 3D Embedding
## UMAP
reducer = umap.UMAP(
    n_components=3, n_neighbors=5, random_state=42, transform_seed=42, verbose=False
)
reducer.fit(X_train)
galaxy10_umap = reducer.transform(X_train)
fig = plt.figure(1, figsize=(4, 4))
plt.clf()
ax = fig.add_subplot(projection="3d")
p = ax.scatter(
    galaxy10_umap[:, 0],
    galaxy10_umap[:, 1],
    galaxy10_umap[:, 2],
    c=y_train,
    cmap=plt.cm.nipy_spectral,
    edgecolor="k",
    label=y_train,
)
fig.colorbar(p, ax=ax, boundaries=np.arange(11) - 0.5).set_ticks(np.arange(10))
plt.savefig("galaxy10_3D_umap.jpg")
## UMAP - Supervised
reducer = umap.UMAP(
    n_components=3, n_neighbors=15, random_state=42, transform_seed=42, verbose=False
)
reducer.fit(X_train, y_train)
galaxy10_umap_supervised = reducer.transform(X_train)
fig = plt.figure(1, figsize=(4, 4))
plt.clf()
ax = fig.add_subplot(projection="3d")
p = ax.scatter(
    galaxy10_umap_supervised[:, 0],
    galaxy10_umap_supervised[:, 1],
    galaxy10_umap_supervised[:, 2],
    c=y_train,
    cmap=plt.cm.nipy_spectral,
    edgecolor="k",
    label=y_train,
)
fig.colorbar(p, ax=ax, boundaries=np.arange(11) - 0.5).set_ticks(np.arange(10))
plt.savefig("galaxy10_3D_umap_supervised.jpg")
## UMAP - Supervised prediction
galaxy10_umap_supervised_prediction = reducer.transform(X_test)
fig = plt.figure(1, figsize=(4, 4))
plt.clf()
ax = fig.add_subplot(projection="3d")
p = ax.scatter(
    galaxy10_umap_supervised_prediction[:, 0],
    galaxy10_umap_supervised_prediction[:, 1],
    galaxy10_umap_supervised_prediction[:, 2],
    c=y_test,
    cmap=plt.cm.nipy_spectral,
    edgecolor="k",
    label=y_test,
)
fig.colorbar(p, ax=ax, boundaries=np.arange(11) - 0.5).set_ticks(np.arange(10))
plt.savefig("galaxy10_3D_umap_supervised_prediction.jpg")
