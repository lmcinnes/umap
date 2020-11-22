from sklearn.datasets import make_moons
import numpy as np
import tensorflow as tf
from umap.parametric_umap import ParametricUMAP, load_ParametricUMAP
import platform
import tempfile


def test_create_model():
    """test a simple parametric UMAP network"""
    X, y = make_moons(100)
    embedder = ParametricUMAP()
    embedding = embedder.fit_transform(X)


def test_inverse_transform():
    """tests inverse_transform"""

    def norm(x):
        return (x - np.min(x)) / (np.max(x) - np.min(x))

    X, y = make_moons(100)
    X = norm(X)
    embedder = ParametricUMAP(parametric_reconstruction=True)
    embedding = embedder.fit_transform(X)
    Z = embedder.transform(X)
    X_r = embedder.inverse_transform(Z)


def test_nonparametric():
    """test nonparametric embedding"""
    X, y = make_moons(100)
    embedder = ParametricUMAP(parametric_embedding=False)
    embedding = embedder.fit_transform(X)


def test_custom_encoder_decoder():
    """test using a custom encoder / decoder"""
    X, y = make_moons(100)

    dims = (2,)
    n_components = 2
    encoder = tf.keras.Sequential(
        [
            tf.keras.layers.InputLayer(input_shape=dims),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(units=100, activation="relu"),
            tf.keras.layers.Dense(units=100, activation="relu"),
            tf.keras.layers.Dense(units=100, activation="relu"),
            tf.keras.layers.Dense(units=n_components, name="z"),
        ]
    )

    decoder = tf.keras.Sequential(
        [
            tf.keras.layers.InputLayer(input_shape=n_components),
            tf.keras.layers.Dense(units=100, activation="relu"),
            tf.keras.layers.Dense(units=100, activation="relu"),
            tf.keras.layers.Dense(units=100, activation="relu"),
            tf.keras.layers.Dense(
                units=np.product(dims), name="recon", activation=None
            ),
            tf.keras.layers.Reshape(dims),
        ]
    )

    embedder = ParametricUMAP(
        encoder=encoder,
        decoder=decoder,
        dims=dims,
        parametric_reconstruction=True,
        verbose=True,
    )
    embedding = embedder.fit_transform(X)


def test_validation():
    """tests adding a validation dataset"""
    X, y = make_moons(100)

    X_valid, y = make_moons(100)
    embedder = ParametricUMAP(
        parametric_reconstruction=True, reconstruction_validation=X_valid, verbose=True,
    )
    embedding = embedder.fit_transform(X)


def test_save_load():
    """tests saving and loading"""
    X, y = make_moons(100)
    embedder = ParametricUMAP()
    embedding = embedder.fit_transform(X)

    # if platform.system() != "Windows":
    # Portable tempfile
    model_path = tempfile.mkdtemp(suffix="_umap_model")

    embedder.save(model_path)
    embedder = load_ParametricUMAP(model_path)
