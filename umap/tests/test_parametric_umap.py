import numpy as np
import tempfile
import pytest
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from numpy.testing import assert_array_almost_equal
import platform

try:
    import tensorflow as tf

    IMPORT_TF = True
except ImportError:
    IMPORT_TF = False
else:
    from umap.parametric_umap import ParametricUMAP, load_ParametricUMAP

tf_only = pytest.mark.skipif(not IMPORT_TF, reason="TensorFlow >= 2.0 is not installed")
not_windows = pytest.mark.skipif(platform.system() == "Windows", reason="Windows file access issues")

@pytest.fixture(scope="session")
def moon_dataset():
    X, _ = make_moons(200)
    return X


@tf_only
def test_create_model(moon_dataset):
    """test a simple parametric UMAP network"""
    embedder = ParametricUMAP()
    embedding = embedder.fit_transform(moon_dataset)
    # completes successfully
    assert embedding is not None
    assert embedding.shape == (moon_dataset.shape[0], 2)


@tf_only
def test_global_loss(moon_dataset):
    """test a simple parametric UMAP network"""
    embedder = ParametricUMAP(global_correlation_loss_weight=1.0)
    embedding = embedder.fit_transform(moon_dataset)
    # completes successfully
    assert embedding is not None
    assert embedding.shape == (moon_dataset.shape[0], 2)


@tf_only
def test_inverse_transform(moon_dataset):
    """tests inverse_transform"""

    def norm(x):
        return (x - np.min(x)) / (np.max(x) - np.min(x))

    X = norm(moon_dataset)
    embedder = ParametricUMAP(parametric_reconstruction=True)
    Z = embedder.fit_transform(X)
    X_r = embedder.inverse_transform(Z)
    # completes successfully
    assert X_r is not None
    assert X_r.shape == X.shape


@tf_only
def test_nonparametric(moon_dataset):
    """test nonparametric embedding"""
    embedder = ParametricUMAP(parametric_embedding=False)
    embedding = embedder.fit_transform(moon_dataset)
    # completes successfully
    assert embedding is not None
    assert embedding.shape == (moon_dataset.shape[0], 2)


@tf_only
def test_custom_encoder_decoder(moon_dataset):
    """test using a custom encoder / decoder"""
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
    embedding = embedder.fit_transform(moon_dataset)
    # completes successfully
    assert embedding is not None
    assert embedding.shape == (moon_dataset.shape[0], 2)


@tf_only
def test_validation(moon_dataset):
    """tests adding a validation dataset"""
    X_train, X_valid = train_test_split(moon_dataset, train_size=0.5)
    embedder = ParametricUMAP(
        parametric_reconstruction=True, reconstruction_validation=X_valid, verbose=True
    )
    embedding = embedder.fit_transform(X_train)
    # completes successfully
    assert embedding is not None
    assert embedding.shape == (X_train.shape[0], 2)


@not_windows
@tf_only
def test_save_load(moon_dataset):
    """tests saving and loading"""

    embedder = ParametricUMAP()
    embedding = embedder.fit_transform(moon_dataset)
    # completes successfully
    assert embedding is not None
    assert embedding.shape == (moon_dataset.shape[0], 2)

    # Portable tempfile
    model_path = tempfile.mkdtemp(suffix="_umap_model")

    embedder.save(model_path)
    loaded_model = load_ParametricUMAP(model_path)
    assert loaded_model is not None

    loaded_embedding = loaded_model.transform(moon_dataset)
    assert_array_almost_equal(
        embedding,
        loaded_embedding,
        decimal=5,
        err_msg="Loaded model transform fails to match original embedding",
    )
