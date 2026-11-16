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
not_windows = pytest.mark.skipif(
    platform.system() == "Windows", reason="Windows file access issues"
)


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
def test_custom_encoder_decoder(moon_dataset):
    """test using a custom encoder / decoder"""
    dims = (2,)
    n_components = 2
    encoder = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=dims),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(units=100, activation="relu"),
            tf.keras.layers.Dense(units=100, activation="relu"),
            tf.keras.layers.Dense(units=100, activation="relu"),
            tf.keras.layers.Dense(units=int(n_components), name="z"),
        ]
    )

    decoder = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=(n_components,)),
            tf.keras.layers.Dense(units=100, activation="relu"),
            tf.keras.layers.Dense(units=100, activation="relu"),
            tf.keras.layers.Dense(units=100, activation="relu"),
            tf.keras.layers.Dense(
                units=int(np.prod(dims)), name="recon", activation=None
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


# @not_windows
# @tf_only
# def test_save_load(moon_dataset):
#     """tests saving and loading"""

#     embedder = ParametricUMAP()
#     embedding = embedder.fit_transform(moon_dataset)
#     # completes successfully
#     assert embedding is not None
#     assert embedding.shape == (moon_dataset.shape[0], 2)

#     # Portable tempfile
#     model_path = tempfile.mkdtemp(suffix="_umap_model")

#     embedder.save(model_path)
#     loaded_model = load_ParametricUMAP(model_path)
#     assert loaded_model is not None

#     loaded_embedding = loaded_model.transform(moon_dataset)
#     assert_array_almost_equal(
#         embedding,
#         loaded_embedding,
#         decimal=5,
#         err_msg="Loaded model transform fails to match original embedding",
#     )


@tf_only
def test_landmark_retraining_no_nan():
    """Retrain with landmarks should not produce NaN loss."""
    from sklearn.datasets import load_digits

    X, y = load_digits(return_X_y=True)
    x1, x2 = X[y != 9], X[y == 9]
    p = ParametricUMAP(n_epochs=50)
    p.fit(x1)
    p.add_landmarks(x1, sample_pct=0.05, landmark_loss_weight=0.01)
    p.fit(x2)
    assert not np.any(np.isnan(p._history["loss"][-5:]))
    assert p.parametric_model.landmark_loss_weight == 0.01


@tf_only
def test_umap_loss_grads_finite_under_xla():
    """UMAP loss must not give NaN gradients when two embeddings coincide.

    jit_compile is forced on so that CPU-only CI exercises the GPU path: Keras
    resolves jit_compile="auto" to True whenever a GPU is visible.
    """
    from umap.parametric_umap import UMAPModel, prepare_networks

    batch_size, dims = 128, 16
    encoder, decoder = prepare_networks(None, None, 2, [dims], batch_size, False)
    model = UMAPModel(1.577, 0.895, 5, encoder, decoder)

    to_x = np.random.RandomState(42).rand(batch_size, dims).astype(np.float32)
    # roll rather than randomise, so coincident pairs are guaranteed every run
    from_x = np.roll(to_x, batch_size // 2, axis=0)

    @tf.function(jit_compile=True)
    def grads():
        with tf.GradientTape() as tape:
            loss = model._umap_loss(model((to_x, from_x)))
        return tape.gradient(loss, model.trainable_variables)

    # the loss value stays finite while the gradients are already NaN, so this
    # has to assert on the gradients
    for g in grads():
        assert g is not None
        assert not np.any(np.isnan(np.asarray(g))), "NaN gradient under XLA"


class _LargeArray(np.ndarray):
    """Exercise the large-array branch without allocating half a gigabyte."""

    @property
    def nbytes(self):
        return 500_000_001


@tf_only
@pytest.mark.parametrize("dims", [(4,), (2, 2, 1)])
@pytest.mark.parametrize(
    "large_data,large_landmarks",
    [(True, False), (False, True), (True, True), (False, False)],
)
def test_edge_dataset_static_shapes(dims, large_data, large_landmarks):
    from scipy.sparse import coo_matrix
    from umap.parametric_umap import construct_edge_dataset

    X = np.arange(8 * np.prod(dims), dtype=np.float32).reshape((8,) + dims)
    landmarks = np.arange(16, dtype=np.float32).reshape(8, 2)
    if large_data:
        X = X.view(_LargeArray)
    if large_landmarks:
        landmarks = landmarks.view(_LargeArray)
    graph = coo_matrix(
        (np.ones(8), (np.arange(8), np.roll(np.arange(8), 1))), shape=(8, 8)
    )
    dataset, *_ = construct_edge_dataset(X, graph, 2, 4, True, 1.0, landmarks)
    inputs, outputs = dataset.element_spec
    assert inputs[0].shape == tf.TensorShape((4,) + dims)
    assert inputs[1].shape == tf.TensorShape((4,) + dims)
    assert outputs["reconstruction"].shape == inputs[0].shape
    assert outputs["global_correlation"].shape == inputs[0].shape
    assert outputs["landmark_to"].shape == tf.TensorShape((4, 2))
    options = tf.data.Options()
    options.threading.private_threadpool_size = 1
    (head, tail), values = next(iter(dataset.with_options(options)))
    # The feature's first value encodes its row, so check the sampled landmarks.
    rows = np.asarray(head).reshape(4, -1)[:, 0].astype(int) // np.prod(dims)
    np.testing.assert_array_equal(values["landmark_to"], landmarks[rows])
    np.testing.assert_array_equal(values["reconstruction"], head)
    np.testing.assert_array_equal(values["global_correlation"], head)
    assert tail.shape == head.shape


@tf_only
@pytest.mark.parametrize(
    "large_data,large_landmarks",
    [(True, False), (False, True), (True, True), (False, False)],
)
def test_large_edge_dataset_trains(large_data, large_landmarks):
    from scipy.sparse import coo_matrix
    from umap.parametric_umap import construct_edge_dataset, UMAPModel

    X = np.linspace(0, 1, 32, dtype=np.float32).reshape(8, 4)
    landmarks = np.zeros((8, 2), dtype=np.float32)
    if large_data:
        X = X.view(_LargeArray)
    if large_landmarks:
        landmarks = landmarks.view(_LargeArray)
    graph = coo_matrix(
        (np.ones(8), (np.arange(8), np.roll(np.arange(8), 1))), shape=(8, 8)
    )
    dataset, *_ = construct_edge_dataset(X, graph, 2, 4, False, 0.0, landmarks)
    options = tf.data.Options()
    options.threading.private_threadpool_size = 1
    dataset = dataset.with_options(options)
    encoder = tf.keras.Sequential(
        [tf.keras.layers.Input(shape=(4,)), tf.keras.layers.Dense(2)]
    )
    model = UMAPModel(1.0, 1.0, 2, encoder, None)
    history = model.fit(dataset, epochs=1, steps_per_epoch=2, verbose=0)
    assert np.isfinite(history.history["loss"]).all()
