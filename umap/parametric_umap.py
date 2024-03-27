import numpy as np
from umap import UMAP
from warnings import warn, catch_warnings, filterwarnings
from numba import TypingError
import os
from umap.spectral import spectral_layout
from sklearn.utils import check_random_state
import codecs, pickle
from sklearn.neighbors import KDTree

try:
    # Used for tf.data.
    import tensorflow as tf
except ImportError:
    warn(
        """The umap.parametric_umap package requires Tensorflow > 2.0 to be installed.
    You can install Tensorflow at https://www.tensorflow.org/install
    
    or you can install the CPU version of Tensorflow using 

    pip install umap-learn[parametric_umap]

    """
    )
    raise ImportError("umap.parametric_umap requires Tensorflow >= 2.0") from None

try:
    import keras
    from keras import ops
except ImportError:
    warn(
        """The umap.parametric_umap package requires Keras >= 3 to be installed."""
    )
    raise ImportError("umap.parametric_umap requires Keras") from None


class ParametricUMAP(UMAP):
    def __init__(
        self,
        batch_size=None,
        dims=None,
        encoder=None,
        decoder=None,
        parametric_reconstruction=False,
        parametric_reconstruction_loss_fcn=None,
        parametric_reconstruction_loss_weight=1.0,
        autoencoder_loss=False,
        reconstruction_validation=None,
        global_correlation_loss_weight=0,
        keras_fit_kwargs={},
        **kwargs
    ):
        """
        Parametric UMAP subclassing UMAP-learn, based on keras/tensorflow.
        There is also a non-parametric implementation contained within to compare
        with the base non-parametric implementation.

        Parameters
        ----------
        batch_size : int, optional
            size of batch used for batch training, by default None
        dims :  tuple, optional
            dimensionality of data, if not flat (e.g. (32x32x3 images for ConvNet), by default None
        encoder : keras.Sequential, optional
            The encoder Keras network
        decoder : keras.Sequential, optional
            the decoder Keras network
        parametric_reconstruction : bool, optional
            Whether the decoder is parametric or non-parametric, by default False
        parametric_reconstruction_loss_fcn : bool, optional
            What loss function to use for parametric reconstruction,
            by default keras.losses.BinaryCrossentropy
        parametric_reconstruction_loss_weight : float, optional
            How to weight the parametric reconstruction loss relative to umap loss, by default 1.0
        autoencoder_loss : bool, optional
            [description], by default False
        reconstruction_validation : array, optional
            validation X data for reconstruction loss, by default None
        global_correlation_loss_weight : float, optional
            Whether to additionally train on correlation of global pairwise relationships (>0), by default 0
        keras_fit_kwargs : dict, optional
            additional arguments for model.fit (like callbacks), by default {}
        """
        super().__init__(**kwargs)

        # add to network
        self.dims = dims  # if this is an image, we should reshape for network
        self.encoder = encoder  # neural network used for embedding
        self.decoder = decoder  # neural network used for decoding
        self.parametric_reconstruction = parametric_reconstruction
        self.parametric_reconstruction_loss_weight = (
            parametric_reconstruction_loss_weight
        )
        self.parametric_reconstruction_loss_fcn = parametric_reconstruction_loss_fcn
        self.autoencoder_loss = autoencoder_loss
        self.batch_size = batch_size
        self.loss_report_frequency = 10
        self.global_correlation_loss_weight = global_correlation_loss_weight

        self.reconstruction_validation = (
            reconstruction_validation  # holdout data for reconstruction acc
        )
        self.keras_fit_kwargs = keras_fit_kwargs  # arguments for model.fit
        self.parametric_model = None

        # How many epochs to train for
        # (different than n_epochs which is specific to each sample)
        self.n_training_epochs = 1

        # Set optimizer.
        # Adam is better for parametric_embedding. Use gradient clipping by value.
        self.optimizer = keras.optimizers.Adam(1e-3, clipvalue=4.0)

        if self.encoder is not None:
            if encoder.outputs[0].shape[-1] != self.n_components:
                raise ValueError(
                    (
                        "Dimensionality of embedder network output ({}) does"
                        "not match n_components ({})".format(
                            encoder.outputs[0].shape[-1], self.n_components
                        )
                    )
                )

    def fit(self, X, y=None, precomputed_distances=None):
        if self.metric == "precomputed":
            if precomputed_distances is None:
                raise ValueError(
                    "Precomputed distances must be supplied if metric \
                    is precomputed."
                )
            # prepare X for training the network
            self._X = X
            # geneate the graph on precomputed distances
            return super().fit(precomputed_distances, y)
        else:
            return super().fit(X, y)

    def fit_transform(self, X, y=None, precomputed_distances=None):

        if self.metric == "precomputed":
            if precomputed_distances is None:
                raise ValueError(
                    "Precomputed distances must be supplied if metric \
                    is precomputed."
                )
            # prepare X for training the network
            self._X = X
            # generate the graph on precomputed distances
            return super().fit_transform(precomputed_distances, y)
        else:
            return super().fit_transform(X, y)

    def transform(self, X):
        """Transform X into the existing embedded space and return that
        transformed output.
        Parameters
        ----------
        X : array, shape (n_samples, n_features)
            New data to be transformed.
        Returns
        -------
        X_new : array, shape (n_samples, n_components)
            Embedding of the new data in low-dimensional space.
        """
        return self.encoder.predict(
            np.asanyarray(X), batch_size=self.batch_size, verbose=self.verbose
        )

    def inverse_transform(self, X):
        """ Transform X in the existing embedded space back into the input
        data space and return that transformed output.
        Parameters
        ----------
        X : array, shape (n_samples, n_components)
            New points to be inverse transformed.
        Returns
        -------
        X_new : array, shape (n_samples, n_features)
            Generated data points new data in data space.
        """
        if self.parametric_reconstruction:
            return self.decoder.predict(
                np.asanyarray(X), batch_size=self.batch_size, verbose=self.verbose
            )
        else:
            return super().inverse_transform(X)

    def _define_model(self):
        """Define the model in keras"""
        prlw = self.parametric_reconstruction_loss_weight
        self.parametric_model = UMAPModel(
            self._a,
            self._b,
            negative_sample_rate=self.negative_sample_rate,
            encoder=self.encoder,
            decoder=self.decoder,
            parametric_reconstruction_loss_fn=self.parametric_reconstruction_loss_fcn,
            parametric_reconstruction=self.parametric_reconstruction,
            parametric_reconstruction_loss_weight=prlw,
            global_correlation_loss_weight=self.global_correlation_loss_weight,
            autoencoder_loss=self.autoencoder_loss,
        )

    def _fit_embed_data(self, X, n_epochs, init, random_state):

        if self.metric == "precomputed":
            X = self._X

        # get dimensionality of dataset
        if self.dims is None:
            self.dims = [np.shape(X)[-1]]
        else:
            # reshape data for network
            if len(self.dims) > 1:
                X = np.reshape(X, [len(X)] + list(self.dims))

        if self.parametric_reconstruction and (np.max(X) > 1.0 or np.min(X) < 0.0):
            warn(
                "Data should be scaled to the range 0-1 for cross-entropy reconstruction loss."
            )

        # get dataset of edges
        (
            edge_dataset,
            self.batch_size,
            n_edges,
            head,
            tail,
            self.edge_weight,
        ) = construct_edge_dataset(
            X,
            self.graph_,
            self.n_epochs,
            self.batch_size,
            self.parametric_reconstruction,
            self.global_correlation_loss_weight,
        )
        self.head = ops.array(ops.expand_dims(head.astype(np.int64), 0))
        self.tail = ops.array(ops.expand_dims(tail.astype(np.int64), 0))

        init_embedding = None

        # create encoder and decoder model
        n_data = len(X)
        self.encoder, self.decoder = prepare_networks(
            self.encoder,
            self.decoder,
            self.n_components,
            self.dims,
            n_data,
            self.parametric_reconstruction,
            init_embedding,
        )

        # create the model
        self._define_model()

        # report every loss_report_frequency subdivision of an epochs
        steps_per_epoch = int(
            n_edges / self.batch_size / self.loss_report_frequency
        )

        # Validation dataset for reconstruction
        if (
            self.parametric_reconstruction
            and self.reconstruction_validation is not None
        ):

            # reshape data for network
            if len(self.dims) > 1:
                self.reconstruction_validation = np.reshape(
                    self.reconstruction_validation,
                    [len(self.reconstruction_validation)] + list(self.dims),
                )

            validation_data = (
                (
                    self.reconstruction_validation,
                    ops.zeros_like(self.reconstruction_validation),
                ),
                {"reconstruction": self.reconstruction_validation},
            )
        else:
            validation_data = None

        # create embedding
        history = self.parametric_model.fit(
            edge_dataset,
            epochs=self.loss_report_frequency * self.n_training_epochs,
            steps_per_epoch=steps_per_epoch,
            validation_data=validation_data,
            **self.keras_fit_kwargs
        )
        # save loss history dictionary
        self._history = history.history

        # get the final embedding
        embedding = self.encoder.predict(X, verbose=self.verbose)

        return embedding, {}

    def __getstate__(self):
        # this function supports pickling, making sure that objects can be pickled
        return dict(
            (k, v)
            for (k, v) in self.__dict__.items()
            if should_pickle(k, v) and k not in ("optimizer", "encoder", "decoder", "parametric_model")
        )

    def save(self, save_location, verbose=True):

        # save encoder
        if self.encoder is not None:
            encoder_output = os.path.join(save_location, "encoder.keras")
            self.encoder.save(encoder_output)
            if verbose:
                print("Keras encoder model saved to {}".format(encoder_output))

        # save decoder
        if self.decoder is not None:
            decoder_output = os.path.join(save_location, "decoder.keras")
            self.decoder.save(decoder_output)
            if verbose:
                print("Keras decoder model saved to {}".format(decoder_output))

        # save parametric_model
        if self.parametric_model is not None:
            parametric_model_output = os.path.join(save_location, "parametric_model.keras")
            self.parametric_model.save(parametric_model_output)
            if verbose:
                print("Keras full model saved to {}".format(parametric_model_output))

        # # save model.pkl (ignoring unpickleable warnings)
        with catch_warnings():
            filterwarnings("ignore")
            model_output = os.path.join(save_location, "model.pkl")
            with open(model_output, "wb") as output:
                pickle.dump(self, output, pickle.HIGHEST_PROTOCOL)
            if verbose:
                print("Pickle of ParametricUMAP model saved to {}".format(model_output))


def get_graph_elements(graph_, n_epochs):
    """
    gets elements of graphs, weights, and number of epochs per edge

    Parameters
    ----------
    graph_ : scipy.sparse.csr.csr_matrix
        umap graph of probabilities
    n_epochs : int
        maximum number of epochs per edge

    Returns
    -------
    graph scipy.sparse.csr.csr_matrix
        umap graph
    epochs_per_sample np.array
        number of epochs to train each sample for
    head np.array
        edge head
    tail np.array
        edge tail
    weight np.array
        edge weight
    n_vertices int
        number of vertices in graph
    """
    ### should we remove redundancies () here??
    # graph_ = remove_redundant_edges(graph_)

    graph = graph_.tocoo()
    # eliminate duplicate entries by summing them together
    graph.sum_duplicates()
    # number of vertices in dataset
    n_vertices = graph.shape[1]
    # get the number of epochs based on the size of the dataset
    if n_epochs is None:
        # For smaller datasets we can use more epochs
        if graph.shape[0] <= 10000:
            n_epochs = 500
        else:
            n_epochs = 200
    # remove elements with very low probability
    graph.data[graph.data < (graph.data.max() / float(n_epochs))] = 0.0
    graph.eliminate_zeros()
    # get epochs per sample based upon edge probability
    epochs_per_sample = n_epochs * graph.data

    head = graph.row
    tail = graph.col
    weight = graph.data

    return graph, epochs_per_sample, head, tail, weight, n_vertices


def init_embedding_from_graph(
    _raw_data, graph, n_components, random_state, metric, _metric_kwds, init="spectral"
):
    """Initialize embedding using graph. This is for direct embeddings.

    Parameters
    ----------
    init : str, optional
        Type of initialization to use. Either random, or spectral, by default "spectral"

    Returns
    -------
    embedding : np.array
        the initialized embedding
    """
    if random_state is None:
        random_state = check_random_state(None)

    if isinstance(init, str) and init == "random":
        embedding = random_state.uniform(
            low=-10.0, high=10.0, size=(graph.shape[0], n_components)
        ).astype(np.float32)
    elif isinstance(init, str) and init == "spectral":
        # We add a little noise to avoid local minima for optimization to come

        initialisation = spectral_layout(
            _raw_data,
            graph,
            n_components,
            random_state,
            metric=metric,
            metric_kwds=_metric_kwds,
        )
        expansion = 10.0 / np.abs(initialisation).max()
        embedding = (initialisation * expansion).astype(
            np.float32
        ) + random_state.normal(
            scale=0.0001, size=[graph.shape[0], n_components]
        ).astype(
            np.float32
        )

    else:
        init_data = np.array(init)
        if len(init_data.shape) == 2:
            if np.unique(init_data, axis=0).shape[0] < init_data.shape[0]:
                tree = KDTree(init_data)
                dist, ind = tree.query(init_data, k=2)
                nndist = np.mean(dist[:, 1])
                embedding = init_data + random_state.normal(
                    scale=0.001 * nndist, size=init_data.shape
                ).astype(np.float32)
            else:
                embedding = init_data

    return embedding


def convert_distance_to_log_probability(distances, a=1.0, b=1.0):
    """
     convert distance representation into log probability,
        as a function of a, b params

    Parameters
    ----------
    distances : array
        euclidean distance between two points in embedding
    a : float, optional
        parameter based on min_dist, by default 1.0
    b : float, optional
        parameter based on min_dist, by default 1.0

    Returns
    -------
    float
        log probability in embedding space
    """
    return -ops.log1p(a * distances ** (2 * b))


def compute_cross_entropy(
    probabilities_graph, log_probabilities_distance, EPS=1e-4, repulsion_strength=1.0
):
    """
    Compute cross entropy between low and high probability

    Parameters
    ----------
    probabilities_graph : array
        high dimensional probabilities
    log_probabilities_distance : array
        low dimensional log probabilities
    EPS : float, optional
        offset to ensure log is taken of a positive number, by default 1e-4
    repulsion_strength : float, optional
        strength of repulsion between negative samples, by default 1.0

    Returns
    -------
    attraction_term: float
        attraction term for cross entropy loss
    repellant_term: float
        repellent term for cross entropy loss
    cross_entropy: float
        cross entropy umap loss

    """
    # cross entropy
    attraction_term = -probabilities_graph * ops.log_sigmoid(
        log_probabilities_distance
    )
    # use numerically stable repellent term
    # Shi et al. 2022 (https://arxiv.org/abs/2111.08851)
    # log(1 - sigmoid(logits)) = log(sigmoid(logits)) - logits
    repellant_term = (
        -(1.0 - probabilities_graph)
        * (ops.log_sigmoid(log_probabilities_distance) - log_probabilities_distance)
        * repulsion_strength
    )

    # balance the expected losses between attraction and repel
    CE = attraction_term + repellant_term
    return attraction_term, repellant_term, CE



def prepare_networks(
    encoder,
    decoder,
    n_components,
    dims,
    n_data,
    parametric_reconstruction,
    init_embedding=None,
):
    """
    Generates a set of keras networks for the encoder and decoder if one has not already
    been predefined.

    Parameters
    ----------
    encoder : keras.Sequential
        The encoder Keras network
    decoder : keras.Sequential
        the decoder Keras network
    n_components : int
        the dimensionality of the latent space
    dims : tuple of shape (dim1, dim2, dim3...)
        dimensionality of data
    n_data : number of elements in dataset
        # of elements in training dataset
    parametric_reconstruction : bool
        Whether the decoder is parametric or non-parametric
    init_embedding : array (optional, default None)
        The initial embedding, for nonparametric embeddings

    Returns
    -------
    encoder: keras.Sequential
        encoder keras network
    decoder: keras.Sequential
        decoder keras network
    """

    if encoder is None:
        encoder = keras.Sequential(
            [
                keras.layers.Input(shape=dims),
                keras.layers.Flatten(),
                keras.layers.Dense(units=100, activation="relu"),
                keras.layers.Dense(units=100, activation="relu"),
                keras.layers.Dense(units=100, activation="relu"),
                keras.layers.Dense(units=n_components, name="z"),
            ]
        )

    if decoder is None:
        if parametric_reconstruction:
            decoder = keras.Sequential(
                [
                    keras.layers.Input(shape=(n_components,)),
                    keras.layers.Dense(units=100, activation="relu"),
                    keras.layers.Dense(units=100, activation="relu"),
                    keras.layers.Dense(units=100, activation="relu"),
                    keras.layers.Dense(
                        units=np.product(dims), name="recon", activation=None
                    ),
                    keras.layers.Reshape(dims),
                ]
            )

    return encoder, decoder


def construct_edge_dataset(
    X,
    graph_,
    n_epochs,
    batch_size,
    parametric_reconstruction,
    global_correlation_loss_weight,
):
    """
    Construct a tf.data.Dataset of edges, sampled by edge weight.

    Parameters
    ----------
    X : array, shape (n_samples, n_features)
        New data to be transformed.
    graph_ : scipy.sparse.csr.csr_matrix
        Generated UMAP graph
    n_epochs : int
        # of epochs to train each edge
    batch_size : int
        batch size
    parametric_reconstruction : bool
        Whether the decoder is parametric or non-parametric
    """
    def gather_index(index):
        return X[index]

    # if X is > 512Mb in size, we need to use a different, slower method for
    #    batching data.
    gather_indices_in_python = True if X.nbytes * 1e-9 > 0.5 else False

    def gather_X(edge_to, edge_from):
        # gather data from indexes (edges) in either numpy of tf, depending on array size
        if gather_indices_in_python:
            edge_to_batch = tf.py_function(gather_index, [edge_to], [tf.float32])[0]
            edge_from_batch = tf.py_function(gather_index, [edge_from], [tf.float32])[0]
        else:
            edge_to_batch = tf.gather(X, edge_to)
            edge_from_batch = tf.gather(X, edge_from)
        return edge_to_batch, edge_from_batch

    def get_outputs(edge_to_batch, edge_from_batch):
        outputs = {"umap": ops.repeat(0, batch_size)}
        if global_correlation_loss_weight > 0:
            outputs["global_correlation"] = edge_to_batch
        if parametric_reconstruction:
            # add reconstruction to iterator output
            # edge_out = ops.concatenate([edge_to_batch, edge_from_batch], axis=0)
            outputs["reconstruction"] = edge_to_batch
        return (edge_to_batch, edge_from_batch), outputs

    # get data from graph
    _, epochs_per_sample, head, tail, weight, n_vertices = get_graph_elements(
        graph_, n_epochs
    )

    # number of elements per batch for embedding
    if batch_size is None:
        # batch size can be larger if its just over embeddings
        batch_size = int(np.min([n_vertices, 1000]))

    edges_to_exp, edges_from_exp = (
        np.repeat(head, epochs_per_sample.astype("int")),
        np.repeat(tail, epochs_per_sample.astype("int")),
    )

    # shuffle edges
    shuffle_mask = np.random.permutation(range(len(edges_to_exp)))
    edges_to_exp = edges_to_exp[shuffle_mask].astype(np.int64)
    edges_from_exp = edges_from_exp[shuffle_mask].astype(np.int64)

    # create edge iterator
    edge_dataset = tf.data.Dataset.from_tensor_slices(
        (edges_to_exp, edges_from_exp)
    )
    edge_dataset = edge_dataset.repeat()
    edge_dataset = edge_dataset.shuffle(10000)
    edge_dataset = edge_dataset.batch(batch_size, drop_remainder=True)
    edge_dataset = edge_dataset.map(
        gather_X, num_parallel_calls=tf.data.experimental.AUTOTUNE
    )
    edge_dataset = edge_dataset.map(
        get_outputs, num_parallel_calls=tf.data.experimental.AUTOTUNE
    )
    edge_dataset = edge_dataset.prefetch(10)

    return edge_dataset, batch_size, len(edges_to_exp), head, tail, weight


def should_pickle(key, val):
    """
    Checks if a dictionary item can be pickled

    Parameters
    ----------
    key : try
        key for dictionary element
    val : None
        element of dictionary

    Returns
    -------
    picklable: bool
        whether the dictionary item can be pickled
    """
    try:
        ## make sure object can be pickled and then re-read
        # pickle object
        pickled = codecs.encode(pickle.dumps(val), "base64").decode()
        # unpickle object
        _ = pickle.loads(codecs.decode(pickled.encode(), "base64"))
    except (
        pickle.PicklingError,
        tf.errors.InvalidArgumentError,
        TypeError,
        tf.errors.InternalError,
        tf.errors.NotFoundError,
        OverflowError,
        TypingError,
        AttributeError,
    ) as e:
        warn("Did not pickle {}: {}".format(key, e))
        return False
    except ValueError as e:
        warn(f"Failed at pickling {key}:{val} due to {e}")
        return False
    return True


def load_ParametricUMAP(save_location, verbose=True):
    """
    Load a parametric UMAP model consisting of a umap-learn UMAP object
    and corresponding keras models.

    Parameters
    ----------
    save_location : str
        the folder that the model was saved in
    verbose : bool, optional
        Whether to print the loading steps, by default True

    Returns
    -------
    parametric_umap.ParametricUMAP
        Parametric UMAP objects
    """

    ## Loads a ParametricUMAP model and its related keras models

    model_output = os.path.join(save_location, "model.pkl")
    model = pickle.load((open(model_output, "rb")))
    if verbose:
        print("Pickle of ParametricUMAP model loaded from {}".format(model_output))

    # load encoder
    encoder_output = os.path.join(save_location, "encoder.keras")
    if os.path.exists(encoder_output):
        model.encoder = keras.models.load_model(encoder_output)
        if verbose:
            print("Keras encoder model loaded from {}".format(encoder_output))

    # save decoder
    decoder_output = os.path.join(save_location, "decoder.keras")
    if os.path.exists(decoder_output):
        model.decoder = keras.models.load_model(decoder_output)
        print("Keras decoder model loaded from {}".format(decoder_output))

    # save parametric_model
    parametric_model_output = os.path.join(save_location, "parametric_model")
    if os.path.exists(parametric_model_output):
        model.parametric_model = keras.models.load_model(
            parametric_model_output
        )
        print("Keras full model loaded from {}".format(parametric_model_output))

    return model


def covariance(x,
               y=None,
               keepdims=False):
    """Adapted from TF Probability."""
    x = ops.convert_to_tensor(x)
    # Covariance *only* uses the centered versions of x (and y).
    x = x - ops.mean(x, axis=0, keepdims=True)

    if y is None:
        y = x
        event_axis = ops.mean(
          x * ops.conj(y), axis=0, keepdims=keepdims)
    else:
        y = ops.convert_to_tensor(y, dtype=x.dtype)
        y = y - ops.mean(y, axis=0, keepdims=True)
        event_axis = [len(x.shape) - 1]
    sample_axis = [0]

    event_axis = ops.cast(event_axis, dtype="int32")
    sample_axis = ops.cast(sample_axis, dtype="int32")

    x_permed = ops.transpose(x)
    y_permed = ops.transpose(y)

    n_events = ops.shape(x_permed)[0]
    n_samples = ops.shape(x_permed)[1]

    # Flatten sample_axis into one long dim.
    x_permed_flat = ops.reshape(
        x_permed, (n_events, n_samples))
    y_permed_flat = ops.reshape(
        y_permed, (n_events, n_samples))
    # Do the same for event_axis.
    x_permed_flat = ops.reshape(
        x_permed, (n_events, n_samples))
    y_permed_flat = ops.reshape(
        y_permed, (n_events, n_samples))

    # After matmul, cov.shape = batch_shape + [n_events, n_events]
    cov = ops.matmul(
        x_permed_flat, ops.transpose(y_permed_flat)) / ops.cast(
            n_samples, x.dtype)

    cov = ops.reshape(
        cov,
        (n_events**2, 1),
    )

    # Permuting by the argsort inverts the permutation, making
    # cov.shape have ones in the position where there were samples, and
    # [n_events * n_events] in the event position.
    cov = ops.transpose(cov)

    # Now expand event_shape**2 into event_shape + event_shape.
    # We here use (for the first time) the fact that we require event_axis to be
    # contiguous.
    cov = ops.reshape(
        cov,
        ops.shape(cov)[:1] + (n_events, n_events),
    )

    if not keepdims:
        cov = ops.squeeze(cov, axis=0)
    return cov


def correlation(x,
                y=None,
                keepdims=False):
    x = x / ops.std(x, axis=0, keepdims=True)
    if y is not None:
        y = y / ops.std(y, axis=0, keepdims=True)
    return covariance(
        x=x,
        y=y,
        keepdims=keepdims)


class StopGradient(keras.layers.Layer):
    def call(self, x):
        return ops.stop_gradient(x)


class UMAPModel(keras.Model):
    def __init__(self,
                 umap_loss_a,
                 umap_loss_b,
                 negative_sample_rate,
                 encoder,
                 decoder,
                 optimizer=None,
                 parametric_reconstruction_loss_fn=None,
                 parametric_reconstruction=False,
                 parametric_reconstruction_loss_weight=1.,
                 global_correlation_loss_weight=0.,
                 autoencoder_loss=False,
                 name="umap_model"):
        super().__init__(name=name)

        self.encoder = encoder
        self.decoder = decoder
        self.parametric_reconstruction = parametric_reconstruction
        self.global_correlation_loss_weight = global_correlation_loss_weight
        self.parametric_reconstruction_loss_weight = (
            parametric_reconstruction_loss_weight
        )
        self.negative_sample_rate = negative_sample_rate
        self.umap_loss_a = umap_loss_a
        self.umap_loss_b = umap_loss_b
        self.autoencoder_loss = autoencoder_loss

        optimizer = optimizer or keras.optimizers.Adam(1e-3, clipvalue=4.0)
        self.compile(optimizer=optimizer)

        self.flatten = keras.layers.Flatten()
        self.seed_generator = keras.random.SeedGenerator()
        if parametric_reconstruction_loss_fn is None:
            self.parametric_reconstruction_loss_fn = keras.losses.BinaryCrossentropy(
                from_logits=True
            )
        else:
            self.parametric_reconstruction_loss_fn = (
                parametric_reconstruction_loss_fn
            )

    def call(self, inputs):
        to_x, from_x = inputs
        embedding_to = self.encoder(to_x)
        embedding_from = self.encoder(from_x)

        y_pred = {
            "embedding_to": embedding_to,
            "embedding_from": embedding_from,
        }
        if self.parametric_reconstruction:
            # parametric reconstruction
            if self.autoencoder_loss:
                embedding_to_recon = self.decoder(embedding_to)
            else:
                # stop gradient of reconstruction loss before it reaches the encoder
                embedding_to_recon = self.decoder(ops.stop_gradient(embedding_to))
            y_pred["reconstruction"] = embedding_to_recon
        return y_pred

    def compute_loss(
        self, x=None, y=None, y_pred=None, sample_weight=None, **kwargs
    ):
        losses = []
        # Regularization losses.
        for loss in self.losses:
            losses.append(ops.cast(loss, dtype=keras.backend.floatx()))

        # umap loss
        losses.append(self._umap_loss(y_pred))

        # global correlation loss
        if self.global_correlation_loss_weight > 0:
            losses.append(self._global_correlation_loss(y, y_pred))

        # parametric reconstruction loss
        if self.parametric_reconstruction:
            losses.append(self._parametric_reconstruction_loss(y, y_pred))

        return ops.sum(losses)

    def _umap_loss(self, y_pred, repulsion_strength=1.0):
        # split out to/from
        embedding_to = y_pred["embedding_to"]
        embedding_from = y_pred["embedding_from"]

        # get negative samples
        embedding_neg_to = ops.repeat(embedding_to, self.negative_sample_rate, axis=0)
        repeat_neg = ops.repeat(embedding_from, self.negative_sample_rate, axis=0)

        repeat_neg_batch_dim = ops.shape(repeat_neg)[0]
        shuffled_indices = keras.random.shuffle(
            ops.arange(repeat_neg_batch_dim), seed=self.seed_generator)

        if keras.config.backend() == "tensorflow":
            embedding_neg_from = tf.gather(
                repeat_neg, shuffled_indices
            )
        else:
            embedding_neg_from = repeat_neg[shuffled_indices]

        #  distances between samples (and negative samples)
        distance_embedding = ops.concatenate(
            [
                ops.norm(embedding_to - embedding_from, axis=1),
                ops.norm(embedding_neg_to - embedding_neg_from, axis=1),
            ],
            axis=0,
        )

        # convert distances to probabilities
        log_probabilities_distance = convert_distance_to_log_probability(
            distance_embedding, self.umap_loss_a, self.umap_loss_b
        )

        # set true probabilities based on negative sampling
        batch_size = ops.shape(embedding_to)[0]
        probabilities_graph = ops.concatenate(
            [
                ops.ones((batch_size,)),
                ops.zeros((batch_size * self.negative_sample_rate,)),
            ],
            axis=0
        )

        # compute cross entropy
        (attraction_loss, repellant_loss, ce_loss) = compute_cross_entropy(
            probabilities_graph,
            log_probabilities_distance,
            repulsion_strength=repulsion_strength,
        )

        return ops.mean(ce_loss)

    def _global_correlation_loss(self, y, y_pred):
        # flatten data
        x = self.flatten(y["global_correlation"])
        z_x = self.flatten(y_pred["embedding_to"])

        # z score data
        def z_score(x):
            return (x - ops.mean(x)) / ops.std(x)

        x = z_score(x)
        z_x = z_score(z_x)

        # clip distances to 10 standard deviations for stability
        x = ops.clip(x, -10, 10)
        z_x = ops.clip(z_x, -10, 10)

        dx = ops.norm(x[1:] - x[:-1], axis=1)
        dz = ops.norm(z_x[1:] - z_x[:-1], axis=1)

        # jitter dz to prevent mode collapse
        dz = dz + keras.random.uniform(dz.shape, seed=self.seed_generator) * 1e-10

        # compute correlation
        corr_d = ops.squeeze(
            correlation(
                x=ops.expand_dims(dx, -1), y=ops.expand_dims(dz, -1)
            )
        )
        return -corr_d * self.global_correlation_loss_weight

    def _parametric_reconstruction_loss(self, y, y_pred):
        loss = self.parametric_reconstruction_loss_fn(
            y["reconstruction"], y_pred["reconstruction"])
        return loss * self.parametric_reconstruction_loss_weight
