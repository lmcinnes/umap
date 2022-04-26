import numpy as np
from umap import UMAP
from warnings import warn, catch_warnings, filterwarnings
from numba import TypingError
import os
from umap.spectral import spectral_layout
from sklearn.utils import check_random_state
import codecs, pickle
from sklearn.neighbors import KDTree
import sys


try:
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

TF_MAJOR_VERSION = int(tf.__version__.split(".")[0])
if TF_MAJOR_VERSION < 2:
    warn(
        """The umap.parametric_umap package requires Tensorflow > 2.0 to be installed.
    You can install Tensorflow at https://www.tensorflow.org/install
    
    or you can install the CPU version of Tensorflow using 

    pip install umap-learn[parametric_umap]

    """
    )
    raise ImportError("umap.parametric_umap requires Tensorflow >= 2.0") from None

try:
    import tensorflow_probability
except ImportError:
    warn(
        """ Global structure preservation in the umap.parametric_umap package requires 
        tensorflow_probability to be installed. You can install tensorflow_probability at
        https://www.tensorflow.org/probability, 
        
        or via

        pip install --upgrade tensorflow-probability

        Please ensure to install a version which is compatible to your tensorflow 
        installation. You can verify the correct release at 
        https://github.com/tensorflow/probability/releases.

        """
    )


class ParametricUMAP(UMAP):
    def __init__(
        self,
        optimizer=None,
        batch_size=None,
        dims=None,
        encoder=None,
        decoder=None,
        parametric_embedding=True,
        parametric_reconstruction=False,
        parametric_reconstruction_loss_fcn=tf.keras.losses.BinaryCrossentropy(
            from_logits=True
        ),
        parametric_reconstruction_loss_weight=1.0,
        autoencoder_loss=False,
        reconstruction_validation=None,
        loss_report_frequency=10,
        n_training_epochs=1,
        global_correlation_loss_weight=0,
        run_eagerly=False,
        keras_fit_kwargs={},
        **kwargs
    ):
        """
        Parametric UMAP subclassing UMAP-learn, based on keras/tensorflow.
        There is also a non-parametric implementation contained within to compare
        with the base non-parametric implementation.

        Parameters
        ----------
        optimizer : tf.keras.optimizers, optional
            The tensorflow optimizer used for embedding, by default None
        batch_size : int, optional
            size of batch used for batch training, by default None
        dims :  tuple, optional
            dimensionality of data, if not flat (e.g. (32x32x3 images for ConvNet), by default None
        encoder : tf.keras.Sequential, optional
            The encoder Keras network
        decoder : tf.keras.Sequential, optional
            the decoder Keras network
        parametric_embedding : bool, optional
            Whether the embedder is parametric or non-parametric, by default True
        parametric_reconstruction : bool, optional
            Whether the decoder is parametric or non-parametric, by default False
        parametric_reconstruction_loss_fcn : bool, optional
            What loss function to use for parametric reconstruction, by default tf.keras.losses.BinaryCrossentropy
        parametric_reconstruction_loss_weight : float, optional
            How to weight the parametric reconstruction loss relative to umap loss, by default 1.0
        autoencoder_loss : bool, optional
            [description], by default False
        reconstruction_validation : array, optional
            validation X data for reconstruction loss, by default None
        loss_report_frequency : int, optional
            how many times per epoch to report loss, by default 1
        n_training_epochs : int, optional
            number of epochs to train for, by default 1
        global_correlation_loss_weight : float, optional
            Whether to additionally train on correlation of global pairwise relationships (>0), by default 0
        run_eagerly : bool, optional
            Whether to run tensorflow eagerly
        keras_fit_kwargs : dict, optional
            additional arguments for model.fit (like callbacks), by default {}
        """
        super().__init__(**kwargs)

        # add to network
        self.dims = dims  # if this is an image, we should reshape for network
        self.encoder = encoder  # neural network used for embedding
        self.decoder = decoder  # neural network used for decoding
        self.parametric_embedding = (
            parametric_embedding  # nonparametric vs parametric embedding
        )
        self.parametric_reconstruction = parametric_reconstruction
        self.parametric_reconstruction_loss_fcn = parametric_reconstruction_loss_fcn
        self.parametric_reconstruction_loss_weight = (
            parametric_reconstruction_loss_weight
        )
        self.run_eagerly = run_eagerly
        self.autoencoder_loss = autoencoder_loss
        self.batch_size = batch_size
        self.loss_report_frequency = (
            loss_report_frequency  # how many times per epoch to report loss in keras
        )
        if "tensorflow_probability" in sys.modules:
            self.global_correlation_loss_weight = global_correlation_loss_weight
        else:
            warn(
                "tensorflow_probability not installed or incompatible to current \
                tensorflow installation. Setting global_correlation_loss_weight to zero."
            )
            self.global_correlation_loss_weight = 0

        self.reconstruction_validation = (
            reconstruction_validation  # holdout data for reconstruction acc
        )
        self.keras_fit_kwargs = keras_fit_kwargs  # arguments for model.fit
        self.parametric_model = None

        # how many epochs to train for (different than n_epochs which is specific to each sample)
        self.n_training_epochs = n_training_epochs
        # set optimizer
        if optimizer is None:
            if parametric_embedding:
                # Adam is better for parametric_embedding
                self.optimizer = tf.keras.optimizers.Adam(1e-3)
            else:
                # Larger learning rate can be used for embedding
                self.optimizer = tf.keras.optimizers.Adam(1e-1)
        else:
            self.optimizer = optimizer
        if parametric_reconstruction and not parametric_embedding:
            warn(
                "Parametric decoding is not implemented with nonparametric \
            embedding. Turning off parametric decoding"
            )
            self.parametric_reconstruction = False

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
            # geneate the graph on precomputed distances
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
        if self.parametric_embedding:
            return self.encoder.predict(
                np.asanyarray(X), batch_size=self.batch_size, verbose=self.verbose
            )
        else:
            warn(
                "Embedding new data is not supported by ParametricUMAP. \
                Using original embedder."
            )
            return super().transform(X)

    def inverse_transform(self, X):
        """ "Transform X in the existing embedded space back into the input
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

        # network outputs
        outputs = {}

        # inputs
        if self.parametric_embedding:
            to_x = tf.keras.layers.Input(shape=self.dims, name="to_x")
            from_x = tf.keras.layers.Input(shape=self.dims, name="from_x")
            inputs = [to_x, from_x]

            # parametric embedding
            embedding_to = self.encoder(to_x)
            embedding_from = self.encoder(from_x)

            if self.parametric_reconstruction:
                # parametric reconstruction
                if self.autoencoder_loss:
                    embedding_to_recon = self.decoder(embedding_to)
                else:
                    # stop gradient of reconstruction loss before it reaches the encoder
                    embedding_to_recon = self.decoder(tf.stop_gradient(embedding_to))

                embedding_to_recon = tf.keras.layers.Lambda(
                    lambda x: x, name="reconstruction"
                )(embedding_to_recon)

                outputs["reconstruction"] = embedding_to_recon

        else:
            # this is the sham input (its just a 0) to make keras think there is input data
            batch_sample = tf.keras.layers.Input(
                shape=(1), dtype=tf.int32, name="batch_sample"
            )

            # gather all of the edges (so keras model is happy)
            to_x = tf.squeeze(tf.gather(self.head, batch_sample[0]))
            from_x = tf.squeeze(tf.gather(self.tail, batch_sample[0]))

            # grab relevant embeddings
            embedding_to = self.encoder(to_x)[:, -1, :]
            embedding_from = self.encoder(from_x)[:, -1, :]

            inputs = [batch_sample]

        # concatenate to/from projections for loss computation
        embedding_to_from = tf.concat([embedding_to, embedding_from], axis=1)
        embedding_to_from = tf.keras.layers.Lambda(lambda x: x, name="umap")(
            embedding_to_from
        )
        outputs["umap"] = embedding_to_from

        if self.global_correlation_loss_weight > 0:
            outputs["global_correlation"] = tf.keras.layers.Lambda(
                lambda x: x, name="global_correlation"
            )(embedding_to)

        # create model

        # self.parametric_model = tf.keras.Model(inputs=inputs, outputs=outputs)
        self.parametric_model = GradientClippedModel(inputs=inputs, outputs=outputs)

    def _compile_model(self):
        """
        Compiles keras model with losses
        """
        losses = {}
        loss_weights = {}

        umap_loss_fn = umap_loss(
            self.batch_size,
            self.negative_sample_rate,
            self._a,
            self._b,
            self.edge_weight,
            self.parametric_embedding,
        )
        losses["umap"] = umap_loss_fn
        loss_weights["umap"] = 1.0

        if self.global_correlation_loss_weight > 0:
            losses["global_correlation"] = distance_loss_corr
            loss_weights["global_correlation"] = self.global_correlation_loss_weight
            if self.run_eagerly == False:
                # this is needed to avoid a 'NaN' error bug in tensorflow_probability (v0.12.2)
                warn("Setting tensorflow to run eagerly for global_correlation_loss.")
                self.run_eagerly = True

        if self.parametric_reconstruction:
            losses["reconstruction"] = self.parametric_reconstruction_loss_fcn
            loss_weights["reconstruction"] = self.parametric_reconstruction_loss_weight

        self.parametric_model.compile(
            optimizer=self.optimizer,
            loss=losses,
            loss_weights=loss_weights,
            run_eagerly=self.run_eagerly,
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
            self.parametric_embedding,
            self.parametric_reconstruction,
            self.global_correlation_loss_weight,
        )
        self.head = tf.constant(tf.expand_dims(head.astype(np.int64), 0))
        self.tail = tf.constant(tf.expand_dims(tail.astype(np.int64), 0))

        if self.parametric_embedding:
            init_embedding = None
        else:
            init_embedding = init_embedding_from_graph(
                X,
                self.graph_,
                self.n_components,
                self.random_state,
                self.metric,
                self._metric_kwds,
                init="spectral",
            )

        # create encoder and decoder model
        n_data = len(X)
        self.encoder, self.decoder = prepare_networks(
            self.encoder,
            self.decoder,
            self.n_components,
            self.dims,
            n_data,
            self.parametric_embedding,
            self.parametric_reconstruction,
            init_embedding,
        )

        # create the model
        self._define_model()
        self._compile_model()

        # report every loss_report_frequency subdivision of an epochs
        if self.parametric_embedding:
            steps_per_epoch = int(
                n_edges / self.batch_size / self.loss_report_frequency
            )
        else:
            # all edges are trained simultaneously with nonparametric, so this is arbitrary
            steps_per_epoch = 100

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
                    tf.zeros_like(self.reconstruction_validation),
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
            max_queue_size=100,
            validation_data=validation_data,
            **self.keras_fit_kwargs
        )
        # save loss history dictionary
        self._history = history.history

        # get the final embedding
        if self.parametric_embedding:
            embedding = self.encoder.predict(X, verbose=self.verbose)
        else:
            embedding = self.encoder.trainable_variables[0].numpy()

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
            encoder_output = os.path.join(save_location, "encoder")
            self.encoder.save(encoder_output)
            if verbose:
                print("Keras encoder model saved to {}".format(encoder_output))

        # save decoder
        if self.decoder is not None:
            decoder_output = os.path.join(save_location, "decoder")
            self.decoder.save(decoder_output)
            if verbose:
                print("Keras decoder model saved to {}".format(decoder_output))

        # save parametric_model
        if self.parametric_model is not None:
            parametric_model_output = os.path.join(save_location, "parametric_model")
            self.parametric_model.save(parametric_model_output)
            if verbose:
                print("Keras full model saved to {}".format(parametric_model_output))

        # # save model.pkl (ignoring unpickleable warnings)
        with catch_warnings():
            filterwarnings("ignore")
            # work around optimizers not pickling anymore (since tf 2.4)
            self._optimizer_dict = self.optimizer.get_config()
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
        number of verticies in graph
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
    return -tf.math.log1p(a * distances ** (2 * b))


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
        offset to to ensure log is taken of a positive number, by default 1e-4
    repulsion_strength : float, optional
        strength of repulsion between negative samples, by default 1.0

    Returns
    -------
    attraction_term: tf.float32
        attraction term for cross entropy loss
    repellant_term: tf.float32
        repellant term for cross entropy loss
    cross_entropy: tf.float32
        cross entropy umap loss

    """
    # cross entropy
    attraction_term = -probabilities_graph * tf.math.log_sigmoid(
        log_probabilities_distance
    )
    # use numerically stable repellant term
    # Shi et al. 2022 (https://arxiv.org/abs/2111.08851)
    # log(1 - sigmoid(logits)) = log(sigmoid(logits)) - logits
    repellant_term = (
        -(1.0 - probabilities_graph)
        * (tf.math.log_sigmoid(log_probabilities_distance) - log_probabilities_distance)
        * repulsion_strength
    )

    # balance the expected losses between atrraction and repel
    CE = attraction_term + repellant_term
    return attraction_term, repellant_term, CE


def umap_loss(
    batch_size,
    negative_sample_rate,
    _a,
    _b,
    edge_weights,
    parametric_embedding,
    repulsion_strength=1.0,
):
    """
    Generate a keras-ccompatible loss function for UMAP loss

    Parameters
    ----------
    batch_size : int
        size of mini-batches
    negative_sample_rate : int
        number of negative samples per positive samples to train on
    _a : float
        distance parameter in embedding space
    _b : float float
        distance parameter in embedding space
    edge_weights : array
        weights of all edges from sparse UMAP graph
    parametric_embedding : bool
        whether the embeddding is parametric or nonparametric
    repulsion_strength : float, optional
        strength of repulsion vs attraction for cross-entropy, by default 1.0

    Returns
    -------
    loss : function
        loss function that takes in a placeholder (0) and the output of the keras network
    """

    if not parametric_embedding:
        # multiply loss by weights for nonparametric
        weights_tiled = np.tile(edge_weights, negative_sample_rate + 1)

    @tf.function
    def loss(placeholder_y, embed_to_from):
        # split out to/from
        embedding_to, embedding_from = tf.split(
            embed_to_from, num_or_size_splits=2, axis=1
        )

        # get negative samples
        embedding_neg_to = tf.repeat(embedding_to, negative_sample_rate, axis=0)
        repeat_neg = tf.repeat(embedding_from, negative_sample_rate, axis=0)
        embedding_neg_from = tf.gather(
            repeat_neg, tf.random.shuffle(tf.range(tf.shape(repeat_neg)[0]))
        )

        #  distances between samples (and negative samples)
        distance_embedding = tf.concat(
            [
                tf.norm(embedding_to - embedding_from, axis=1),
                tf.norm(embedding_neg_to - embedding_neg_from, axis=1),
            ],
            axis=0,
        )

        # convert distances to probabilities
        log_probabilities_distance = convert_distance_to_log_probability(
            distance_embedding, _a, _b
        )

        # set true probabilities based on negative sampling
        probabilities_graph = tf.concat(
            [tf.ones(batch_size), tf.zeros(batch_size * negative_sample_rate)], axis=0
        )

        # compute cross entropy
        (attraction_loss, repellant_loss, ce_loss) = compute_cross_entropy(
            probabilities_graph,
            log_probabilities_distance,
            repulsion_strength=repulsion_strength,
        )

        if not parametric_embedding:
            ce_loss = ce_loss * weights_tiled

        return tf.reduce_mean(ce_loss)

    return loss


def distance_loss_corr(x, z_x):
    """Loss based on the distance between elements in a batch"""

    # flatten data
    x = tf.keras.layers.Flatten()(x)
    z_x = tf.keras.layers.Flatten()(z_x)

    ## z score data
    def z_score(x):
        return (x - tf.reduce_mean(x)) / tf.math.reduce_std(x)

    x = z_score(x)
    z_x = z_score(z_x)

    # clip distances to 10 standard deviations for stability
    x = tf.clip_by_value(x, -10, 10)
    z_x = tf.clip_by_value(z_x, -10, 10)

    dx = tf.math.reduce_euclidean_norm(x[1:] - x[:-1], axis=1)
    dz = tf.math.reduce_euclidean_norm(z_x[1:] - z_x[:-1], axis=1)

    # jitter dz to prevent mode collapse
    dz = dz + tf.random.uniform(dz.shape) * 1e-10

    # compute correlation
    corr_d = tf.squeeze(
        tensorflow_probability.stats.correlation(
            x=tf.expand_dims(dx, -1), y=tf.expand_dims(dz, -1)
        )
    )
    if tf.math.is_nan(corr_d):
        raise ValueError("NaN values found in correlation loss.")

    return -corr_d


def prepare_networks(
    encoder,
    decoder,
    n_components,
    dims,
    n_data,
    parametric_embedding,
    parametric_reconstruction,
    init_embedding=None,
):
    """
    Generates a set of keras networks for the encoder and decoder if one has not already
    been predefined.

    Parameters
    ----------
    encoder : tf.keras.Sequential
        The encoder Keras network
    decoder : tf.keras.Sequential
        the decoder Keras network
    n_components : int
        the dimensionality of the latent space
    dims : tuple of shape (dim1, dim2, dim3...)
        dimensionality of data
    n_data : number of elements in dataset
        # of elements in training dataset
    parametric_embedding : bool
        Whether the embedder is parametric or non-parametric
    parametric_reconstruction : bool
        Whether the decoder is parametric or non-parametric
    init_embedding : array (optional, default None)
        The initial embedding, for nonparametric embeddings

    Returns
    -------
    encoder: tf.keras.Sequential
        encoder keras network
    decoder: tf.keras.Sequential
        decoder keras network
    """

    if parametric_embedding:
        if encoder is None:
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
    else:
        embedding_layer = tf.keras.layers.Embedding(
            n_data, n_components, input_length=1
        )
        embedding_layer.build(input_shape=(1,))
        embedding_layer.set_weights([init_embedding])
        encoder = tf.keras.Sequential([embedding_layer])

    if decoder is None:
        if parametric_reconstruction:
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

    return encoder, decoder


def construct_edge_dataset(
    X,
    graph_,
    n_epochs,
    batch_size,
    parametric_embedding,
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
    parametric_embedding : bool
        Whether the embedder is parametric or non-parametric
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
        outputs = {"umap": tf.repeat(0, batch_size)}
        if global_correlation_loss_weight > 0:
            outputs["global_correlation"] = edge_to_batch
        if parametric_reconstruction:
            # add reconstruction to iterator output
            # edge_out = tf.concat([edge_to_batch, edge_from_batch], axis=0)
            outputs["reconstruction"] = edge_to_batch
        return (edge_to_batch, edge_from_batch), outputs

    def make_sham_generator():
        """
        The sham generator is a placeholder when all data is already intrinsic to
        the model, but keras wants some input data. Used for non-parametric
        embedding.
        """

        def sham_generator():
            while True:
                yield tf.zeros(1, dtype=tf.int32), tf.zeros(1, dtype=tf.int32)

        return sham_generator

    # get data from graph
    graph, epochs_per_sample, head, tail, weight, n_vertices = get_graph_elements(
        graph_, n_epochs
    )

    # number of elements per batch for embedding
    if batch_size is None:
        # batch size can be larger if its just over embeddings
        if parametric_embedding:
            batch_size = np.min([n_vertices, 1000])
        else:
            batch_size = len(head)

    edges_to_exp, edges_from_exp = (
        np.repeat(head, epochs_per_sample.astype("int")),
        np.repeat(tail, epochs_per_sample.astype("int")),
    )

    # shuffle edges
    shuffle_mask = np.random.permutation(range(len(edges_to_exp)))
    edges_to_exp = edges_to_exp[shuffle_mask].astype(np.int64)
    edges_from_exp = edges_from_exp[shuffle_mask].astype(np.int64)

    # create edge iterator
    if parametric_embedding:
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
    else:
        # nonparametric embedding uses a sham dataset
        gen = make_sham_generator()
        edge_dataset = tf.data.Dataset.from_generator(
            gen,
            (tf.int32, tf.int32),
            output_shapes=(tf.TensorShape(1), tf.TensorShape((1,))),
        )
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
        unpickled = pickle.loads(codecs.decode(pickled.encode(), "base64"))
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

    # Work around optimizer not pickling anymore (since tf 2.4)
    class_name = model._optimizer_dict["name"]
    OptimizerClass = getattr(tf.keras.optimizers, class_name)
    model.optimizer = OptimizerClass.from_config(model._optimizer_dict)

    # load encoder
    encoder_output = os.path.join(save_location, "encoder")
    if os.path.exists(encoder_output):
        model.encoder = tf.keras.models.load_model(encoder_output)
        if verbose:
            print("Keras encoder model loaded from {}".format(encoder_output))

    # save decoder
    decoder_output = os.path.join(save_location, "decoder")
    if os.path.exists(decoder_output):
        model.decoder = tf.keras.models.load_model(decoder_output)
        print("Keras decoder model loaded from {}".format(decoder_output))

    # get the custom loss function
    umap_loss_fn = umap_loss(
        model.batch_size,
        model.negative_sample_rate,
        model._a,
        model._b,
        model.edge_weight,
        model.parametric_embedding,
    )

    # save parametric_model
    parametric_model_output = os.path.join(save_location, "parametric_model")
    if os.path.exists(parametric_model_output):
        model.parametric_model = tf.keras.models.load_model(
            parametric_model_output, custom_objects={"loss": umap_loss_fn}
        )
        print("Keras full model loaded from {}".format(parametric_model_output))

    return model


class GradientClippedModel(tf.keras.Model):
    """
    We need to define a custom keras model here for gradient clipping,
    to stabilize training.
    """

    def train_step(self, data):
        # Unpack the data. Its structure depends on your model and
        # on what you pass to `fit()`.
        x, y = data

        with tf.GradientTape() as tape:
            y_pred = self(x, training=True)  # Forward pass
            # Compute the loss value
            # (the loss function is configured in `compile()`)
            loss = self.compiled_loss(y, y_pred, regularization_losses=self.losses)

        # Compute gradients
        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)
        gradients = [tf.clip_by_value(grad, -4.0, 4.0) for grad in gradients]
        gradients = [
            (tf.where(tf.math.is_nan(grad), tf.zeros_like(grad), grad))
            for grad in gradients
        ]

        # Update weights
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))
        # Update metrics (includes the metric that tracks the loss)
        self.compiled_metrics.update_state(y, y_pred)
        # Return a dict mapping metric names to current value
        return {m.name: m.result() for m in self.metrics}
