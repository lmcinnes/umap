"""
Pytorch implimentation of ParametricUMAP
Borrows ideas/code from:
    * https://github.com/lmcinnes/umap/issues/580
    * https://colab.research.google.com/drive/1CYxt0GD-Y2zPMOnJIXJWsAhr0LdqI0R6
"""

import numpy as np
from pynndescent import NNDescent
from sklearn.utils import check_random_state
from tqdm.auto import tqdm as tq
from warnings import warn

from .umap_ import fuzzy_simplicial_set, find_ab_params

try:
    import torch
    from torch.utils.data import Dataset, DataLoader

except ImportError:
    warn(
        """The umap.torch package requires PyTorch to be installed.
    You can install PyTorch at https://pytorch.org/
    

    """
    )
    raise ImportError("umap.torch requires torch") from None


def convert_distance_to_probability(distances, a=1.0, b=1.0):
    return -torch.log1p(a * distances ** (2 * b))


def compute_cross_entropy(
    probabilities_graph, probabilities_distance, repulsion_strength=1.0
):
    # cross entropy
    attraction_term = -probabilities_graph * torch.nn.functional.logsigmoid(
        probabilities_distance
    )
    repellant_term = (
        -(1.0 - probabilities_graph)
        * (
            torch.nn.functional.logsigmoid(probabilities_distance)
            - probabilities_distance
        )
        * repulsion_strength
    )

    # balance the expected losses between attraction and repulsion
    CE = attraction_term + repellant_term
    return attraction_term, repellant_term, CE


def umap_loss(embedding_to, embedding_from, _a, _b, batch_size, negative_sample_rate=5):
    # get negative samples by randomly shuffling the batch
    embedding_neg_to = embedding_to.repeat(negative_sample_rate, 1)
    repeat_neg = embedding_from.repeat(negative_sample_rate, 1)
    embedding_neg_from = repeat_neg[torch.randperm(repeat_neg.shape[0])]
    distance_embedding = torch.cat(
        (
            (embedding_to - embedding_from).norm(dim=1),
            (embedding_neg_to - embedding_neg_from).norm(dim=1),
        ),
        dim=0,
    )

    # convert probabilities to distances
    probabilities_distance = convert_distance_to_probability(distance_embedding, _a, _b)
    # set true probabilities based on negative sampling
    probabilities_graph = torch.cat(
        (torch.ones(batch_size), torch.zeros(batch_size * negative_sample_rate)),
        dim=0,
    )

    # compute cross entropy
    (attraction_loss, repellant_loss, ce_loss) = compute_cross_entropy(
        probabilities_graph.cuda(),
        probabilities_distance.cuda(),
    )
    loss = torch.mean(ce_loss)
    return loss


def get_umap_graph(
    X, n_neighbors=10, metric="cosine", random_state=None, verbose=False
):
    random_state = check_random_state(None) if random_state == None else random_state
    # number of trees in random projection forest
    n_trees = 5 + int(round((X.shape[0]) ** 0.5 / 20.0))
    # max number of nearest neighbor iters to perform
    n_iters = max(5, int(round(np.log2(X.shape[0]))))
    # distance metric

    # get nearest neighbors
    nnd = NNDescent(
        X.reshape((len(X), np.product(np.shape(X)[1:]))),
        n_neighbors=n_neighbors,
        metric=metric,
        n_trees=n_trees,
        n_iters=n_iters,
        max_candidates=60,
        verbose=verbose,
    )
    # get indices and distances
    knn_indices, knn_dists = nnd.neighbor_graph

    # get indices and distances
    knn_indices, knn_dists = nnd.neighbor_graph
    # build fuzzy_simplicial_set
    umap_graph, sigmas, rhos = fuzzy_simplicial_set(
        X=X,
        n_neighbors=n_neighbors,
        metric=metric,
        random_state=random_state,
        knn_indices=knn_indices,
        knn_dists=knn_dists,
    )

    return umap_graph


def get_graph_elements(graph_, n_epochs):

    graph = graph_.tocoo()
    # eliminate duplicate entries by summing them together
    graph.sum_duplicates()
    # number of vertices in dataset
    n_vertices = graph.shape[1]
    # get the number of epochs based on the size of the dataset
    if n_epochs is None:
        # For smaller datasets we can use more epochs
        if graph.shape[0] <= 10_000:
            n_epochs = 25
        else:
            n_epochs = 10

    # remove elements with very low probability
    graph.data[graph.data < (graph.data.max() / float(n_epochs))] = 0.0
    graph.eliminate_zeros()
    # get epochs per sample based upon edge probability
    epochs_per_sample = n_epochs * graph.data

    head = graph.row
    tail = graph.col
    weight = graph.data

    return graph, epochs_per_sample, head, tail, weight, n_vertices


class UMAPDataset(Dataset):
    """A dataset containing positive edges from the umap graph.
    If data is provided, returns the data vectors, otherwise returns the indices
    """

    def __init__(self, graph_, data=None, n_epochs=None):
        graph, epochs_per_sample, head, tail, weight, n_vertices = get_graph_elements(
            graph_, n_epochs
        )

        self.edges_to_ix, self.edges_from_ix = (
            np.repeat(head, epochs_per_sample.astype("int")),
            np.repeat(tail, epochs_per_sample.astype("int")),
        )

        if data is not None:
            self.data = torch.Tensor(data)
        else:
            self.data = None

    def __len__(self):
        return int(self.edges_to_ix.shape[0])

    def __getitem__(self, index):
        edges_to_ix = self.edges_to_ix[index]
        edges_from_ix = self.edges_from_ix[index]

        if self.data is not None:
            edges_to_exp = self.data[edges_to_ix]
            edges_from_exp = self.data[edges_from_ix]
            return edges_to_exp, edges_from_exp
        else:
            return edges_to_ix, edges_from_ix


class Encoder(torch.nn.Module):
    """
    Default encoder for ParametricUmap class
    """

    def __init__(
        self,
        input_channels,
        output_channels,
        hidden_channels=128,
        activation=torch.nn.LeakyReLU,
    ):
        super().__init__()
        self.encoder = torch.nn.Sequential(
            torch.nn.Linear(input_channels, hidden_channels),
            activation(),
            torch.nn.Linear(hidden_channels, hidden_channels),
            activation(),
            torch.nn.Linear(hidden_channels, output_channels),
        )

    def forward(self, X):
        return self.encoder(X)


class ParametricUMAP:
    def __init__(
        self,
        n_components=2,
        n_neighbors=15,
        metric="euclidean",
        n_epochs=3,
        n_subepochs=None,
        lr=1e-3,
        min_dist=0.1,
        encoder=None,
        decoder=None,
        beta=1,
        batch_size=1024,
        num_workers=4,
        random_state=None,
        device=None,
        verbose=True,
    ):
        """
            Parametric UMAP implimentation in PyTorch

            Parameters
        ----------
        n_components: int (optional, default 2)
            The dimension of the space to embed into. This defaults to 2 to
            provide easy visualization, but can reasonably be set to any
            integer value in the range 2 to 100.

        n_neighbors: float (optional, default 15)
            The size of local neighborhood (in terms of number of neighboring
            sample points) used for manifold approximation. Larger values
            result in more global views of the manifold, while smaller
            values result in more local data being preserved. In general
            values should be in the range 2 to 100.

        metric: string or function (optional, default 'euclidean')
            The metric to use to compute distances in high dimensional space.
            If a string is passed it must match a valid predefined metric. If
            a general metric is required a function that takes two 1d arrays and
            returns a float can be provided. For performance purposes it is
            required that this be a numba jit'd function. Valid string metrics
            include:

            * euclidean
            * manhattan
            * chebyshev
            * minkowski
            * canberra
            * braycurtis
            * mahalanobis
            * wminkowski
            * seuclidean
            * cosine
            * correlation
            * haversine
            * hamming
            * jaccard
            * dice
            * russelrao
            * kulsinski
            * ll_dirichlet
            * hellinger
            * rogerstanimoto
            * sokalmichener
            * sokalsneath
            * yule

            TODO: The torch implimentation currently does not support additional
            arguments that should be passed to the metric (e.g. minkowski, mahalanobis etc.)

        n_epochs: int (optional, default 3)
            The number of training epochs to be used in optimizing the
            low dimensional embedding. Corresponds to the number of times
            we optimisze over the training dataloader.

        n_subepochs: int (optional, default None)
            The number of epochs used in constructing the UMAP dataset.
            The highest probability edge in the umap graph will appear in
            the train dataset n_subepochs times. edges with lower probability
            are represented proportionally. A larger value will result in
            a larger, but more accurate dataset.
            Defaults to 25 for small datasets, 10 for large.

        lr: float (optional, default 1e-3)
            The learning rate for the embedding optimization.
            Passed to the torch optimizer.

        min_dist: float (optional, default 0.1)
            The effective minimum distance between embedded points. Smaller values
            will result in a more clustered/clumped embedding where nearby points
            on the manifold are drawn closer together, while larger values will
            result on a more even dispersal of points. The value should be set
            relative to the ``spread`` value, which determines the scale at which
            embedded points will be spread out.

        encoder: torch.nn.Module (optional, default None)
            An encoder which takes items from your data and maps them to
            vectors of size n_components. Defaults to a standard multi-layer
            encoder model (3 linear layers with LeakyReLU activation).

        decoder: torch.nn.Module (optional, default None)
            A decoder for inverting vectors of shape n_components, returning
            vectors shaped like the input data. Default is none, meaning that
            we do not train a decoder.

        beta: float (optional, default 1)
            The contribution of the decoder loss to the total loss. Total loss
            is given by umap_loss + beta * decoder_loss. Increasing/decreasing
            this will prioritise decoder loss over umap loss and vice versa.

        batch_size: int (optional, default 1024)
            Training batch size. 1024 is a sensible default for medium-large datasets.

        num_workers: int (optional, default 4)
            Number of workers used to manage the training dataloader.
            Defaults to 4, but performance may be boosted by increasing this for
            large datasets on machines with many cores.

        random_state: int or instance of RandomState (optional, default None)
            controls the random_state which is used in creating the umap graph.
            Setting this seed does not guarantee reproducability since it is
            not passed through to the torch modules.

        device: str, 'cpu' or 'cuda' (optional, default None)
            Controls the device on which we train the umap model. Set to 'cpu'
            for cpu training, or 'cuda' for gpu training. Default behaviour is
            to search for the active device via torch.cuda.is_available().

        verbose: bool (optional, default True)
            Controls whether we have progress bars during training.

        """
        self.n_components = n_components
        self.encoder = encoder
        self.decoder = decoder
        self.n_neighbors = n_neighbors
        self.min_dist = min_dist
        self.beta = beta
        self.metric = metric
        self.lr = lr
        self.n_epochs = n_epochs
        self.n_subepochs = n_subepochs
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.random_state = random_state
        self._a, self._b = find_ab_params(1.0, self.min_dist)

        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        self.verbose = verbose

    def fit(self, X):

        if type(X) is np.ndarray:
            X = torch.from_numpy(X).float()

        assert isinstance(X, torch.Tensor)

        if self.encoder is None:
            self.encoder = Encoder(
                input_channels=X.shape[-1], output_channels=self.n_components
            )

        # Move encoder/decoder to correct device
        self.encoder.to(self.device)
        if self.decoder is not None:
            self.decoder.to(self.device)

        graph = get_umap_graph(
            X,
            n_neighbors=self.n_neighbors,
            metric=self.metric,
            random_state=self.random_state,
        )

        dataset = UMAPDataset(graph, data=X, n_epochs=self.n_subepochs)

        dataloader = DataLoader(
            dataset=dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
        )

        # Don't forget to add decoder to optimizer if it is present
        if self.decoder is None:
            optimizer = torch.optim.AdamW(self.encoder.parameters(), lr=self.lr)
        else:
            optimizer = torch.optim.AdamW(
                (*self.encoder.parameters(), *self.decoder.parameters()), lr=self.lr
            )

        # Use tqdm for nice loading bars if verbose flag set
        # otherwise run silently
        if self.verbose:
            wrapper = lambda loader: tq(
                enumerate(loader), total=len(loader), leave=False
            )
        else:
            wrapper = lambda loader: enumerate(loader)

        for epoch in range(self.n_epochs):

            for ib, batch in (batch_pbar := wrapper(dataloader)):

                total_loss = 0

                edges_to_exp, edges_from_exp = batch
                edges_to_exp = edges_to_exp.to(self.device)
                edges_from_exp = edges_from_exp.to(self.device)

                embedding_to = self.encoder(edges_to_exp)
                embedding_from = self.encoder(edges_from_exp)

                encoder_loss = umap_loss(
                    embedding_to,
                    embedding_from,
                    self._a,
                    self._b,
                    edges_to_exp.shape[0],
                    negative_sample_rate=5,
                )

                total_loss += encoder_loss

                if self.decoder != None:
                    recon = self.decoder(embedding_to)
                    recon_loss = torch.nn.functional.mse_loss(recon, edges_to_exp)
                    total_loss += self.beta * recon_loss

                total_loss.backward()
                optimizer.step()
                optimizer.zero_grad()

                if self.verbose:
                    desc = f"Batch: {ib}  Training loss: {total_loss.item():5.3f}"
                    if self.decoder != None:
                        desc += f" | Umap loss: {encoder_loss.item():5.3f}"
                        desc += f" | Reconstruction loss: {recon_loss.item():5.3f}"
                    batch_pbar.set_description(desc)

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)

    @torch.no_grad()
    def transform(self, X):
        self.embedding_ = self.encoder(X.to(self.device)).detach().cpu().numpy()
        return self.embedding_

    @torch.no_grad()
    def inverse_transform(self, Z):
        assert (
            self.decoder is not None
        ), "No inverse_transform available, decoder is None."
        if type(Z) is np.ndarray:
            Z = torch.from_numpy(Z).float()
        return self.decoder(Z.to(self.device)).detach().cpu().numpy()
