from .umap_utils import fuzzy_simplicial_set, embed_simplicial_set
from scipy.optimize import curve_fit
from sklearn.base import BaseEstimator

import numpy as np

class UMAP (BaseEstimator):

    def __init__(self,
                 n_neighbors=50,
                 n_components=2,
                 gamma=1.0,
                 n_edge_samples=None,
                 alpha=1.0,
                 init='spectral',
                 spread=1.0,
                 min_dist=0.25,
                 a=None,
                 b=None,
                 oversampling=3
                 ):

        self.n_neighbors = n_neighbors
        self.n_edge_samples = n_edge_samples
        self.init = init
        self.n_components = n_components
        self.gamma = gamma
        self.initial_alpha = alpha
        self.alpha = alpha

        self.spread = spread
        self.min_dist = min_dist

        self.oversampling = 3

        if a is None or b is None:
            self._find_ab_params()
        else:
            self.a = a
            self.b = b

    def _find_ab_params(self):
        def curve(x, a, b):
            return 1.0 / (1.0 + a * x ** (2 * b))

        xv = np.linspace(0, self.spread * 3, 300)
        yv = np.zeros(xv.shape)
        yv[xv < self.min_dist] = 1.0
        yv[xv >= self.min_dist] = np.exp(
            -(xv[xv >= self.min_dist] - self.min_dist) / self.spread)
        params, covar = curve_fit(curve, xv, yv)
        self.a = params[0]
        self.b = params[1]

    def fit(self, X, y=None):

        graph = fuzzy_simplicial_set(X, self.n_neighbors, self.oversampling)

        if self.n_edge_samples is None:
            n_edge_samples = 0
        else:
            n_edge_samples = self.n_edge_samples

        self.embedding_ = embed_simplicial_set(
            graph,
            self.n_components,
            self.initial_alpha,
            self.a,
            self.b,
            self.gamma,
            n_edge_samples,
            self.init
        )

    def fit_transform(self, X, y=None):
        self.fit(X)
        return self.embedding_
