====
UMAP
====

Uniform Manifold Approximation and Projection (UMAP) is a dimension reduction technique that can be used for visualisation similarly to t-SNE, but also for general non-linear dimension reduction. The algorithm is founded on three assumptions about the data

1. The data is uniformly distributed on Riemannian manifold;
2. The Riemannian metric is locally constant (or can be approximated as such);
3. The manifold is locally connected (not globally, but locally).

From these assumptions it is possible to model the manifold with a fuzzy topological structure. The embedding is found by searching for a low dimensional projection of the data that has the closest possible equivalent fuzzy topological structure.

----------
Installing
----------

UMAP should mostly work providing you have all the requirements. The primary
requirements at this time are:

* numpy
* scipy
* scikit-learn
* cython

--------
Examples
--------

How well does it work? Here are some examples on the digits dataset.

First the small sklearn digits data (~1700 digits in 64 space):

.. image:: images/sklearn_digits.png
    :alt: Embedding of the sklearn digits dataset

Now on the full test set (10000 digits in 784 space):

.. image:: images/mnist_digits.png
    :alt: Embedding of the full digits test dataset

And because no ML presentation is complete without it, the Iris dataset:

.. image:: images/iris.png
    :alt: Embedding of the iris dataset under several techniques


------------
Contributing
------------

Contributions are more than welcome! All of the code is highly experimental
at the current time (the algorithm itself is under active development), so
unfortunately it is not as approachable as it will hopefully eventually be.
However, if you are interested please contact me (file and issue) and we
can discuss ideas for contributions. Everything from code to notebooks to
examples and documentation are all *equally valuable* so please don't feel
you can't contribute.


