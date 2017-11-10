====
UMAP
====

Uniform Manifold Approximation and Projection (UMAP) is a dimension reduction
technique that can be used for visualisation similarly to t-SNE, but also for
general non-linear dimension reduction. The algorithm is founded on three
assumptions about the data

1. The data is uniformly distributed on Riemannian manifold;
2. The Riemannian metric is locally constant (or can be approximated as such);
3. The manifold is locally connected (not globally, but locally).

From these assumptions it is possible to model the manifold with a fuzzy
topological structure. The embedding is found by searching for a low dimensional
projection of the data that has the closest possible equivalent fuzzy
topological structure.

The important thing is that you don't need to worry about that -- you can use
UMAP right now for dimension reduction and visualisation as easily as a drop
in replacement for scikit-learn's t-SNE.

---------------
How to use UMAP
---------------

----------------
Benefits of UMAP
----------------

UMAP has a few signficant wins in its current incarnation.

First of all UMAP is *fast*. It can handle large datasets and high
dimensional data without too much difficulty, scaling beyond what most t-SNE
packages can manage.

Second, UMAP scales well in embedding dimension -- it isn't just for
visualisation! You can use UMAP as a general purpose dimension reduction
technique as a preliminary step to other machine learning tasks. With a
little care (documentation on how to be careful is coming) it partners well
with the `hdbscan <https://github.com/scikit-learn-contrib/hdbscan>`_
clustering library.

Third, UMAP often performs better at preserving aspects of global structure of
the data than t-SNE. This means that it can often provide a better "big
picture" view of your data as well as preserving local neighbor relations.

Fourth, UMAP supports a wide variety of distance functions, including
non-metric distance functions such as *cosine distance* and *correlation
distance*. You can finally embed word vectors properly using cosine distance!

Finally UMAP has solid theoretical foundations in manifold learning (paper
forthcoming). This both justifies the approach and allows for further
extensions that will soon be added to the library (embedding dataframes,
semi-supervised dimension reduction, etc.).

------------------------
Performance and Examples
------------------------

UMAP is very efficient at embedding large high dimensional datasets. In
particular it scales well with both input dimension and embedding dimension.
Thus, for a problem such as the 784-dimensional MNIST digits dataset with
70000 data samples, UMAP can complete the embedding in around 2.5 minutes (as
compared with around 45 minutes for most t-SNE implementations). Despite this
runtime efficiency UMAP still produces high quality embeddings.

The obligatory MNIST digits dataset, embedded in 2 minutes  and 22
seconds using a 3.1 GHz Intel Core i7 processor (n_neighbors=10, min_dist=0
.001):

.. image:: images/umap_example_mnist1.png
    :alt: UMAP embedding of MNIST digits

The MNIST digits dataset is fairly straightforward however. A better test is
the more recent "Fashion MNIST" dataset of images of fashion items (again
70000 data sample in 784 dimensions). UMAP
produced this embedding in 2 minutes exactly (n_neighbors=5, min_dist=0.1):

.. image:: images/umap_example_fashion_mnist1.png
    :alt: UMAP embedding of "Fashion MNIST"

The UCI shuttle dataset (43500 sample in 8 dimensions) embeds well under
*correlation* distance in 2 minutes and 39 seconds (note the longer time
required for correlation distance computations):

.. image:: images/umap_example_shuttle.png
    :alt: UMAP embedding the UCI Shuttle dataset

----------
Installing
----------

UMAP depends upon ``scikit-learn``, and thus ``scikit-learn``'s dependencies
such as ``numpy`` and ``scipy``. UMAP adds a requirement for ``numba`` for
performance reasons. The original version used Cython, but the improved code
clarity, simplicity and performance of Numba made the transition necessary.

Requirements:

* numpy
* scipy
* scikit-learn
* numba

----------------
Help and Support
----------------

This project is still very young. I am currently trying to get example
notebooks and documentation prepared, but it may be a while before those are
available. In the meantime please `open an issue <https://github.com/lmcinnes/umap/issues/new`_
and I will try to provide any help and guidance that I can. Please also check
 the docstrings on the code, which provide some descriptions of the parameters.

-------
License
-------

The umap package is 3-clause BSD licensed. Enjoy.

------------
Contributing
------------

Contributions are more than welcome! There are lots of opportunities
for potential projects, so please get in touch if you would like to
help out. Everything from code to notebooks to
examples and documentation are all *equally valuable* so please don't feel
you can't contribute. To contribute please `fork the project <https://github.com/lmcinnes/umap/issues#fork-destination-box>`_ make your changes and
submit a pull request. We will do our best to work through any issues with
you and get your code merged into the main branch.


