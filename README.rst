.. image:: https://img.shields.io/pypi/v/umap-learn.svg
    :target: https://pypi.python.org/pypi/umap-learn/
    :alt: PyPI Version
.. image:: https://anaconda.org/conda-forge/umap-learn/badges/version.svg
    :target: https://anaconda.org/conda-forge/umap-learn
    :alt: Conda-forge Version
.. image:: https://anaconda.org/conda-forge/umap-learn/badges/downloads.svg
    :target: https://anaconda.org/conda-forge/umap-learn
    :alt: Downloads from conda-forge
.. image:: https://img.shields.io/pypi/l/umap-learn.svg
    :target: https://github.com/lmcinnes/umap/blob/master/LICENSE.txt
    :alt: License
.. image:: https://travis-ci.org/lmcinnes/umap.svg
    :target: https://travis-ci.org/lmcinnes/umap
    :alt: Travis Build Status
.. image:: https://coveralls.io/repos/github/lmcinnes/umap/badge.svg
    :target: https://coveralls.io/github/lmcinnes/umap
    :alt: Test Coverage Status
.. image:: https://readthedocs.org/projects/umap-learn/badge/?version=latest
    :target: https://umap-learn.readthedocs.io/en/latest/?badge=latest
    :alt: Documentation Status
.. image:: http://joss.theoj.org/papers/10.21105/joss.00861/status.svg
    :target: https://doi.org/10.21105/joss.00861
    :alt: JOSS article for this repository

====
UMAP
====

Uniform Manifold Approximation and Projection (UMAP) is a dimension reduction
technique that can be used for visualisation similarly to t-SNE, but also for
general non-linear dimension reduction. The algorithm is founded on three
assumptions about the data

1. The data is uniformly distributed on a Riemannian manifold;
2. The Riemannian metric is locally constant (or can be approximated as such);
3. The manifold is locally connected.

From these assumptions it is possible to model the manifold with a fuzzy
topological structure. The embedding is found by searching for a low dimensional
projection of the data that has the closest possible equivalent fuzzy
topological structure.

The details for the underlying mathematics can be found in
`our paper on ArXiv <https://arxiv.org/abs/1802.03426>`_:

McInnes, L, Healy, J, *UMAP: Uniform Manifold Approximation and Projection
for Dimension Reduction*, ArXiv e-prints 1802.03426, 2018

The important thing is that you don't need to worry about that -- you can use
UMAP right now for dimension reduction and visualisation as easily as a drop
in replacement for scikit-learn's t-SNE.

Documentation is `available via ReadTheDocs <https://umap-learn.readthedocs.io/>`_.

---------------
How to use UMAP
---------------

The umap package inherits from sklearn classes, and thus drops in neatly
next to other sklearn transformers with an identical calling API.

.. code:: python

    import umap
    from sklearn.datasets import load_digits

    digits = load_digits()

    embedding = umap.UMAP().fit_transform(digits.data)

There are a number of parameters that can be set for the UMAP class; the
major ones are as follows:

 -  ``n_neighbors``: This determines the number of neighboring points used in
    local approximations of manifold structure. Larger values will result in
    more global structure being preserved at the loss of detailed local
    structure. In general this parameter should often be in the range 5 to
    50, with a choice of 10 to 15 being a sensible default.

 -  ``min_dist``: This controls how tightly the embedding is allowed compress
    points together. Larger values ensure embedded points are more evenly
    distributed, while smaller values allow the algorithm to optimise more
    accurately with regard to local structure. Sensible values are in the
    range 0.001 to 0.5, with 0.1 being a reasonable default.

 -  ``metric``: This determines the choice of metric used to measure distance
    in the input space. A wide variety of metrics are already coded, and a user
    defined function can be passed as long as it has been JITd by numba.

An example of making use of these options:

.. code:: python

    import umap
    from sklearn.datasets import load_digits

    digits = load_digits()

    embedding = umap.UMAP(n_neighbors=5,
                          min_dist=0.3,
                          metric='correlation').fit_transform(digits.data)

UMAP also supports fitting to sparse matrix data. For more details
please see `the UMAP documentation <https://umap-learn.readthedocs.io/>`_

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

Fifth, UMAP supports adding new points to an existing embedding via
the standard sklearn ``transform`` method. This means that UMAP can be
used as a preprocessing transformer in sklearn pipelines.

Sixth, UMAP supports supervised and semi-supervised dimension reduction.
This means that if you have label information that you wish to use as
extra information for dimension reduction (even if it is just partial
labelling) you can do that -- as simply as providing it as the ``y``
parameter in the fit method.

Finally UMAP has solid theoretical foundations in manifold learning
(see `our paper on ArXiv <https://arxiv.org/abs/1802.03426>`_).
This both justifies the approach and allows for further
extensions that will soon be added to the library
(embedding dataframes etc.).

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

**Install Options**

Conda install, via the excellent work of the conda-forge team:

.. code:: bash

    conda install -c conda-forge umap-learn

The conda-forge packages are available for linux, OS X, and Windows 64 bit.

PyPI install, presuming you have numba and sklearn and all its requirements
(numpy and scipy) installed:

.. code:: bash

    pip install umap-learn

If pip is having difficulties pulling the dependencies then we'd suggest installing
the dependencies manually using anaconda followed by pulling umap from pip:

.. code:: bash

    conda install numpy scipy
    conda install scikit-learn
    conda install numba
    pip install umap-learn

For a manual install get this package:

.. code:: bash

    wget https://github.com/lmcinnes/umap/archive/master.zip
    unzip master.zip
    rm master.zip
    cd umap-master

Install the requirements

.. code:: bash

    sudo pip install -r requirements.txt

or

.. code:: bash

    conda install scikit-learn numba

Install the package

.. code:: bash

    python setup.py install

----------------
Help and Support
----------------

Documentation is at `ReadTheDocs <https://umap-learn.readthedocs.io/>`_.
The documentation `includes a FAQ <https://umap-learn.readthedocs.io/en/latest/faq.html>`_ that
may answer your questions. If you still have questions then please
`open an issue <https://github.com/lmcinnes/umap/issues/new>`_
and I will try to provide any help and guidance that I can.

--------
Citation
--------

If you make use of this software for your work we would appreciate it if you
would cite the paper from the Journal of Open Source Software:

.. code:: bibtex

    @article{mcinnes2018umap-software,
      title={UMAP: Uniform Manifold Approximation and Projection},
      author={McInnes, Leland and Healy, John and Saul, Nathaniel and Grossberger, Lukas},
      journal={The Journal of Open Source Software},
      volume={3},
      number={29},
      pages={861},
      year={2018}
    }

If you would like to cite this algorithm in your work the ArXiv paper is the
current reference:

.. code:: bibtex

   @article{2018arXivUMAP,
        author = {{McInnes}, L. and {Healy}, J. and {Melville}, J.},
        title = "{UMAP: Uniform Manifold Approximation
        and Projection for Dimension Reduction}",
        journal = {ArXiv e-prints},
        archivePrefix = "arXiv",
        eprint = {1802.03426},
        primaryClass = "stat.ML",
        keywords = {Statistics - Machine Learning,
                    Computer Science - Computational Geometry,
                    Computer Science - Learning},
        year = 2018,
        month = feb,
   }

-------
License
-------

The umap package is 3-clause BSD licensed.

We would like to note that the umap package makes heavy use of
NumFOCUS sponsored projects, and would not be possible without
their support of those projects, so please `consider contributing to NumFOCUS <https://www.numfocus.org/membership>`_.

------------
Contributing
------------

Contributions are more than welcome! There are lots of opportunities
for potential projects, so please get in touch if you would like to
help out. Everything from code to notebooks to
examples and documentation are all *equally valuable* so please don't feel
you can't contribute. To contribute please
`fork the project <https://github.com/lmcinnes/umap/issues#fork-destination-box>`_
make your changes and
submit a pull request. We will do our best to work through any issues with
you and get your code merged into the main branch.


