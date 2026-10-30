Performance
===========

For current runtime measurements and a discussion of how UMAP scales in
comparison with other maintained dimension reduction implementations, see
:doc:`benchmarking`.

The benchmark is generated from ``notebooks/benchmarking.ipynb`` and records
the package versions, thread configuration, dataset preparation, and estimator
settings used to produce the published results.

Nearest-neighbor backends
-------------------------

Building the nearest-neighbor graph is often the most expensive part of
fitting UMAP on large datasets. By default UMAP uses
`PyNNDescent <https://github.com/lmcinnes/pynndescent>`_. Version 0.6 can
optionally use `RINND <https://pypi.org/project/rinnd/>`_ instead, which
is often considerably faster:

.. code:: bash

    pip install umap-learn[rinnd]

The backend is selected with ``nearest_neighbor_backend``:

``"auto"`` (default)
    Use RINND when it is installed and every requirement below is met, and
    PyNNDescent otherwise. In ``"auto"`` mode RINND is only used for
    datasets with at least 4,096 samples; smaller datasets are handled
    exactly anyway unless ``force_approximation_algorithm=True``.

``"rinnd"``
    Require RINND. A ``ValueError`` explains which requirement is not met
    instead of falling back.

``"pynndescent"``
    Always use PyNNDescent.

RINND is used only when:

* ``compatibility_layout=False`` (the compatibility layout always uses
  PyNNDescent, so that results match earlier releases);
* RINND 0.3.1 or later is installed;
* the input is a dense, two-dimensional ``float32`` NumPy array (UMAP
  converts dense input to ``float32`` automatically);
* the metric is one of the named metrics verified against PyNNDescent, such
  as ``"euclidean"``, ``"manhattan"``, ``"cosine"``, ``"correlation"``,
  ``"hamming"``, ``"jaccard"`` or ``"hellinger"`` (``"dot"`` is
  excluded), with no ``metric_kwds``; and
* ``n_neighbors`` is at least 2.

RINND honors ``n_jobs`` and, like the rest of the modern pipeline, gives
reproducible results with a fixed ``random_state`` while running in
parallel. Fitted models that used RINND support ``transform`` and
``update``. RINND search indexes cannot be pickled, so they are rebuilt
lazily the first time a reloaded model needs to query them.

