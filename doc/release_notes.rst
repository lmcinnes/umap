Release Notes
=============

Some notes on new features in various releases

What's new in 0.6
-----------------

0.6 introduces a new nearest-neighbor, initialization and layout-optimization
pipeline. This is a transition release: it keeps the established (0.5.x)
behavior by default while making the new pipeline available for evaluation. A
future release will make the new pipeline -- recursive initialization with the
Adam optimizer -- the default.

Defaults, compatibility and migration
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* ``compatibility_layout`` now defaults to ``None``. When it is not set, UMAP
  uses the 0.5.x pipeline and emits a ``FutureWarning`` that the default will
  change to ``False``. Set ``compatibility_layout=True`` to keep the current
  behavior (and silence the warning), or ``compatibility_layout=False`` to use
  the new pipeline now. Selecting ``optimizer="adam"`` or
  ``optimizer="momentum"`` without setting ``compatibility_layout`` also opts
  in to the new pipeline. Combining ``compatibility_layout=True`` with a modern
  optimizer raises a ``ValueError``.
* With a fixed ``random_state``, ``compatibility_layout=True`` reproduces the
  final 0.5.x code line bit-for-bit -- embeddings, ``transform``,
  ``inverse_transform``, DensMAP radii and AlignedUMAP -- given the same
  platform and dependency versions. That reference includes the 0.5.x bug
  fixes listed below, so a few configurations differ from 0.5.12 itself; see
  :doc:`reproducibility`.
* ``init`` now defaults to ``"auto"``, which resolves to ``"spectral"`` under
  the compatibility layout and ``"recursive"`` otherwise. Resolved values are
  stored privately, so ``get_params`` and ``sklearn.base.clone`` round-trip
  the parameters as given.
* Models pickled with umap-learn 0.5.x can be loaded, and continue to use the
  compatibility pipeline for ``transform``, ``inverse_transform`` and
  ``update``.
* The public functions ``nearest_neighbors``, ``fuzzy_simplicial_set``,
  ``simplicial_set_embedding`` and the spectral layout functions keep their
  0.5.x behavior by default; pass ``compatibility_layout=False`` to use the
  new pipeline.
* AlignedUMAP accepts ``compatibility_layout`` with the same default and
  warning, and now has a full parameter docstring.

The new pipeline (``compatibility_layout=False``)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* New layout optimizers ``optimizer="adam"`` (the default for the new
  pipeline) and ``optimizer="momentum"``. The original immediate-update
  optimizer is available as ``optimizer="compatibility"``.
* Adam and momentum give reproducible results with a fixed ``random_state``
  while still optimizing in parallel, for fitting, ``transform`` and DensMAP.
* Optimizer selection is independent of DensMAP: combine ``densmap=True``
  with ``optimizer="adam"`` or ``optimizer="momentum"``. The development-only
  names ``densmap_adam``, ``densmap_momentum`` and ``standard`` are rejected.
* ``init="recursive"`` adds a hierarchical label-propagation initialization
  that recursively coarsens the fuzzy graph, lays out the small graph, and
  expands the result. ``recursive_coarsening_ratio`` (2, 3 or 4; 4 is fastest)
  controls how gradually this happens. ``coarse_layout_weights``,
  ``coarsen_on_layout_weights`` and ``recursive_init_kwds`` provide advanced
  control.
* ``init_rescale`` (``"isotropic"``, ``"per_axis"`` or ``"graph_edge"``) and
  ``init_scale_factor`` control how the initialization is centered and scaled
  before optimization.
* Hard-negative sampling controls for the modern kernels:
  ``negative_selection_range``, ``negative_sample_scale``,
  ``negative_sample_scale_adaptation_samples`` and
  ``exclude_graph_neighbors``. See :doc:`optimizers` for where each applies.
* Faster nearest-neighbor descent settings and a parallel sparse kernel for
  the fuzzy union/intersection.
* Optional `RINND <https://pypi.org/project/rinnd/>`_ backend for building the
  nearest-neighbor graph (``pip install umap-learn[rinnd]``), selected with
  ``nearest_neighbor_backend`` (``"auto"``, ``"rinnd"`` or
  ``"pynndescent"``). See :doc:`performance` for the requirements.

Other improvements
~~~~~~~~~~~~~~~~~~

* ``inverse_transform`` validates the embedding dimension. For queries outside
  the embedding's convex hull it warns and extrapolates from the nearest
  embedded point, instead of silently using an unrelated simplex.
* Fitting with the default ``unique=False`` no longer copies or reindexes the
  data.
* The input fingerprint used to recognise ``transform`` of the training data
  is computed with a parallel BLAKE2 hash.
* The input dtype is preserved in ``transform`` and ``fit_transform`` output.
* Lower peak memory in fit and transform, and faster component layout,
  breadth-first search, k-nearest-neighbor selection, DensMAP density
  computation and AlignedUMAP relation handling.
* Type checkers can use the bundled ``py.typed`` marker.
* UMAP now requires NumPy 2.0 or later and supports Python 3.10 to 3.14.

Bug fixes since 0.5.12
~~~~~~~~~~~~~~~~~~~~~~

* Layout optimization no longer uses fast-math settings that could produce
  NaN embeddings (#1228).
* ``smooth_knn_dist`` no longer gives an infinite sigma for points with a
  disconnected neighbor.
* Stable tie-breaking in ``fast_knn_indices`` is restored.
* Spectral initialization works for graphs with at most ``n_components + 1``
  nodes.
* ``transform`` and ``inverse_transform`` work when ``n_epochs`` is a list;
  ``transform`` matches the sparsity of the fitted data and accepts
  precomputed distances when fitting used nearest-neighbor descent.
* ``update`` initialisation no longer divides by zero.
* ``csr_unique`` handles sparse rows that all have the same number of
  non-zeros, and empty unions no longer mutate sparse inputs.
* Gradients are fixed for the Gaussian energy and symmetric KL metrics;
  Hellinger distances no longer produce NaN for proportional inputs;
  ``symmetric_kl`` no longer mutates its inputs; named special metrics respect
  their keyword arguments.
* AlignedUMAP supports per-model ``n_epochs``, and ``update`` no longer
  mutates the caller's relations or parameters.
* ParametricUMAP no longer produces NaN gradients under XLA.
* Fixes to ``trustworthiness_vector``, plot diagnostics for small datasets,
  and plotting with hex background colours.

What's new in 0.5
-----------------

* ParametricUMAP learns embeddings with neural networks.
* AlignedUMAP can align multiple embeddings using relations between datasets.
* DensMAP can preserve local density information in embeddings.
* UMAP now depends on PyNNDescent, but has faster more parallel performance as a result.
* UMAP now supports an ``update`` method to add new data and retrain.
* Various performance improvements and bug fixes.
* Additional plotting support, including text searching in interactive plots.
* Support for "maximal distances" in neighbor graphs.

What's new in 0.4
-----------------

* Inverse transform method. Generate points in the original space corresponding to points in embedded space. (Thanks to Joseph Courtney)
* Different embedding spaces. Support for embedding to a variety of different spaces other than Euclidean. (Thanks to Joseph Courtney)
* New metrics, including Hellinger distance for sparse count data.
* New discrete/label metrics, including hierarchical categories, counts, ordinal data, and string edit distance.
* Support for parallelism in neighbor search and layout optimization. (Thanks to Tom White)
* Support for alternative methods to handling duplicated data samples. (Thanks to John Healy)
* New plotting methods for fast and easy plots.
* Initial support for dataframe embedding -- still experimental, but worth trying.
* Support for transform methods with sparse data.
* Multithreading support when no random seed is set.


What's new in 0.3
-----------------

* Supervised and semi-supervised dimension reduction. Support for using labels or partial labels for dimension reduction.
* Transform method. Support for adding new unseen points to an existing embedding.
* Performance improvements.


What's new in 0.2
-----------------

* A new layout algorithm that handles large datasets (more) correctly.
* Performance improvements.