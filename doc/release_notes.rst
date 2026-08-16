Release Notes
=============

Some notes on new features in various releases

What's new in 0.6
-----------------

0.6 introduces a new initialization and layout-optimization stack. This is a
transition release: it preserves the established layout behavior by default
while making the new behavior available for evaluation. A future release will
make recursive initialization and the Adam optimizer the defaults.

* ``compatibility_layout=True`` is the default in this release. It selects the
  legacy nearest-neighbor and layout behavior, and emits a deprecation warning
  that its default will change in a future release. Set
  ``compatibility_layout=False`` to use the new layout path now.
* The new layout optimizers are ``optimizer="adam"`` and
  ``optimizer="momentum"``. The legacy immediate-update optimizer remains
  available as ``optimizer="compatibility"``. In the next release the default
  configuration will be recursive initialization with the Adam optimizer;
  compatibility mode will remain available but will no longer be enabled by
  default.
* Optimizer selection is now independent of DensMAP: use ``densmap=True`` with
  ``optimizer="adam"`` or ``optimizer="momentum"`` as appropriate. The old
  composite optimizer names ``densmap_adam`` and ``densmap_momentum`` are no
  longer accepted. Likewise, the historical ``standard`` optimizer name has
  been replaced by ``momentum``.
* ``init="recursive"`` adds a hierarchical label-propagation initialization.
  It recursively coarsens the fuzzy graph, lays out the smaller graph, and
  expands the result back to the full graph. The
  ``recursive_coarsening_ratio`` option controls how gradually the graph is
  coarsened and expanded (2, 3, or 4; 4 is the fastest).
* Adam and momentum support reproducible parallel optimization, including when
  ``random_state`` is fixed. Compatibility optimization must use a single
  thread with a fixed seed, because its immediate updates would otherwise race
  and make the result non-reproducible.
* Modern layout kernels also support controls for hard-negative sampling,
  including ``negative_selection_range`` and, for Euclidean layouts where
  applicable, ``negative_sample_scale``,
  ``negative_sample_scale_adaptation_samples``, and
  ``exclude_graph_neighbors``. These options do not apply to compatibility
  layout; see :doc:`optimizers` for the boundaries of each option.
* Inverse transforms now validate the embedding dimension. For queries outside
  the fitted embedding's convex hull, inverse transform now warns and uses the
  nearest embedded vertex as an extrapolation seed instead of silently using
  an unrelated simplex.
* Fitting with the default ``unique=False`` no longer performs unnecessary
  data uniquing, improving fit performance and avoiding needless index
  remapping.

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