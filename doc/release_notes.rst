Release Notes
=============

Some notes on new features in various releases

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