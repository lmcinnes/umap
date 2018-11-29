Release Notes
=============

Some notes on new features in various releases

What's new in 0.4
-----------------

* Inverse transform method. Generate points in the original space corresponding to points in embedded space. (Thanks for Joseph Courtney)
* Different embedding spaces. Support for embedding to a variety of different spaces other than Euclidean. (Thanks for Joseph Courtney)
* New metrics, including Hellinger distance for sparse count data.
* New discrete/label metrics, including hierarchical categories, counts, ordinal data, and string edit distance.
* Initial support for dataframe embedding -- still experimental, but worth trying.


What's new in 0.3
-----------------

* Supervised and semi-supervised dimension reduction. Support for using labels or partial labels for dimension reduction.
* Transform method. Support for adding new unseen points to an existing embedding.
* Performance improvements.


What's new in 0.2
-----------------

* A new layout algorithm that handles large datasets (more) correctly.
* Performance improvements.