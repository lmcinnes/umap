.. _interpreting_embeddings:

Interpreting UMAP embeddings
============================

In the :doc:`basic_usage` tutorial, we plotted penguins using four body
measurements and colored each point by species. Let's look at those
penguins again. It seems that the green Gentoo points form a group of
their own. Now, the question is: which species seems closer to Gentoo?

.. figure:: images/interpretation_penguin_centroids.png
   :alt: The solid line from Gentoo to Adelie is shorter than the dashed line from Gentoo to Chinstrap. The order reverses in the four body measurements.

   Each dot refers to a penguin, and diamonds are the mean positions of the three species.
   The lines connect the centroid of Gentoo to the centroids of the other two species.
   The solid line is shorter than the dashed line in the UMAP layout, but the corresponding
   distances in the four-dimensional measurements are reversed.


Two questions about the Gentoo penguins
---------------------------------------

Looking at the above UMAP layout, we can ask a few questions about
the Gentoo penguins. One intuitive question is:

**Q1**. **Do Gentoo penguins form a distinct, well-condensed group?**
*The green points are tightly packed and separated from the other
species. Does this mean that Gentoo penguins have similar characteristics
that distinguish them from other species?*

**A1**. *Yes, the Gentoo penguins are similar to each other in the four body measurements,
which makes them well clustered in both the original data and the UMAP layout. When we
check the nearest neighbors of each Gentoo penguin in the original four-dimensional measurements,
we find that all 15 nearest neighbors are also Gentoo penguins, without exception. UMAP
well preserves this neighborhood structure, making the Gentoo penguins appear as a distinct
cluster in the 2D embedding.*

However, the answer to the next question, which also seems intuitive,
is not as straightforward.

**Q2**. **Are Adelie penguins more similar to Gentoo penguins than Chinstrap penguins are?**
*The solid line from Gentoo to Adelie is shorter than the dashed line from Gentoo to Chinstrap in the
UMAP layout. Does this mean that Adelie penguins are more similar to Gentoo penguins than Chinstrap penguins are
in the original four-dimensional measurements?*

**A2**. *Not necessarily. The distances in the UMAP layout do not directly correspond to the distances
in the original four-dimensional measurements. In fact, when we compute the Euclidean distances between
the centroids of each species in the original four-dimensional measurements, we find that Chinstrap
penguins are actually closer to Gentoo penguins than Adelie penguins (2.88 vs 3.49). This shows that
UMAP can distort distances between groups, and we should be cautious about interpreting distances in
the embedding as reflecting true similarities in the original data.*


.. list-table:: Euclidean distances between species centroids
   :header-rows: 1
   :widths: 42 29 29

   * - Species pair
     - UMAP layout (2D)
     - Body measurements (4D)
   * - Gentoo and Adelie
     - **13.15**
     - 3.49
   * - Gentoo and Chinstrap
     - 18.17
     - **2.88**
   * - Adelie and Chinstrap
     - 5.18
     - 1.88

So, this example illustrates that UMAP is useful for identifying clusters and local neighborhoods, but distances between clusters in the embedding may not reflect true distances in the original data. Let's explore this further by projecting the 2D input data into 2D again using UMAP and alternative methods.

What happens if the data are already two-dimensional?
-----------------------------------------------------

In this example, we can see the original data directly. Here, our input is
**two-dimensional** (leftmost panel), where we can see three apparent clusters of points (blue, yellow and red).
The other three panels show the result of applying UMAP, densMAP, and UMATO to this two-dimensional input.
Each method produces a different layout of the same points, illustrating how different dimensionality reduction
techniques can emphasize different aspects of the data.


.. figure:: images/interpretation_layouts.png
   :alt: Four panels in one row show the 2D input, UMAP, densMAP and UMATO. The small yellow group spreads out in UMAP and is compact in densMAP. UMATO also places the blue and yellow groups on opposite sides below the red group.

   The original 2D input data (leftmost) and the results of applying three dimensionality reduction methods (UMAP, densMAP, UMATO) to the data.
   Colors and black symbols track the same observations across panels.

Let's first look at the **UMAP** embedding. You can see that three clusters are still visible, but (1) the global arrangement of the clusters has changed, and (2) the small yellow and blue clusters have spread out. The distance between the two black dots that almost touch in the original data has increased in the UMAP embedding. This example illustrates that UMAP can help identify clusters, but their relative positions and density may not reflect the original data -- resonating with the earlier observation about Gentoo penguins.

Now look at **densMAP**; here, the global arrangement of the clusters is still different from the original data, but now the density of clusters is better preserved. The small yellow cluster is compact, while the red cluster occupies more space. This shows that densMAP can help preserve density information, which also aligns with the purpose of the densMAP method (which can be enabled in this library by passing ``densmap=True`` to the UMAP constructor).

Finally, in **UMATO**, the global arrangement of clusters is more similar to the original data, while the density of clusters is somewhat distorted (e.g., the red cluster is not uniformly distributed). This shows that UMATO can help preserve global arrangement of clusters, which is useful when the relative positions of clusters are important.

This example shows that different dimensionality reduction methods can emphasize different aspects of the data, and the choice of method should depend on the specific questions you want to answer. For example, if you want to count the number of clusters, UMAP will be a great choice. However, if you want to investigate the density of clusters, densMAP will be a better choice. Now let's look at why such differences arise.


Why do these differences arise?
-------------------------------

UMAP focuses on neighbors
~~~~~~~~~~~~~~~~~~~~~~~~~

UMAP tries to place observations that are neighbors in the input close
together in the embedding. If two points have no direct neighborhood
connection, they can end up far apart even when their original distance
was fairly small (see :doc:`how_umap_works` for details). Therefore, UMAP
preserves local neighborhood structure well, making it useful for identifying
clusters and local relationships. However, it does not guarantee that distances
between clusters in the embedding reflect true distances in the original data, as
all non-neighbor points can be placed arbitrarily far apart.

When relative density matters
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

densMAP takes into account how far each point had to reach to find
its neighbors in the input. It encourages neighborhoods that were
relatively tight in the data to also stay relatively tight in the plot.
Therefore, densMAP is useful for investigating the density of clusters.

Note that **densMAP is available in this library.** To try it, pass
``densmap=True`` to the ``UMAP`` constructor.
Please refer to the :doc:`densmap_demo` tutorial for a working example.

.. code-block:: python

   import umap

   embedding = umap.UMAP(densmap=True, random_state=42).fit_transform(X)

When the arrangement of several groups matters
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

UMATO accounts for the global arrangement of clusters by selecting representative points, called hubs,
and arranging them first in the embedding without neighborhood-based approximation.
These hubs thus form a skeletal layout of the embedding; UMATO then places the
remaining observations (following the same process as UMAP) and refines their
neighborhoods around that skeletal layout.
This makes UMATO worth exploring when you want to see how several groups, or clusters, are positioned relative to one another.
Note that UMATO is available in the separate `umato package <https://github.com/hyungkwonko/umato>`_, and
we plan to integrate it into this library in the near future.

What can I read from my own UMAP plot?
--------------------------------------

Now let's detail how to interpret your own UMAP plot. The following
figure illustrates common tasks and questions that arise when investigating an embedding,
regardless of whether UMAP or another dimensionality reduction method was used.

.. figure:: images/interpretation_tasks.png
   :alt: The first row illustrates neighborhood, outlier and cluster identification. The second illustrates point distance, cluster distance, cluster density and class separability investigation.

   The tasks in the first row can be appropriately answered using UMAP embeddings.
   However, for the tasks in the second row, UMAP embeddings may not be appropriate, and you should check the original data to answer those questions.


The following are descriptions of the tasks illustrated in the figure above, along with the reasoning on why UMAP embeddings may or may not be appropriate for answering them.

Patterns that can be faithfully read from the UMAP embedding
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Neighborhood identification.** This task finds the data points that
are similar to a given point by looking at the points right next to it
in the embedding. UMAP is built to keep points that are neighbors in the
input data close together in the embedding. Points that sit next to each
other in the plot are therefore usually similar in the original data
too. This is exactly what UMAP optimizes, so you can trust the plot for
this task.

**Outlier identification.** This task finds outliers, that is, points
that do not belong to any cluster. UMAP places neighbors close together
and keeps non-neighbors apart, so a point that is not a neighbor of the
points in a cluster ends up clearly separated from that cluster. Points
that sit apart from the clusters in the plot are therefore good
candidates for outliers.

**Cluster identification.** This task finds clusters, that is, sets of
points that belong together. For example, you may count the clusters or
select one of them to look at it in detail. Points in the same cluster
are neighbors of one another, and UMAP keeps neighbors close and
non-neighbors apart. A cluster in the input data therefore shows up as a
clearly visible cluster in the embedding. You can trust the plot for
finding clusters and for seeing which points belong to the same cluster.

Patterns that require checking in the input data
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Point distance investigation.** This task reads the distance between
two points in the embedding as a measure of how different they are, for
example to decide whether A is closer to B than to C. UMAP only tries to
keep neighbors close. Points that are not neighbors are simply pushed
apart, and UMAP does not control by how much. The distance between two
far-apart points in the plot can therefore be much larger or smaller
than their real distance. To compare distances, compute them in the
input data.

**Cluster distance investigation.** This task reads the distance between
clusters in the embedding as a measure of how similar they are, for
example to decide which of two clusters is closer to a third one. Points
in different clusters are mostly not neighbors, so the data do not
decide the size of the gap between clusters. The gaps and the placement
of the clusters can also change with the random seed and the parameters.
To compare clusters, measure the distance between them in the input
data, for example between the cluster means.

**Cluster density investigation.** This task compares how tightly packed
different clusters are, to see how much the points within each cluster
vary. UMAP looks only at the neighbors of each point and rescales
distances according to how far away those neighbors are. As a result,
tight clusters and loose clusters end up looking about equally crowded.
A cluster that looks compact in the plot may be spread out in the input
data, and the reverse can also happen. Please check densities in
the input data, for example with the distance from each point to its
k-th nearest neighbor. You can also use densMAP, which is designed to
keep this information.

**Class separability investigation.** This task looks at how clearly
labeled classes, shown as colors, are separated from each other in the
embedding. The plot does show whether two classes are mixed with each
other. However, UMAP pushes non-neighbors apart, so classes can look more
cleanly separated than they really are. How far apart the colors are in
the plot therefore does not tell you how separable the classes are. To
measure separability directly, use the input data, for example by checking
how well a classifier predicts the labels of held-out points.

Further reading
---------------

* Hyeon Jeon, Jeongin Park, Sungbok Shin and Jinwook Seo.
  `Stop Misusing t-SNE and UMAP for Visual Analytics <https://arxiv.org/abs/2506.08725v3>`_.
  IEEE Transactions on Visualization and Computer Graphics (2026).
* Dylan Cashman, Mark Keller, Hyeon Jeon, Bum Chul Kwon and Qianwen
  Wang. `A Critical Analysis of the Usage of Dimensionality
  Reduction in Four Domains <https://doi.org/10.1109/TVCG.2025.3567989>`_.
  IEEE Transactions on Visualization and Computer Graphics (2025).
* Ashwin Narayan, Bonnie Berger, and Hyunghoon Cho, `Assessing single-cell transcriptomic variability through density-preserving data visualization`. Nature Biotechnology (2021).
* Hyeon Jeon, Kwon Ko, Soohyun Lee, Jake Hyun, Taehyun Yang, Gyehun Go, Jaemin Jo, and Jinwook Seo. `UMATO: Bridging local and global structures for reliable visual analytics with dimensionality reduction.` IEEE Transactions on Visualization and Computer Graphics (2025).