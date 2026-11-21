.. _interpreting_embeddings:

Interpreting UMAP embeddings
============================

In :doc:`basic_usage` tutorial, we plotted penguins using four body 
measurements and colored each point by species. Let's look at those 
penguins again. It seems that the green Gentoo points form a group of 
their own. Now, the question is: which species seems closer to Gentoo?

.. figure:: images/interpretation_penguin_centroids.png
   :alt: The solid line from Gentoo to Adelie is shorter than the dashed line from Gentoo to Chinstrap. The order reverses in the four body measurements.

   Each dot refers to a penguin, and diamonds are the mean positions of the three species.
   The lines connect the the centroid of Gentoo to the centroids of the other two species. 
   The solid line is shorter than the dashed line in the UMAP layout, but the corresponding 
   distances in the four-dimensional measurements are reversed. 


Two questions about the Gentoo penguins
---------------------------------------

Looking at the above UMAP layout, we can ask a few questions about
the Gentoo penguines. One intuitive question will be:

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

**Q2**. **Do Adelie penguins have similar charactersistics to Gentoo penguins than Chinstrap penguins?**
*The solid line from Gentoo to Adelie is shorter than the dashed line from Gentoo to Chinstrap in the 
UMAP layout. Does this mean that Adelie penguins are more similar to Gentoo penguins than Chinstrap penguins
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

In this examaple, we can see the original data directly. Here, our input is 
**two-dimensional** (leftmost panel), where we can see three apparent clusters of points (blue, yellow and red).
The other three panels show the result of applying UMAP, densMAP, and UMATO to this two-dimensional input.
Each method produces a different layout of the same points, illustrating how different dimensionality reduction 
techniques can emphasize different aspects of the data.


.. figure:: images/interpretation_layouts.png
   :alt: Four panels in one row show the 2D input, UMAP, densMAP and UMATO. The small yellow group spreads out in UMAP and is compact in densMAP. UMATO also places the blue and yellow groups on opposite sides below the red group.

   The original 2D input data (leftmost) and the results of applying three dimensionality reduction methods (UMAP, densMAP, UMATO) to the data. 
   Colors and black symbols track the same observations across panels.

Let's first look at the **UMAP** embedding. You can see that three clusters are still visible, but (1) global arrangement of the clusters has changed, and (2) the small yellow and blue clusters have spread out. The distance between the two black dots that almost touch in the original data has increased in the UMAP embedding. This example illustrates that UMAP can help identify clusters, but their relative positions and density may not reflect the original data -- resonating with the earlier observation about Gentoo penguins. 

Now look at **densMAP**; here, the global arrangement of the clusters is still different from the original data, but now the density of clusters is better preserved. The small yellow cluster is compact, while the red cluster occupies more space. This shows that densMAP can help preserve density information, which also aligns with the purpose of the densMAP method (which can be enabled in this library by passing ``densmap=True`` to the UMAP constructor). 

Finally, in **UMATO**, the global arrangement of clusters is more similar to the original data, while the density of clusters is somewhat distorted (e.g., red clusters is not uniformly distributed). This shows that UMATO can help preserve global arrangement of clusters, which is useful when the relative positions of clusters are important.

This example shows that different dimensionality reduction methods can emphasize different aspects of the data, and the choice of method should depend on the specific questions you want to answer. For example, if you want to count the number of clusters, UMAP will be a great choice. However, if you want to investigate the density of clusters, densMAP will be a better choice. Now let's look at why such differences arise.


Why do these differences arise?
-------------------------------

UMAP focuses on neighbors
~~~~~~~~~~~~~~~~~~~~~~~~~

UMAP tries to place observations that are neighbors in the input close
together in the embedding. If two points have no direct neighborhood
connection, they can end up far apart even when their original distance
was fairly small (See :doc:`how_umap_works` for detail). Therefore, UMAP
well preserves local neighborhood structure, making it useful for identifying
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
``densmap=True`` to the `umap` constructor.
Please refer to the :doc:`densmap_demo` tutorial for an working example.

.. code-block:: python

   import umap

   embedding = umap.UMAP(densmap=True, random_state=42).fit_transform(X)

When the arrangement of several groups matters
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

UMATO accounts for global arrangement of clusters by selecting representative points, called hubs,
and arranging them first in the embedding without neighborhood-based approximation. 
These hubs thus form a skeletal layout of the embeding; UMATO then places the
remaining observations (following the same process of UMAP) and refines their
neighborhoods around that skeletal layout.
This makes UMATO worth exploring how several group, or clusters, are positioned relative to one another.
Note that UMATO is available in the separate `umato package <https://github.com/hyungkwonko/umato>`_, where 
we plan to integrate it into this library in near future.

What can I read from my own UMAP plot?
--------------------------------------

Pick a single penguin and you might ask which other birds resemble it.
Step back and the space between the colored groups might make you wonder
which species resemble each other. Start with the question
you are trying to answer.

.. figure:: images/interpretation_tasks.png
   :alt: The first row illustrates neighbors, isolated points and groups. The second illustrates distances between points and groups, local density and class separability.

   The first row helps you select observations to examine together.
   The second asks how those observations relate in the input data.

Patterns to look for in the embedding
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Which cases resemble this one?** Select a penguin and inspect the
records for several nearby dots. Do they have similar bills, similar
body masses, or both? If the identity of a particular neighbor matters,
list the closest observations in the original four features and check
whether that individual appears. Finding that all 15 neighbors are
Gentoo establishes the species of the neighbors. It leaves open whether
the *same 15 individuals* are neighbors in the plot.

**What makes this isolated point worth a look?** Suppose an orange
Chinstrap dot sits next to a blue Adelie group. Compare its measurements
with those of other Chinstraps. One feature might be unusual, or a rare
combination might be made up of individually common values. If you find
many similar Chinstraps in the input, describe the point as isolated
in this view. Calling it an outlier in the data would require further
evidence.

**What do the points in this group share?** Select several points
from the middle and several from the edge of an island. Does the common
feature you found in the middle extend to the edge? Compare observations
on opposite sides of a visible gap. If many are neighbors in the input,
a continuous change may have appeared as a split. Counting the islands
would then give a misleading count of groups.

Relationships to check in the input data
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**How different are two observations?** Suppose the line from A to B
looks twice as long as the line from A to C. To check the ordering or
the ratio, calculate both distances using the same input features,
scaling and metric. Then inspect individual features if you want to
explain the difference. Check how much bill length and body mass each contributed.

**Which groups are most similar?** In the penguin example the question
concerned the *average* member of each species, so we compared their
centroids in the four measurements. Two groups can have nearby centroids
even when individual members differ widely. If you care how different
a typical cross-group pair is, take one observation from each group and
examine the distribution of distances across such pairs.

**Which group is denser?** Recall how the small yellow group spread out
in UMAP. To compare local crowding in your own data, find the distance
from each observation to its tenth nearest other observation in the
input. With the same features, metric and neighborhood size, a smaller
radius means more observations fit nearby. If you want to know which
group contains more observations, count the records. A large sample and
a tightly packed neighborhood answer different questions.

**How well do known classes separate?** Select points where two colors
meet and inspect whether their input-space neighbors have mixed labels
too. In the penguin example, labels were added after fitting. If labels
were used to fit a :doc:`supervised embedding <supervised>`, their
separation in the plot also reflects that supervision. If the question
is whether you can predict the species of a new bird, split the data
before fitting the pipeline and evaluate predictions on held-out birds.

Further reading
---------------

* Hyeon Jeon, Jeongin Park, Sungbok Shin and Jinwook Seo.
  `Stop Misusing t-SNE and UMAP for Visual Analytics <https://arxiv.org/abs/2506.08725v3>`_.
  IEEE Transactions on Visualization and Computer Graphics (2026)
* Dylan Cashman, Mark Keller, Hyeon Jeon, Bum Chul Kwon and Qianwen
  Wang. `A Critical Analysis of the Usage of Dimensionality
  Reduction in Four Domains <https://doi.org/10.1109/TVCG.2025.3567989>`_.
  IEEE Transactions on Visualization and Computer Graphics  (2025)
* Ashwin Narayan, Bonnie Berger, and Hyunghoon Cho, `Assessing single-cell transcriptomic variability through density-preserving data visualization`. Nat Biotechnology (2021)
* Hyeon Jeon, Kwon Ko, Soohyun Lee, Jake Hyun, Taehyun Yang, Gyehun Go, Jaemin Jo, and Jinwook Seo. "UMATO: Bridging local and global structures for reliable visual analytics with dimensionality reduction." IEEE Transactions on Visualization and Computer Graphics (2025).