UMAP API Guide
==============

UMAP provides three estimator classes: :class:`~umap.umap_.UMAP`,
:class:`~umap.aligned_umap.AlignedUMAP` for jointly embedding related
datasets, and :class:`~umap.parametric_umap.ParametricUMAP`, which inherits
from :class:`~umap.umap_.UMAP` and learns the embedding with a neural network.

UMAP
----

.. autoclass:: umap.umap_.UMAP
   :members:

AlignedUMAP
-----------

.. autoclass:: umap.aligned_umap.AlignedUMAP
   :members:

ParametricUMAP
--------------

.. autoclass:: umap.parametric_umap.ParametricUMAP
   :members:

A number of internal functions can also be accessed separately for more fine tuned work.

Useful Functions
----------------

.. automodule:: umap.umap_
   :members:
   :exclude-members: UMAP

Recursive initialization
------------------------

.. automodule:: umap.label_prop
   :members: recursive_init, label_propagation_init

Spectral initialization
-----------------------

.. automodule:: umap.spectral
   :members: spectral_layout, tswspectral_layout

UMATO
-----

UMATO is enabled with ``UMAP(umato=True)``; its parameters are listed with
:class:`~umap.umap_.UMAP`. The steps of its layout can also be called
separately.

.. automodule:: umap.umato

.. autofunction:: umap.umato.umato_embedding
.. autofunction:: umap.umato.default_n_hubs
.. autofunction:: umap.umato.select_hubs
.. autofunction:: umap.umato.expand_hubs
.. autofunction:: umap.umato.global_hub_layout
.. autofunction:: umap.umato.embed_from_hubs
.. autofunction:: umap.umato.local_graph
.. autofunction:: umap.umato.optimize_local_layout
.. autofunction:: umap.umato.place_outliers_in_groups
.. autofunction:: umap.umato.place_outliers
