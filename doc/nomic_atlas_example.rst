UMAP of Text Embeddingswith Nomic Atlas
=======================

This example demonstrates how to use UMAP with `Nomic Atlas <https://docs.nomic.ai/atlas/embeddings-and-retrieval/guides/using-umap-with-atlas>`_ to create interactive maps of embeddings of natural language data. Nomic Atlas automatically generates embeddings for your data and allows you to explore large datasets in a web browser with an AI analyst, providing API access to your embeddings for integration into downstream applications.

First, ensure you have the necessary libraries installed:

.. code:: bash

    pip install umap-learn nomic pandas

Below is an example Python script that loads a sample dataset, uses UMAP parameters within Nomic Atlas to generate an embedding, and then creates an interactive map.

.. code:: python3

    from nomic import AtlasDataset
    from nomic.data_inference import ProjectionOptions
    import pandas as pd

    # Example data
    df = pd.read_csv("https://docs.nomic.ai/singapore_airlines_reviews.csv")

    dataset = AtlasDataset("example-text-dataset-airline-reviews")

    dataset.add_data(df)

    # model="umap" is how you specify UMAP for the projection.
    # You can adjust n_neighbors, min_dist, and n_epochs as you would with the UMAP library.
    atlas_map = dataset.create_index(
        indexed_field='text', # The field in your DataFrame to embed
        projection=ProjectionOptions(
          model="umap",
          n_neighbors=20,
          min_dist=0.01,
          n_epochs=200
      )
    )

    print(f"Explore your interactive map at: {atlas_map.map_link}")

You can adapt this script for your own datasets by:

1.  Changing the input data (e.g., loading your own CSV or DataFrame).
2.  Specifying the correct `indexed_field` that contains the text or data you want to embed.
3.  Tuning the UMAP parameters (`n_neighbors`, `min_dist`, `n_epochs`) within the `ProjectionOptions` to best suit your data and visualization needs.