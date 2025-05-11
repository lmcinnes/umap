from nomic import AtlasDataset
from nomic.data_inference import ProjectionOptions
import pandas as pd

# Example data
df = pd.read_csv("https://docs.nomic.ai/singapore_airlines_reviews.csv")

dataset = AtlasDataset("example-dataset-airline-reviews")

dataset.add_data(df)

atlas_map = dataset.create_index(
    indexed_field="text",
    projection=ProjectionOptions(
        model="umap", n_neighbors=20, min_dist=0.01, n_epochs=200
    ),
)
