import umap
from sklearn.datasets import make_blobs

data, _ = make_blobs(1000)
embedding = umap.UMAP().fit_transform(data)

