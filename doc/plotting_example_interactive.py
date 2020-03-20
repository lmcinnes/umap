import sklearn.datasets
import pandas as pd
import numpy as np
import umap
import umap.plot

fmnist = sklearn.datasets.fetch_openml("Fashion-MNIST")

mapper = umap.UMAP().fit(fmnist.data[:30000])

hover_data = pd.DataFrame({"index": np.arange(30000), "label": fmnist.target[:30000]})
hover_data["item"] = hover_data.label.map(
    {
        "0": "T-shirt/top",
        "1": "Trouser",
        "2": "Pullover",
        "3": "Dress",
        "4": "Coat",
        "5": "Sandal",
        "6": "Shirt",
        "7": "Sneaker",
        "8": "Bag",
        "9": "Ankle Boot",
    }
)

umap.plot.output_file("plotting_interactive_example.html")

p = umap.plot.interactive(
    mapper, labels=fmnist.target[:30000], hover_data=hover_data, point_size=2
)
umap.plot.show(p)
