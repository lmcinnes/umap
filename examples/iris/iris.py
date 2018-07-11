from bokeh.plotting import figure, output_file, show
from bokeh.models import CategoricalColorMapper, ColumnDataSource
from bokeh.palettes import Category10

import umap
from sklearn.datasets import load_iris

iris = load_iris()
embedding = umap.UMAP(n_neighbors=50,
                      learning_rate=0.5,
                      init="random",
                      min_dist=0.001)\
                .fit_transform(iris.data)

output_file("iris.html")


targets = [str(d) for d in iris.target_names]

source = ColumnDataSource(dict(
    x=[e[0] for e in embedding],
    y=[e[1] for e in embedding],
    label=[targets[d] for d in iris.target]
))

cmap = CategoricalColorMapper(factors=targets, palette=Category10[10])

p = figure(title="Test UMAP on Iris dataset")
p.circle(x='x',
         y='y',
         source=source,
         color={"field": 'label', "transform": cmap},
         legend='label')

show(p)
