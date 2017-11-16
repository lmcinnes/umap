from bokeh.plotting import figure, output_file, show
from bokeh.models import CategoricalColorMapper, ColumnDataSource
from bokeh.palettes import Category10

import umap
from sklearn.datasets import load_digits

digits = load_digits()
embedding = umap.UMAP().fit_transform(digits.data)

output_file("digits.html")

targets = [str(d) for d in digits.target_names]

source = ColumnDataSource(dict(
    x = [e[0] for e in embedding],
    y = [e[1] for e in embedding],
    label = [targets[d] for d in digits.target]
))

cmap = CategoricalColorMapper(factors=targets, palette=Category10[10])

p = figure(title="test umap")
p.circle(x='x',
         y='y',
         source=source,
         color={"field": 'label', "transform": cmap},
         legend='label')

show(p)
