"""
UMAP on the Fashion MNIST Digits dataset using Datashader
---------------------------------------------------------

As per the title.
"""
import umap
import numpy as np
import pandas as pd
import requests
import datashader as ds
import datashader.transfer_functions as tf
import matplotlib.pyplot as plt
import seaborn as sns

sns.set(context="paper", style="white")

csv_data = requests.get('https://www.openml.org/data/get_csv/18238735/phpnBqZGZ')
with open('fashion-mnist.csv', 'w') as f:
    f.write(csv_data.text)
source_df = pd.read_csv('fashion-mnist.csv')

data = source_df.iloc[:, :784].values.astype(np.float32)
target = source_df['class'].values

reducer = umap.UMAP(random_state=42)
embedding = reducer.fit_transform(data)

df = pd.DataFrame(embedding, columns=('x', 'y'))
df['digit'] = target

pal = sns.color_palette('Spectral', 10)
color_key = dict(zip(enumerate(pal)))

cvs = ds.Canvas(plot_width=800, plot_height=800)
agg = cvs.points(df, 'x', 'y', ds.count_cat('digit'))
img = tf.shade(agg, color_key=color_key, how='eq_hist')

fig, ax = plt.subplots(figsize=(12, 12))
plt.imshow(img)
plt.setp(ax, xticks=[], yticks=[])
plt.title("Fashion MNIST data embedded into two dimensions by UMAP", fontsize=18)

plt.show()
