"""
UMAP on the Fashion MNIST Digits dataset using Datashader
---------------------------------------------------------

As per the title.
"""
import umap
import numpy as np
import pandas as pd
import requests
import os
import datashader as ds
import datashader.utils as util
import datashader.transfer_functions as tf
import matplotlib.pyplot as plt
import seaborn as sns

sns.set(context="paper", style="white")

if not os.path.isfile('fashion-mnist.csv'):
    csv_data = requests.get('https://www.openml.org/data/get_csv/18238735/phpnBqZGZ')
    with open('fashion-mnist.csv', 'w') as f:
        f.write(csv_data.text)
source_df = pd.read_csv('fashion-mnist.csv')

data = source_df.iloc[:, :784].values.astype(np.float32)
target = source_df['class'].values

pal = sns.color_palette('Spectral', 10)
# color_key = {str(d):c for d,c in enumerate(pal)}
color_key = {'0': 'darkred', '1': 'red', '2': 'orange',
             '3': 'yellow', '4': 'chartreuse', '5': 'lime',
             '6': 'green', '7': 'aqua', '8': 'blue', '9': 'purple'}

reducer = umap.UMAP(random_state=42)
embedding = reducer.fit_transform(data)

df = pd.DataFrame(embedding, columns=('x', 'y'))
df['class'] = pd.Series([str(x) for x in target], dtype="category")

cvs = ds.Canvas(plot_width=400, plot_height=400)
agg = cvs.points(df, 'x', 'y', ds.count_cat('class'))
img = tf.shade(agg, color_key=color_key, how='eq_hist')

util.export_image(img, filename='fashion-mnist', background='black')

image = plt.imread('fashion-mnist.png')

fig, ax = plt.subplots(figsize=(6, 6))
plt.imshow(image)
plt.setp(ax, xticks=[], yticks=[])
plt.title("Fashion MNIST data embedded into two dimensions by UMAP", fontsize=14)

plt.show()
