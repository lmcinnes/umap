import numpy as np
import pandas as pd

import datashader as ds
import datashader.transfer_functions as tf
import matplotlib.pyplot as plt
import colorcet
import matplotlib.colors

import sklearn.decomposition
import sklearn.cluster

fire_cmap = matplotlib.colors.LinearSegmentedColormap.from_list('fire', colorcet.fire)
darkblue_cmap = matplotlib.colors.LinearSegmentedColormap.from_list('darkblue', colorcet.kbc)
darkgreen_cmap = matplotlib.colors.LinearSegmentedColormap.from_list('darkgreen', colorcet.kgy)
darkred_cmap = matplotlib.colors.LinearSegmentedColormap.from_list('darkred',
                                                                   colors=colorcet.linear_kry_5_95_c72[:192],
                                                                   N=256)
plt.register_cmap('fire', fire_cmap)
plt.register_cmap('darkblue', darkblue_cmap)
plt.register_cmap('darkgreen', darkgreen_cmap)
plt.register_cmap('darkred', darkgreen_cmap)


def _to_hex(arr):
    return [matplotlib.colors.to_hex(c) for c in arr]


_themes = {
    'fire': {'cmap': 'fire', 'color_key_cmap': 'rainbow', 'background': 'black'},
    'viridis': {'cmap': 'viridis', 'color_key_cmap': 'Spectral', 'background': 'black'},
    'inferno': {'cmap': 'inferno', 'color_key_cmap': 'Spectral', 'background': 'black'},
    'blue': {'cmap': 'Blues', 'color_key_cmap': 'tab20', 'background': 'white'},
    'red': {'cmap': 'Reds', 'color_key_cmap': 'tab20b', 'background': 'white'},
    'green': {'cmap': 'Greens', 'color_key_cmap': 'tab20c', 'background': 'white'},
    'darkblue': {'cmap': 'darkblue', 'color_key_cmap': 'rainbow', 'background': 'black'},
    'darkred': {'cmap': 'darkred', 'color_key_cmap': 'rainbow', 'background': 'black'},
    'darkgreen': {'cmap': 'darkgreen', 'color_key_cmap': 'rainbow', 'background': 'black'},
}


def _datashade_points(
        points,
        labels=None,
        values=None,
        cmap='Blues',
        color_key=None,
        color_key_cmap='Spectral',
        background='white',
        width=800,
        height=800,
):
    extent = np.round(np.max(np.abs(points)) * 1.1)
    canvas = ds.Canvas(plot_width=width,
                       plot_height=height,
                       x_range=(-extent, extent),
                       y_range=(-extent, extent))
    data = pd.DataFrame(points, columns=('x', 'y'))
    if labels is not None:
        if labels.shape[0] != points.shape[0]:
            raise ValueError('Labels must have a label for '
                             'each sample (size mismatch: {} {})'.format(labels.shape[0],
                                                                         points.shape[0]))

        data['label'] = pd.Categorical(labels)
        aggregation = canvas.points(data, 'x', 'y', agg=ds.count_cat('label'))
        if color_key is None and color_key_cmap is None:
            result = tf.shade(aggregation, how='eq_hist')
        elif color_key is None:
            unique_labels = np.unique(labels)
            num_labels = unique_labels.shape[0]
            color_key = _to_hex(plt.get_cmap(color_key_cmap)(np.linspace(0, 1, num_labels)))
            result = tf.shade(aggregation, color_key=color_key, how='eq_hist')
        else:
            result = tf.shade(aggregation, color_key=color_key, how='eq_hist')

    elif values is not None:
        if values.shape[0] != points.shape[0]:
            raise ValueError('Values must have a value for '
                             'each sample (size mismatch: {} {})'.format(values.shape[0],
                                                                         points.shape[0]))
        min_val, max_val = np.min(values), np.max(values)
        bin_size = (max_val - min_val) / 256.0
        data['val_cat'] = pd.Categorical(np.round((values - min_val) / bin_size).astype(np.int16))
        aggregation = canvas.points(data, 'x', 'y', agg=ds.count_cat('val_cat'))
        color_key = to_hex(plt.get_cmap(cmap)(np.linspace(0, 1, 256)))
        result = tf.shade(aggregation, color_key=color_key, how='eq_hist')

    else:
        aggregation = canvas.points(data, 'x', 'y', agg=ds.count())
        result = tf.shade(aggregation, cmap=plt.get_cmap(cmap))

    result = tf.set_background(result, background)

    return result


def _matplotlib_points(
        points,
        labels=None,
        values=None,
        cmap='Blues',
        color_key=None,
        color_key_cmap='Spectral',
        background='white',
        width=800,
        height=800,
):
    point_size = 20.0 / np.log10(points.shape[0])
    dpi = plt.rcParams['figure.dpi']
    fig = plt.figure(figsize=(width / dpi, height / dpi))
    ax = fig.add_subplot(111)
    ax.set_facecolor(background)

    if labels is not None:
        if labels.shape[0] != points.shape[0]:
            raise ValueError('Labels must have a label for '
                             'each sample (size mismatch: {} {})'.format(labels.shape[0],
                                                                         points.shape[0]))
        if color_key is None:
            unique_labels = np.unique(labels)
            num_labels = unique_labels.shape[0]
            color_key = plt.get_cmap(color_key_cmap)(np.linspace(0, 1, num_labels))

        if isinstance(color_key, dict):
            colors = pd.Series(labels).map(color_key)
        else:
            unique_labels = np.unique(labels)
            if len(color_key) < unique_labels.shape[0]:
                raise ValueError('Color key must have enough colors for the number of labels')

            new_color_key = {k: color_key[i] for i, k in enumerate(unique_labels)}
            colors = pd.Series(labels).map(new_color_key)

        ax.scatter(points[:, 0], points[:, 1], s=point_size, c=colors)
    elif values is not None:
        if values.shape[0] != points.shape[0]:
            raise ValueError('Values must have a value for '
                             'each sample (size mismatch: {} {})'.format(values.shape[0],
                                                                         points.shape[0]))
        ax.scatter(points[:, 0], points[:, 1], s=point_size, c=values, cmap=cmap)

    else:

        color = plt.get_cmap(cmap)(0.5)
        ax.scatter(points[:, 0], points[:, 1], s=point_size, c=color)

    return ax


def points(
        umap_object,
        labels=None,
        values=None,
        theme=None,
        cmap='Blues',
        color_key=None,
        color_key_cmap='Spectral',
        background='white',
        width=800,
        height=800,
):
    if not hasattr(umap_object, 'embedding_'):
        raise ValueError('UMAP object must perform fit on data before it can be visualized')

    if theme is not None:
        cmap = _themes[theme]['cmap']
        color_key_cmap = _themes[theme]['color_key_cmap']
        background = _themes[theme]['background']

    if labels is not None and values is not None:
        raise ValueError('Conflicting options; only one of labels or values should be set')

    points = umap_object.embedding_

    if points.shape[1] != 2:
        raise ValueError('Plotting is currently only implemented for 2D embeddings')

    if points.shape[0] <= 10000:
        return _matplotlib_points(points, labels, values, cmap, color_key,
                                  color_key_cmap, background, width, height)
    else:
        return _datashade_points(points, labels, values, cmap, color_key,
                                 color_key_cmap, background, width, height)




def connectivity(umap_object):
    pass


def diagnostic(umap_object):
    pass


def interactive(umap_object):
    pass
