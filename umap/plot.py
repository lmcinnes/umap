import numpy as np
import pandas as pd
import numba

import datashader as ds
import datashader.transfer_functions as tf
import datashader.bundling as bd
import matplotlib.pyplot as plt
import colorcet
import matplotlib.colors

import sklearn.decomposition
import sklearn.cluster
import sklearn.neighbors

from umap.nndescent import initialise_search, initialized_nnd_search
from umap.utils import deheap_sort, submatrix

from warnings import warn

fire_cmap = matplotlib.colors.LinearSegmentedColormap.from_list('fire', colorcet.fire)
darkblue_cmap = matplotlib.colors.LinearSegmentedColormap.from_list('darkblue', colorcet.kbc)
darkgreen_cmap = matplotlib.colors.LinearSegmentedColormap.from_list('darkgreen', colorcet.kgy)
darkred_cmap = matplotlib.colors.LinearSegmentedColormap.from_list('darkred',
                                                                   colors=colorcet.linear_kry_5_95_c72[:192],
                                                                   N=256)
darkpurple_cmap = matplotlib.colors.LinearSegmentedColormap.from_list('darkpurple',
                                                                      colorcet.linear_bmw_5_95_c89)

plt.register_cmap('fire', fire_cmap)
plt.register_cmap('darkblue', darkblue_cmap)
plt.register_cmap('darkgreen', darkgreen_cmap)
plt.register_cmap('darkred', darkred_cmap)
plt.register_cmap('darkpurple', darkpurple_cmap)


def _to_hex(arr):
    return [matplotlib.colors.to_hex(c) for c in arr]


_themes = {
    'fire': {
        'cmap': 'fire',
        'color_key_cmap': 'rainbow',
        'background': 'black',
        'edge_cmap': 'Blues_r',
    },
    'viridis': {
        'cmap': 'viridis',
        'color_key_cmap': 'Spectral',
        'background': 'black',
        'edge_cmap': 'gray',
    },
    'inferno': {
        'cmap': 'inferno',
        'color_key_cmap': 'Spectral',
        'background': 'black',
        'edge_cmap': 'gray',
    },
    'blue': {
        'cmap': 'Blues',
        'color_key_cmap': 'tab20',
        'background': 'white',
        'edge_cmap': 'gray_r',
    },
    'red': {
        'cmap': 'Reds',
        'color_key_cmap': 'tab20b',
        'background': 'white',
        'edge_cmap': 'gray_r',
    },
    'green': {
        'cmap': 'Greens',
        'color_key_cmap': 'tab20c',
        'background': 'white',
        'edge_cmap': 'gray_r',
    },
    'darkblue': {
        'cmap': 'darkblue',
        'color_key_cmap': 'rainbow',
        'background': 'black',
        'edge_cmap': 'darkred',
    },
    'darkred': {
        'cmap': 'darkred',
        'color_key_cmap': 'rainbow',
        'background': 'black',
        'edge_cmap': 'darkblue',
    },
    'darkgreen': {
        'cmap': 'darkgreen',
        'color_key_cmap': 'rainbow',
        'background': 'black',
        'edge_cmap': 'darkpurple',
    },
}

_diagnostic_types = np.array(
    [['pca', 'ica'], ['vq', 'neighborhood']]
)


def _nhood_search(umap_object, nhood_size):
    if umap_object._small_data:
        dmat = sklearn.metrics.pairwise_distances(umap_object._raw_data)
        indices = np.argpartition(dmat, nhood_size)[:, :nhood_size]
        dmat_shortened = submatrix(dmat, indices, nhood_size)
        indices_sorted = np.argsort(dmat_shortened)
        indices = submatrix(indices, indices_sorted, nhood_size)
        dists = submatrix(dmat_shortened, indices_sorted, nhood_size)
    else:
        rng_state = np.empty(3, dtype=np.int64)

        init = initialise_search(
            umap_object._rp_forest,
            umap_object._raw_data,
            umap_object._raw_data,
            int(nhood_size * umap_object.transform_queue_size),
            rng_state,
            umap_object._distance_func,
            umap_object._dist_args
        )

        result = initialized_nnd_search(
            umap_object._raw_data,
            umap_object._search_graph.indptr,
            umap_object._search_graph.indices,
            init,
            umap_object._raw_data,
            umap_object._distance_func,
            umap_object._dist_args
        )

        indices, dists = deheap_sort(result)
        indices = indices[:, : nhood_size]
        dists = dists[:, : nhood_size]

    return indices, dists


@numba.jit()
def _nhood_compare(indices_left, indices_right):
    result = np.empty(indices_left.shape[0])

    for i in range(indices_left.shape[0]):
        intersection_size = np.intersect1d(indices_left[i], indices_right[i]).shape[0]
        union_size = np.unique(np.hstack([indices_left[i], indices_right[i]])).shape[0]
        result[i] = float(intersection_size) / float(union_size)

    return result


def _get_extent(points):
    min_x = np.min(points[:, 0])
    max_x = np.max(points[:, 0])
    min_y = np.min(points[:, 1])
    max_y = np.max(points[:, 1])

    extent = (
        np.round(min_x - 0.05 * (max_x - min_x)),
        np.round(max_x + 0.05 * (max_x - min_x)),
        np.round(min_y - 0.05 * (max_y - min_y)),
        np.round(max_y + 0.05 * (max_y - min_y)),
    )

    return extent

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
    extent = _get_extent(points)
    canvas = ds.Canvas(plot_width=width,
                       plot_height=height,
                       x_range=(extent[0], extent[1]),
                       y_range=(extent[2], extent[3]))
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
        unique_values = np.unique(values)
        if unique_values.shape[0] >= 256:
            min_val, max_val = np.min(values), np.max(values)
            bin_size = (max_val - min_val) / 256.0
            data['val_cat'] = pd.Categorical(np.round((values - min_val) / bin_size).astype(np.int16))
            aggregation = canvas.points(data, 'x', 'y', agg=ds.count_cat('val_cat'))
            color_key = _to_hex(plt.get_cmap(cmap)(np.linspace(0, 1, 256)))
            result = tf.shade(aggregation, color_key=color_key, how='eq_hist')
        else:
            data['val_cat'] = pd.Categorical(values)
            aggregation = canvas.points(data, 'x', 'y', agg=ds.count_cat('val_cat'))
            color_key_cols = _to_hex(plt.get_cmap(cmap)(np.linspace(0, 1, unique_values.shape[0])))
            color_key = dict(zip(unique_values, color_key_cols))
            result = tf.shade(aggregation, color_key=color_key, how='eq_hist')

    else:
        aggregation = canvas.points(data, 'x', 'y', agg=ds.count())
        result = tf.shade(aggregation, cmap=plt.get_cmap(cmap))

    if background is not None:
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
    point_size = 100.0 / np.sqrt(points.shape[0])
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

    if points.shape[0] <= width * height // 10:
        return _matplotlib_points(points, labels, values, cmap, color_key,
                                  color_key_cmap, background, width, height)
    else:
        return _datashade_points(points, labels, values, cmap, color_key,
                                 color_key_cmap, background, width, height)


def connectivity(
        umap_object,
        edge_bundling=None,
        edge_cmap='gray_r',
        show_points=True,
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
    if theme is not None:
        cmap = _themes[theme]['cmap']
        color_key_cmap = _themes[theme]['color_key_cmap']
        edge_cmap = _themes[theme]['edge_cmap']
        background = _themes[theme]['background']

    points = umap_object.embedding_
    point_df = pd.DataFrame(points, columns=('x', 'y'))

    coo_graph = umap_object.graph_.tocoo()
    edge_df = pd.DataFrame(np.vstack([coo_graph.row,
                                      coo_graph.col,
                                      coo_graph.data]).T,
                           columns=('source',
                                    'target',
                                    'weight'))
    edge_df['source'] = edge_df.source.astype(np.int32)
    edge_df['target'] = edge_df.target.astype(np.int32)

    extent = _get_extent(points)
    canvas = ds.Canvas(plot_width=width,
                       plot_height=height,
                       x_range=(extent[0], extent[1]),
                       y_range=(extent[2], extent[3]))

    if edge_bundling is None:
        edges = bd.directly_connect_edges(point_df, edge_df, weight='weight')
    elif edge_bundling == 'hammer':
        warn('Hammer edge bundling is expensive for large graphs!\n'
             'This may take a long time to compute!')
        edges = bd.hammer_bundle(point_df, edge_df, weight='weight')
    else:
        raise ValueError('{} is not a recognised bundling method'.format(edge_bundling))

    edge_img = tf.shade(canvas.line(edges, 'x', 'y', agg=ds.sum('weight')),
                        cmap=plt.get_cmap(edge_cmap))
    edge_img = tf.set_background(edge_img, background)

    if show_points:
        point_img = _datashade_points(points, labels, values, cmap, color_key,
                                      color_key_cmap, None, width, height)
        tf.dynspread(point_img, threshold=0.95)
        return tf.stack(edge_img, point_img, how="over")
    else:
        return edge_img


def diagnostic(
        umap_object,
        diagnostic_type='pca',
        nhood_size=15, ax=None,
        cmap='viridis',
        point_size=None
):

    points = umap_object.embedding_

    if points.shape[1] != 2:
        raise ValueError('Plotting is currently only implemented for 2D embeddings')

    if point_size is None:
        point_size = 100.0 / np.sqrt(points.shape[0])

    if ax is None and diagnostic_type != 'all':
        fig = plt.figure()
        ax = fig.add_subplot(111)

    if diagnostic_type == 'pca':
        color_proj = sklearn.decomposition.PCA(n_components=3).fit_transform(umap_object._raw_data)
        color_proj -= np.min(color_proj)
        color_proj /= np.max(color_proj, axis=0)

        ax.scatter(points[:, 0], points[:, 1], s=point_size, c=color_proj, alpha=0.66)
        ax.set_title('Colored by RGB coords of PCA embedding')
        ax.set(xticks=[], yticks=[])

    elif diagnostic_type == 'ica':
        color_proj = sklearn.decomposition.FastICA(n_components=3).fit_transform(umap_object._raw_data)
        color_proj -= np.min(color_proj)
        color_proj /= np.max(color_proj, axis=0)

        ax.scatter(points[:, 0], points[:, 1], s=point_size, c=color_proj, alpha=0.66)
        ax.set_title('Colored by RGB coords of FastICA embedding')
        ax.set(xticks=[], yticks=[])

    elif diagnostic_type == 'vq':
        color_projector = sklearn.cluster.KMeans(n_clusters=3).fit(umap_object._raw_data)
        color_proj = sklearn.metrics.pairwise_distances(umap_object._raw_data,
                                                        color_projector.cluster_centers_)
        color_proj -= np.min(color_proj)
        color_proj /= np.max(color_proj, axis=0)

        ax.scatter(points[:, 0], points[:, 1], s=point_size, c=color_proj, alpha=0.66)
        ax.set_title('Colored by RGB coords of Vector Quantization')
        ax.set(xticks=[], yticks=[])

    elif diagnostic_type == 'neighborhood':
        highd_indices, highd_dists = _nhood_search(umap_object, nhood_size)
        tree = sklearn.neighbors.KDTree(points)
        lowd_dists, lowd_indices = tree.query(points, k=nhood_size)
        accuracy = _nhood_compare(highd_indices.astype(np.int32),
                                  lowd_indices.astype(np.int32))

        ax.scatter(points[:, 0], points[:, 1], s=point_size, c=accuracy,
                   cmap=cmap, vmin=0.0, vmax=1.0)
        ax.set_title('Colored by neighborhood Jaccard index')
        ax.set(xticks=[], yticks=[])

    elif diagnostic_type == 'all':

        fig, axs = plt.subplots(2, 2)
        for i in range(2):
            for j in range(2):
                diagnostic(umap_object,
                           diagnostic_type=_diagnostic_types[i, j],
                           ax=axs[i, j],
                           point_size=point_size / 4.0)

        plt.tight_layout()

    else:
        raise ValueError('Unknown diagnostic; should be one of '
                         '"pca", "ica", "vq", "neighborhood", or "all"')

    return ax


def interactive(umap_object):
    pass
