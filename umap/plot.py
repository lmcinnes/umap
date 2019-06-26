import numpy as np
import pandas as pd
import numba

import datashader as ds
import datashader.transfer_functions as tf
import datashader.bundling as bd
import matplotlib.pyplot as plt
import colorcet
import matplotlib.colors

import bokeh.plotting as bpl
import bokeh.transform as btr
import holoviews as hv
import holoviews.operation.datashader as hd

import sklearn.decomposition
import sklearn.cluster
import sklearn.neighbors

from umap.nndescent import initialise_search, initialized_nnd_search
from umap.utils import deheap_sort, submatrix

from bokeh.plotting import output_notebook, output_file, show
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


@numba.vectorize(['uint8(uint32)', 'uint8(uint32)'])
def _red(x):
    return (x & 0xff0000) >> 16


@numba.vectorize(['uint8(uint32)', 'uint8(uint32)'])
def _green(x):
    return (x & 0x00ff00) >> 8


@numba.vectorize(['uint8(uint32)', 'uint8(uint32)'])
def _blue(x):
    return (x & 0x0000ff)


_themes = {
    'fire': {
        'cmap': 'fire',
        'color_key_cmap': 'rainbow',
        'background': 'black',
        'edge_cmap': 'fire',
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
    [['pca', 'ica'], ['vq', 'local_dim']]
)


def _embed_datashader_in_an_axis(datashader_image, ax):
    img_rev = datashader_image.data[::-1]
    mpl_img = np.dstack([_blue(img_rev),
                         _green(img_rev),
                         _red(img_rev)])
    ax.imshow(mpl_img)
    return ax


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
    """Compute Jaccard index of two neighborhoods"""
    result = np.empty(indices_left.shape[0])

    for i in range(indices_left.shape[0]):
        intersection_size = np.intersect1d(indices_left[i], indices_right[i]).shape[0]
        union_size = np.unique(np.hstack([indices_left[i], indices_right[i]])).shape[0]
        result[i] = float(intersection_size) / float(union_size)

    return result


def _get_extent(points):
    """Compute bounds on a space with appropriate padding"""
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


def _select_font_color(background):
    if background == 'black':
        font_color = 'white'
    elif background.startswith('#'):
        mean_val = np.mean([int('0x' + c)
                            for c in
                            (background[1:3], background[3:5], background[5:7])])
        if mean_val > 126:
            font_color = 'black'
        else:
            font_color = 'white'

    else:
        font_color = 'black'

    return font_color


def _datashade_points(
        points,
        ax=None,
        labels=None,
        values=None,
        cmap='Blues',
        color_key=None,
        color_key_cmap='Spectral',
        background='white',
        width=800,
        height=800,
):

    """Use datashader to plot points"""
    extent = _get_extent(points)
    canvas = ds.Canvas(plot_width=width,
                       plot_height=height,
                       x_range=(extent[0], extent[1]),
                       y_range=(extent[2], extent[3]))
    data = pd.DataFrame(points, columns=('x', 'y'))

    # Color by labels
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

    # Color by values
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

    # Color by density (default datashader option)
    else:
        aggregation = canvas.points(data, 'x', 'y', agg=ds.count())
        result = tf.shade(aggregation, cmap=plt.get_cmap(cmap))

    if background is not None:
        result = tf.set_background(result, background)

    if ax is not None:
        _embed_datashader_in_an_axis(result, ax)
        return ax
    else:
        return result


def _matplotlib_points(
        points,
        ax=None,
        labels=None,
        values=None,
        cmap='Blues',
        color_key=None,
        color_key_cmap='Spectral',
        background='white',
        width=800,
        height=800,
):
    """Use matplotlib to plot points"""
    point_size = 100.0 / np.sqrt(points.shape[0])

    if ax is None:
        dpi = plt.rcParams['figure.dpi']
        fig = plt.figure(figsize=(width / dpi, height / dpi))
        ax = fig.add_subplot(111)

    ax.set_facecolor(background)

    # Color by labels
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

    # Color by values
    elif values is not None:
        if values.shape[0] != points.shape[0]:
            raise ValueError('Values must have a value for '
                             'each sample (size mismatch: {} {})'.format(values.shape[0],
                                                                         points.shape[0]))
        ax.scatter(points[:, 0], points[:, 1], s=point_size, c=values, cmap=cmap)

    # No color (just pick the midpoint of the cmap)
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
    """Plot an embedding as points. Currently this only works
    for 2D embeddings. While there are many optional parameters
    to further control and tailor the plotting, you need only
    pass in the trained/fit umap model to get results. This plot
    utility will attempt to do the hard work of avoiding
    overplotting issues, and make it easy to automatically
    colour points by a categorical labelling or numeric values.

    This method is intended to be used within a Jupyter
    notebook with ``%matplotlib inline``.

    Parameters
    ----------
    umap_object: trained UMAP object
        A trained UMAP object that has a 2D embedding.

    labels: array, shape (n_samples,) (optional, default None)
        An array of labels (assumed integer or categorical),
        one for each data sample.
        This will be used for coloring the points in
        the plot according to their label. Note that
        this option is mutually exclusive to the ``values``
        option.

    values: array, shape (n_samples,) (optional, default None)
        An array of values (assumed float or continuous),
        one for each sample.
        This will be used for coloring the points in
        the plot according to a colorscale associated
        to the total range of values. Note that this
        option is mutually exclusive to the ``labels``
        option.

    theme: string (optional, default None)
        A color theme to use for plotting. A small set of
        predefined themes are provided which have relatively
        good aesthetics. Available themes are:
           * 'blue'
           * 'red'
           * 'green'
           * 'inferno'
           * 'fire'
           * 'viridis'
           * 'darkblue'
           * 'darkred'
           * 'darkgreen'

    cmap: string (optional, default 'Blues')
        The name of a matplotlib colormap to use for coloring
        or shading points. If no labels or values are passed
        this will be used for shading points according to
        density (largely only of relevance for very large
        datasets). If values are passed this will be used for
        shading according the value. Note that if theme
        is passed then this value will be overridden by the
        corresponding option of the theme.

    color_key: dict or array, shape (n_categories) (optional, default None)
        A way to assign colors to categoricals. This can either be
        an explicit dict mapping labels to colors (as strings of form
        '#RRGGBB'), or an array like object providing one color for
        each distinct category being provided in ``labels``. Either
        way this mapping will be used to color points according to
        the label. Note that if theme
        is passed then this value will be overridden by the
        corresponding option of the theme.

    color_key_cmap: string (optional, default 'Spectral')
        The name of a matplotlib colormap to use for categorical coloring.
        If an explicit ``color_key`` is not given a color mapping for
        categories can be generated from the label list and selecting
        a matching list of colors from the given colormap. Note
        that if theme
        is passed then this value will be overridden by the
        corresponding option of the theme.

    background: string (optional, default 'white)
        The color of the background. Usually this will be either
        'white' or 'black', but any color name will work. Ideally
        one wants to match this appropriately to the colors being
        used for points etc. This is one of the things that themes
        handle for you. Note that if theme
        is passed then this value will be overridden by the
        corresponding option of the theme.

    width: int (optional, default 800)
        The desired width of the plot in pixels.

    height: int (optional, default 800)
        The desired height of the plot in pixels

    Returns
    -------
    result: matplotlib axis
        The result is a matplotlib axis with the relevant plot displayed.
        If you are using a notbooks and have ``%matplotlib inline`` set
        then this will simply display inline.
    """
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

    font_color = _select_font_color(background)

    dpi = plt.rcParams['figure.dpi']
    fig = plt.figure(figsize=(width / dpi, height / dpi))
    ax = fig.add_subplot(111)

    if points.shape[0] <= width * height // 10:
        ax = _matplotlib_points(points, ax, labels, values, cmap, color_key,
                                color_key_cmap, background, width, height)
    else:
        ax = _datashade_points(points, ax, labels, values, cmap, color_key,
                               color_key_cmap, background, width, height)

    ax.set(xticks=[], yticks=[])
    if umap_object.metric != 'euclidean':
        ax.text(0.99,
                0.01,
                'UMAP: metric={}, n_neighbors={}, min_dist={}'.format(
                    umap_object.metric,
                    umap_object.n_neighbors,
                    umap_object.min_dist),
                transform=ax.transAxes,
                horizontalalignment='right',
                color=font_color)
    else:
        ax.text(0.99,
                0.01,
                'UMAP: n_neighbors={}, min_dist={}'.format(umap_object.n_neighbors,
                                                           umap_object.min_dist),
                transform=ax.transAxes,
                horizontalalignment='right',
                color=font_color)

    return ax


def connectivity(
        umap_object,
        edge_bundling=None,
        edge_cmap='gray_r',
        show_points=False,
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
    """Plot connectivity relationships of the underlying UMAP
    simplicial set data structure. Internally UMAP will make
    use of what can be viewed as a weighted graph. This graph
    can be plotted using the layout provided by UMAP as a
    potential diagnostic view of the embedding. Currently this only works
    for 2D embeddings. While there are many optional parameters
    to further control and tailor the plotting, you need only
    pass in the trained/fit umap model to get results. This plot
    utility will attempt to do the hard work of avoiding
    overplotting issues and provide options for plotting the
    points as well as using edge bundling for graph visualization.

    Parameters
    ----------
    umap_object: trained UMAP object
        A trained UMAP object that has a 2D embedding.

    edge_bundling: string or None (optional, default None)
        The edge bundling method to use. Currently supported
        are None or 'hammer'. See the datashader docs
        on graph visualization for more details.

    edge_cmap: string (default 'gray_r')
        The name of a matplotlib colormap to use for shading/
        coloring the edges of the connectivity graph. Note that
        the ``theme``, if specified, will override this.

    show_points: bool (optional False)
        Whether to display the points over top of the edge
        connectivity. Further options allow for coloring/
        shading the points accordingly.

    labels: array, shape (n_samples,) (optional, default None)
        An array of labels (assumed integer or categorical),
        one for each data sample.
        This will be used for coloring the points in
        the plot according to their label. Note that
        this option is mutually exclusive to the ``values``
        option.

    values: array, shape (n_samples,) (optional, default None)
        An array of values (assumed float or continuous),
        one for each sample.
        This will be used for coloring the points in
        the plot according to a colorscale associated
        to the total range of values. Note that this
        option is mutually exclusive to the ``labels``
        option.

    theme: string (optional, default None)
        A color theme to use for plotting. A small set of
        predefined themes are provided which have relatively
        good aesthetics. Available themes are:
           * 'blue'
           * 'red'
           * 'green'
           * 'inferno'
           * 'fire'
           * 'viridis'
           * 'darkblue'
           * 'darkred'
           * 'darkgreen'

    cmap: string (optional, default 'Blues')
        The name of a matplotlib colormap to use for coloring
        or shading points. If no labels or values are passed
        this will be used for shading points according to
        density (largely only of relevance for very large
        datasets). If values are passed this will be used for
        shading according the value. Note that if theme
        is passed then this value will be overridden by the
        corresponding option of the theme.

    color_key: dict or array, shape (n_categories) (optional, default None)
        A way to assign colors to categoricals. This can either be
        an explicit dict mapping labels to colors (as strings of form
        '#RRGGBB'), or an array like object providing one color for
        each distinct category being provided in ``labels``. Either
        way this mapping will be used to color points according to
        the label. Note that if theme
        is passed then this value will be overridden by the
        corresponding option of the theme.

    color_key_cmap: string (optional, default 'Spectral')
        The name of a matplotlib colormap to use for categorical coloring.
        If an explicit ``color_key`` is not given a color mapping for
        categories can be generated from the label list and selecting
        a matching list of colors from the given colormap. Note
        that if theme
        is passed then this value will be overridden by the
        corresponding option of the theme.

    background: string (optional, default 'white)
        The color of the background. Usually this will be either
        'white' or 'black', but any color name will work. Ideally
        one wants to match this appropriately to the colors being
        used for points etc. This is one of the things that themes
        handle for you. Note that if theme
        is passed then this value will be overridden by the
        corresponding option of the theme.

    width: int (optional, default 800)
        The desired width of the plot in pixels.

    height: int (optional, default 800)
        The desired height of the plot in pixels

    Returns
    -------
    result: matplotlib axis
        The result is a matplotlib axis with the relevant plot displayed.
        If you are using a notbooks and have ``%matplotlib inline`` set
        then this will simply display inline.
    """
    if theme is not None:
        cmap = _themes[theme]['cmap']
        color_key_cmap = _themes[theme]['color_key_cmap']
        edge_cmap = _themes[theme]['edge_cmap']
        background = _themes[theme]['background']

    points = umap_object.embedding_
    point_df = pd.DataFrame(points, columns=('x', 'y'))

    point_size = 100.0 / np.sqrt(points.shape[0])
    if point_size > 1:
        px_size = int(np.round(point_size))
    else:
        px_size = 1

    if show_points:
        edge_how = 'log'
    else:
        edge_how = 'eq_hist'

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
                        cmap=plt.get_cmap(edge_cmap), how=edge_how)
    edge_img = tf.set_background(edge_img, background)

    if show_points:
        point_img = _datashade_points(points, None, labels, values, cmap, color_key,
                                      color_key_cmap, None, width, height)
        if px_size > 1:
            point_img = tf.dynspread(point_img, threshold=0.5, max_px=px_size)
        result = tf.stack(edge_img, point_img, how="over")
    else:
        result = edge_img

    font_color = _select_font_color(background)

    dpi = plt.rcParams['figure.dpi']
    fig = plt.figure(figsize=(width / dpi, height / dpi))
    ax = fig.add_subplot(111)

    _embed_datashader_in_an_axis(result, ax)

    ax.set(xticks=[], yticks=[])
    ax.text(0.99,
            0.01,
            'UMAP: n_neighbors={}, min_dist={}'.format(umap_object.n_neighbors,
                                                       umap_object.min_dist),
            transform=ax.transAxes,
            horizontalalignment='right',
            color=font_color)

    return ax


def diagnostic(
        umap_object,
        diagnostic_type='pca',
        nhood_size=15,
        local_variance_threshold=0.8,
        ax=None,
        cmap='viridis',
        point_size=None,
        background='white'
):
    """Provide a diagnostic plot or plots for a UMAP embedding.
    There are a number of plots that can be helpful for diagnostic
    purposes in understanding your embedding. Currently these are
    restricted to methods of coloring a scatterplot of the
    embedding to show more about how the embedding is representing
    the data. The first class of such plots uses a linear method
    that preserves global structure well to embed the data into
    three dimensions, and then interprets such coordinates as a
    color space -- coloring the points by their location in the
    linear global structure preserving embedding. In such plots
    one should look for discontinuities of colour, and consider
    overall global gradients of color. The diagnostic types here
    are ``'pca'``, ``'ica'``, and ``'vq'`` (vector quantization).

    The second class consider the local neighbor structure. One
    can either look at how well the neighbor structure is
    preserved, or how the estimated local dimension of the data
    varies. Both of these are available, although the local
    dimension estimation is the preferred option. You can
    access these are diagnostic types ``'local_dim'`` and
    ``'neighborhood'``.

    Finally the diagnostic type ``'all'`` will provide a
    grid of diagnostic plots.

    Parameters
    ----------
    umap_object: trained UMAP object
        A trained UMAP object that has a 2D embedding.

    diagnostic_type: str (optional, default 'pca')
        The type of diagnostic plot to show. The options are
           * 'pca'
           * 'ica'
           * 'vq'
           * 'local_dim'
           * 'neighborhood'
           * 'all'

    nhood_size: int (optional, default 15)
        The size of neighborhood to compare for local
        neighborhood preservation estimates.

    local_variance_threshold: float (optional, default 0.8)
        To estimate the local dimension we consider a PCA of
        the local neighborhood and estimate the dimension
        as that which provides ``local_variance_threshold``
        or more of the ``variance_explained_ratio``.

    ax: matlotlib axis (optional, default None)
        A matplotlib axis to plot to, or, if None, a new
        axis will be created and returned.

    cmap: str (optional, default 'viridis')
        The name of a matplotlib colormap to use for coloring
        points if the ``'local_dim'`` or ``'neighborhood'``
        option are selected.

    point_size: int (optional, None)
        If provided this will fix the point size for the
        plot(s). If None then a suitable point size will
        be estimated from the data.

    Returns
    -------
    result: matplotlib axis
        The result is a matplotlib axis with the relevant plot displayed.
        If you are using a notbooks and have ``%matplotlib inline`` set
        then this will simply display inline.
    """

    points = umap_object.embedding_

    if points.shape[1] != 2:
        raise ValueError('Plotting is currently only implemented for 2D embeddings')

    if point_size is None:
        point_size = 100.0 / np.sqrt(points.shape[0])

    font_color = _select_font_color(background)

    if ax is None and diagnostic_type != 'all':
        fig = plt.figure()
        ax = fig.add_subplot(111)

    if diagnostic_type == 'pca':
        color_proj = sklearn.decomposition.PCA(n_components=3).fit_transform(umap_object._raw_data)
        color_proj -= np.min(color_proj)
        color_proj /= np.max(color_proj, axis=0)

        ax.scatter(points[:, 0], points[:, 1], s=point_size, c=color_proj, alpha=0.66)
        ax.set_title('Colored by RGB coords of PCA embedding')
        ax.text(0.99,
                0.01,
                'UMAP: n_neighbors={}, min_dist={}'.format(umap_object.n_neighbors,
                                                           umap_object.min_dist),
                transform=ax.transAxes,
                horizontalalignment='right',
                color=font_color)
        ax.set(xticks=[], yticks=[])

    elif diagnostic_type == 'ica':
        color_proj = sklearn.decomposition.FastICA(n_components=3).fit_transform(umap_object._raw_data)
        color_proj -= np.min(color_proj)
        color_proj /= np.max(color_proj, axis=0)

        ax.scatter(points[:, 0], points[:, 1], s=point_size, c=color_proj, alpha=0.66)
        ax.set_title('Colored by RGB coords of FastICA embedding')
        ax.text(0.99,
                0.01,
                'UMAP: n_neighbors={}, min_dist={}'.format(umap_object.n_neighbors,
                                                           umap_object.min_dist),
                transform=ax.transAxes,
                horizontalalignment='right',
                color=font_color)
        ax.set(xticks=[], yticks=[])

    elif diagnostic_type == 'vq':
        color_projector = sklearn.cluster.KMeans(n_clusters=3).fit(umap_object._raw_data)
        color_proj = sklearn.metrics.pairwise_distances(umap_object._raw_data,
                                                        color_projector.cluster_centers_)
        color_proj -= np.min(color_proj)
        color_proj /= np.max(color_proj, axis=0)

        ax.scatter(points[:, 0], points[:, 1], s=point_size, c=color_proj, alpha=0.66)
        ax.set_title('Colored by RGB coords of Vector Quantization')
        ax.text(0.99,
                0.01,
                'UMAP: n_neighbors={}, min_dist={}'.format(umap_object.n_neighbors,
                                                           umap_object.min_dist),
                transform=ax.transAxes,
                horizontalalignment='right',
                color=font_color)
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
        ax.text(0.99,
                0.01,
                'UMAP: n_neighbors={}, min_dist={}'.format(umap_object.n_neighbors,
                                                           umap_object.min_dist),
                transform=ax.transAxes,
                horizontalalignment='right',
                color=font_color)
        ax.set(xticks=[], yticks=[])

    elif diagnostic_type == 'local_dim':
        highd_indices, highd_dists = _nhood_search(umap_object, umap_object.n_neighbors)
        data = umap_object._raw_data
        local_dim = np.empty(data.shape[0], dtype=np.int64)
        for i in range(data.shape[0]):
            pca = sklearn.decomposition.PCA().fit(data[highd_indices[i]])
            local_dim[i] = np.where(np.cumsum(pca.explained_variance_ratio_)
                                    > local_variance_threshold)[0][0]
        ax.scatter(points[:, 0], points[:, 1], s=point_size, c=local_dim,
                   cmap=cmap, vmin=0.0, vmax=1.0)
        ax.set_title('Colored by approx local dimension')
        ax.text(0.99,
                0.01,
                'UMAP: n_neighbors={}, min_dist={}'.format(umap_object.n_neighbors,
                                                           umap_object.min_dist),
                transform=ax.transAxes,
                horizontalalignment='right',
                color=font_color)
        ax.set(xticks=[], yticks=[])


    elif diagnostic_type == 'all':

        fig, axs = plt.subplots(2, 2, figsize=(10, 10))
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


def interactive(
        umap_object,
        labels=None,
        values=None,
        hover_data=None,
        theme=None,
        cmap='Blues',
        color_key=None,
        color_key_cmap='Spectral',
        background='white',
        width=800,
        height=800,
        point_size=None,
):
    """Create an interactive bokeh plot of a UMAP embedding.
    While static plots are useful, sometimes a plot that
    supports interactive zooming, and hover tooltips for
    individual points is much more desireable. This function
    provides a simple interface for creating such plots. The
    result is a bokeh plot that will be displayed in a notebook.

    Note that more complex tooltips etc. will require custom
    code -- this is merely meant to provide fast and easy
    access to interactive plotting.

    Parameters
    ----------
    umap_object: trained UMAP object
        A trained UMAP object that has a 2D embedding.

    labels: array, shape (n_samples,) (optional, default None)
        An array of labels (assumed integer or categorical),
        one for each data sample.
        This will be used for coloring the points in
        the plot according to their label. Note that
        this option is mutually exclusive to the ``values``
        option.

    values: array, shape (n_samples,) (optional, default None)
        An array of values (assumed float or continuous),
        one for each sample.
        This will be used for coloring the points in
        the plot according to a colorscale associated
        to the total range of values. Note that this
        option is mutually exclusive to the ``labels``
        option.

    hover_data: DataFrame, shape (n_samples, n_tooltip_features)
    (optional, default None)
        A dataframe of tooltip data. Each column of the dataframe
        should be a Series of length ``n_samples`` providing a value
        for each data point. Column names will be used for
        identifying information within the tooltip.

    theme: string (optional, default None)
        A color theme to use for plotting. A small set of
        predefined themes are provided which have relatively
        good aesthetics. Available themes are:
           * 'blue'
           * 'red'
           * 'green'
           * 'inferno'
           * 'fire'
           * 'viridis'
           * 'darkblue'
           * 'darkred'
           * 'darkgreen'

    cmap: string (optional, default 'Blues')
        The name of a matplotlib colormap to use for coloring
        or shading points. If no labels or values are passed
        this will be used for shading points according to
        density (largely only of relevance for very large
        datasets). If values are passed this will be used for
        shading according the value. Note that if theme
        is passed then this value will be overridden by the
        corresponding option of the theme.

    color_key: dict or array, shape (n_categories) (optional, default None)
        A way to assign colors to categoricals. This can either be
        an explicit dict mapping labels to colors (as strings of form
        '#RRGGBB'), or an array like object providing one color for
        each distinct category being provided in ``labels``. Either
        way this mapping will be used to color points according to
        the label. Note that if theme
        is passed then this value will be overridden by the
        corresponding option of the theme.

    color_key_cmap: string (optional, default 'Spectral')
        The name of a matplotlib colormap to use for categorical coloring.
        If an explicit ``color_key`` is not given a color mapping for
        categories can be generated from the label list and selecting
        a matching list of colors from the given colormap. Note
        that if theme
        is passed then this value will be overridden by the
        corresponding option of the theme.

    background: string (optional, default 'white)
        The color of the background. Usually this will be either
        'white' or 'black', but any color name will work. Ideally
        one wants to match this appropriately to the colors being
        used for points etc. This is one of the things that themes
        handle for you. Note that if theme
        is passed then this value will be overridden by the
        corresponding option of the theme.

    width: int (optional, default 800)
        The desired width of the plot in pixels.

    height: int (optional, default 800)
        The desired height of the plot in pixels

    Returns
    -------

    """
    if theme is not None:
        cmap = _themes[theme]['cmap']
        color_key_cmap = _themes[theme]['color_key_cmap']
        background = _themes[theme]['background']

    if labels is not None and values is not None:
        raise ValueError('Conflicting options; only one of labels or values should be set')

    points = umap_object.embedding_

    if points.shape[1] != 2:
        raise ValueError('Plotting is currently only implemented for 2D embeddings')

    if point_size is None:
        point_size = 100.0 / np.sqrt(points.shape[0])

    data = pd.DataFrame(umap_object.embedding_, columns=('x', 'y'))

    if labels is not None:
        data['label'] = labels

        if color_key is None:
            unique_labels = np.unique(labels)
            num_labels = unique_labels.shape[0]
            color_key = _to_hex(plt.get_cmap(color_key_cmap)(np.linspace(0, 1, num_labels)))

        if isinstance(color_key, dict):
            data['color'] = pd.Series(labels).map(color_key)
        else:
            unique_labels = np.unique(labels)
            if len(color_key) < unique_labels.shape[0]:
                raise ValueError('Color key must have enough colors for the number of labels')

            new_color_key = {k: color_key[i] for i, k in enumerate(unique_labels)}
            data['color'] = pd.Series(labels).map(new_color_key)

        colors = 'color'

    elif values is not None:
        data['value'] = values
        palette = _to_hex(plt.get_cmap(cmap)(np.linspace(0, 1, 256)))
        colors = btr.linear_cmap('value', palette, low=np.min(values), high=np.max(values))

    else:
        colors = matplotlib.colors.rgb2hex(plt.get_cmap(cmap)(0.5))

    if points.shape[0] <= width * height // 10:

        if hover_data is not None:
            tooltip_dict = {}
            for col_name in hover_data:
                data[col_name] = hover_data[col_name]
                tooltip_dict[col_name] = '@' + col_name
            tooltips = list(tooltip_dict.items())
        else:
            tooltips = None

        # bpl.output_notebook(hide_banner=True) # this doesn't work for non-notebook use
        data_source = bpl.ColumnDataSource(data)

        plot = bpl.figure(width=width,
                          height=height,
                          tooltips=tooltips,
                          background_fill_color=background)
        plot.circle(x='x', y='y', source=data_source, color=colors, size=point_size)

        plot.grid.visible = False
        plot.axis.visible = False

        # bpl.show(plot)
    else:
        if hover_data is not None:
            warn('Too many points for hover data -- tooltips will not'
                 'be displayed. Sorry; try subssampling your data.')
        hv.extension('bokeh')
        hv.output(size=300)
        hv.opts('RGB [bgcolor="{}", xaxis=None, yaxis=None]'.format(background))
        if labels is not None:
            point_plot = hv.Points(data, kdims=['x', 'y'], vdims=['color'])
            plot = hd.datashade(point_plot,
                                aggregator=ds.count_cat('color'),
                                cmap=plt.get_cmap(cmap),
                                width=width,
                                height=height)
        elif values is not None:
            min_val = data.values.min()
            val_range = data.values.max() - min_val
            data['val_cat'] = pd.Categorical((data.values - min_val) //
                                             (val_range // 256))
            point_plot = hv.Points(data, kdims=['x', 'y'], vdims=['val_cat'])
            plot = hd.datashade(point_plot,
                                aggregator=ds.count_cat('val_cat'),
                                cmap=plt.get_cmap(cmap),
                                width=width,
                                height=height)
        else:
            point_plot = hv.Points(data, kdims=['x', 'y'])
            plot = hd.datashade(point_plot,
                                aggregator=ds.count(),
                                cmap=plt.get_cmap(cmap),
                                width=width,
                                height=height)

    return plot


