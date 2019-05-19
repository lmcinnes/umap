# -*- coding: utf-8 -*-
import os
import sys
sys.path.insert(0, os.path.abspath('.'))
from umap import __version__
from sktda_docs_config import *


# General information about the project.
project = 'umap'
copyright = '2019, Leland McInnes'
author = 'Leland McInnes'

version = __version__
release = __version__

# Output file base name for HTML help builder.
htmlhelp_basename = 'umapdoc'

html_short_title = project
# html_static_path = ["../examples/output"]

extensions.append('sphinx_gallery.gen_gallery')


# Example configuration for intersphinx: refer to the Python standard library.
intersphinx_mapping = {
    'python': ('https://docs.python.org/{.major}'.format(sys.version_info), None),
    'numpy': ('https://docs.scipy.org/doc/numpy/', None),
    'scipy': ('https://docs.scipy.org/doc/scipy/reference', None),
    'matplotlib': ('https://matplotlib.org/', None),
    'pandas': ('https://pandas.pydata.org/pandas-docs/stable/', None),
    'sklearn': ('http://scikit-learn.org/stable/', None),
    'bokeh': ('http://bokeh.pydata.org/en/latest/', None),
}

# -- Options for sphinx-gallery ---------------------------------------------

sphinx_gallery_conf = {
    # path to your examples scripts
    'examples_dirs': '../examples',
    # path where to save gallery generated examples
    'gallery_dirs': 'auto_examples',
    'plot_gallery': False,  # Turn off running the examples for now
    'reference_url': {
        'umap': None,
        'python': 'https://docs.python.org/{.major}'.format(sys.version_info),
        'numpy': 'https://docs.scipy.org/doc/numpy/',
        'scipy': 'https://docs.scipy.org/doc/scipy/reference',
        'matplotlib': 'https://matplotlib.org/',
        'pandas': 'https://pandas.pydata.org/pandas-docs/stable/',
        'sklearn': 'http://scikit-learn.org/stable/',
        'bokeh': 'http://bokeh.pydata.org/en/latest/',
    }
}

