from Cython.Distutils import build_ext
from setuptools import setup, Extension

import numpy

umap_utils = Extension('umap.umap_utils',
                        sources=['umap/umap_utils.pyx'],
                        libraries=['gsl', 'cblas'],
                        library_dirs=['/usr/lib'],
                        include_dirs=[numpy.get_include(), '/usr/include'])

def readme():
    with open('README.rst') as readme_file:
        return readme_file.read()

configuration = {
    'name' : 'umap',
    'version' : '0.0.1',
    'description' : 'Uniform Manifold Approximation and Projection',
    'long_description' : readme(),
    'classifiers' : [
        'Development Status :: 1 - Alpha',
        'Intended Audience :: Science/Research',
        'Intended Audience :: Developers',
        'License :: OSI Approved',
        'Programming Language :: C',
        'Programming Language :: Python',
        'Topic :: Software Development',
        'Topic :: Scientific/Engineering',
        'Operating System :: Microsoft :: Windows',
        'Operating System :: POSIX',
        'Operating System :: Unix',
        'Operating System :: MacOS',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3.6',
    ],
    'keywords' : 'dimension reduction t-sne manifold',
    'url' : 'http://github.com/lmcinnes/umap',
    'maintainer' : 'Leland McInnes',
    'maintainer_email' : 'leland.mcinnes@gmail.com',
    'license' : 'BSD',
    'packages' : ['umap'],
    'install_requires' : ['scikit-learn>=0.16',
                          'cython >= 0.17'],
    'ext_modules' : [umap_utils,],
    'cmdclass' : {'build_ext' : build_ext},
    'test_suite' : 'nose.collector',
    'tests_require' : ['nose'],
    'data_files' : ()
    }

setup(**configuration)
