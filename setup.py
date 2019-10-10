from setuptools import setup

def readme():
    with open('README.rst') as readme_file:
        return readme_file.read()

configuration = {
    'name' : 'umap-learn',
    'version': '0.4.0',
    'description' : 'Uniform Manifold Approximation and Projection',
    'long_description' : readme(),
    'classifiers' : [
        'Development Status :: 3 - Alpha',
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
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.8',
    ],
    'keywords' : 'dimension reduction t-sne manifold',
    'url' : 'http://github.com/lmcinnes/umap',
    'maintainer' : 'Leland McInnes',
    'maintainer_email' : 'leland.mcinnes@gmail.com',
    'license' : 'BSD',
    'packages' : ['umap'],
    'install_requires': ['numpy >= 1.13',
                         'scikit-learn >= 0.20',
                          'scipy >= 1.0',
                         'numba >= 0.42',
                         'tbb >= 2019.0'],
    'ext_modules' : [],
    'cmdclass' : {},
    'test_suite' : 'nose.collector',
    'tests_require' : ['nose'],
    'data_files' : ()
    }

setup(**configuration)
