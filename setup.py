from setuptools import setup


def readme():
    try:
        with open("README.rst", encoding="UTF-8") as readme_file:
            return readme_file.read()
    except TypeError:
        # Python 2.7 doesn't support encoding argument in builtin open
        import io

        with io.open("README.rst", encoding="UTF-8") as readme_file:
            return readme_file.read()


configuration = {
    "name": "umap-learn",
    "version": "0.5.3",
    "description": "Uniform Manifold Approximation and Projection",
    "long_description": readme(),
    "long_description_content_type": "text/x-rst",
    "classifiers": [
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Developers",
        "License :: OSI Approved",
        "Programming Language :: C",
        "Programming Language :: Python",
        "Topic :: Software Development",
        "Topic :: Scientific/Engineering",
        "Operating System :: Microsoft :: Windows",
        "Operating System :: POSIX",
        "Operating System :: Unix",
        "Operating System :: MacOS",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
    ],
    "keywords": "dimension reduction t-sne manifold",
    "url": "http://github.com/lmcinnes/umap",
    "maintainer": "Leland McInnes",
    "maintainer_email": "leland.mcinnes@gmail.com",
    "license": "BSD",
    "packages": ["umap"],
    "install_requires": [
        "numpy >= 1.17",
        "scikit-learn >= 0.22",
        "scipy >= 1.0",
        "numba >= 0.49",
        "pynndescent >= 0.5",
        "tqdm",
    ],
    "extras_require": {
        "plot": [
            "pandas",
            "matplotlib",
            "datashader",
            "bokeh",
            "holoviews",
            "colorcet",
            "seaborn",
            "scikit-image",
        ],
        "parametric_umap": ["tensorflow >= 2.1", "tensorflow-probability >= 0.10"],
    },
    "ext_modules": [],
    "cmdclass": {},
    "test_suite": "pytest",
    "tests_require": ["pytest"],
    "data_files": (),
    "zip_safe": False,
}

setup(**configuration)
