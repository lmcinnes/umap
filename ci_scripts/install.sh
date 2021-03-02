if [[ "$DISTRIB" == "conda" ]]; then

  # Deactivate the travis-provided virtual environment and setup a
    # conda-based environment instead
  if [ $TRAVIS_OS_NAME = 'linux' ]; then
    # Only Linux has a virtual environment activated; Mac does not.
    deactivate
  fi

  # Use the miniconda installer for faster download / install of conda
  # itself
  pushd .
  cd
  mkdir -p download
  cd download
  echo "Cached in $HOME/download :"
  ls -l
  echo
# For now, ignoring the cached file.
#  if [[ ! -f miniconda.sh ]]
#     then
     if [ $TRAVIS_OS_NAME = 'osx' ]; then
       # MacOS URL found here: https://docs.conda.io/en/latest/miniconda.html
       wget \
       https://repo.anaconda.com/miniconda/Miniconda3-latest-MacOSX-x86_64.sh \
         -O miniconda.sh
     else
       wget \
       http://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh \
         -O miniconda.sh
     fi
#  fi
  chmod +x miniconda.sh && ./miniconda.sh -b -p $HOME/miniconda
  cd ..
  export PATH=$HOME/miniconda/bin:$HOME/miniconda3/bin:$PATH
  conda update --yes conda
  popd

  # Configure the conda environment and put it in the path using the
  # provided versions
#  conda create -n testenv --yes python=$PYTHON_VERSION pip \
#        numpy=$NUMPY_VERSION scipy=$SCIPY_VERSION numba=$NUMBA_VERSION scikit-learn \
#        pytest "tensorflow-mkl>=2.2.0"
  if [ $TRAVIS_OS_NAME = 'osx' ]; then
    conda create -q -n testenv --yes python=$PYTHON_VERSION numpy scipy scikit-learn \
          numba pytest pandas
#    pip install bokeh
#    pip install datashader
#    pip install holoviews
    conda install --yes "tensorflow>=2.0.0"
  else
    conda create -q -n testenv --yes python=$PYTHON_VERSION numpy scipy scikit-learn \
          numba pandas bokeh holoviews datashader scikit-image pytest pytest-benchmark \
          "tensorflow-mkl>=2.2.0"
  fi

  source activate testenv

  # black requires Python 3.x; don't try to install for Python 2.7 test
  if [[ "$PYTHON_VERSION" != "2.7" ]]; then
    pip install black
    pip install pynndescent
  fi

  if [[ "$COVERAGE" == "true" ]]; then
      pip install coverage coveralls
      pip install pytest-cov pytest-benchmark # pytest coverage plugin
  fi

  python --version
  python -c "import numpy; print('numpy %s' % numpy.__version__)"
  python -c "import scipy; print('scipy %s' % scipy.__version__)"
  python -c "import numba; print('numba %s' % numba.__version__)"
  python -c "import sklearn; print('scikit-learn %s' % sklearn.__version__)"
  python setup.py develop
else
  pip install pynndescent # test with optional pynndescent dependency
  pip install pandas
  pip install bokeh
  pip install datashader
  pip install matplotlib
  pip install holoviews
  pip install scikit-image
  pip install "tensorflow>=2.2.0"
  pip install -e .
fi
