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
       http://repo.continuum.io/miniconda/Miniconda-3.6.0-Linux-x86_64.sh \
         -O miniconda.sh
     fi
#  fi
  chmod +x miniconda.sh && ./miniconda.sh -b -p $HOME/miniconda
  cd ..
  export PATH=$HOME/miniconda/bin:$PATH
  conda update --yes conda
  popd

  # Configure the conda environment and put it in the path using the
  # provided versions
  conda create -n testenv --yes python=$PYTHON_VERSION pip nose \
        numpy=$NUMPY_VERSION scipy=$SCIPY_VERSION numba scikit-learn

  source activate testenv


  if [[ "$COVERAGE" == "true" ]]; then
      pip install coverage coveralls
  fi

  python --version
  python -c "import numpy; print('numpy %s' % numpy.__version__)"
  python -c "import scipy; print('scipy %s' % scipy.__version__)"
  python setup.py develop
else
  pip install -e .
fi
