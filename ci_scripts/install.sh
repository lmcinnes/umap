if [[ "$DISTRIB" == "conda" ]]; then

  # Deactivate the travis-provided virtual environment and setup a
    # conda-based environment instead
  deactivate

  # Use the miniconda installer for faster download / install of conda
  # itself
  pushd .
  cd
  mkdir -p download
  cd download
  echo "Cached in $HOME/download :"
  ls -l
  echo
  if [[ ! -f miniconda.sh ]]
     then
     wget http://repo.continuum.io/miniconda/Miniconda-3.6.0-Linux-x86_64.sh \
         -O miniconda.sh
     fi
  chmod +x miniconda.sh && ./miniconda.sh -b
  cd ..
  export PATH=/home/travis/miniconda/bin:$PATH
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
