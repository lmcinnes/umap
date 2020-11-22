set -e

if [[ "$COVERAGE" == "true" ]]; then
    black --check $MODULE
fi

if [[ "$COVERAGE" == "true" ]]; then
    export NUMBA_DISABLE_JIT=1
    pytest --cov=umap/ --show-capture=no -v --disable-warnings
else
    pytest --show-capture=no -v --disable-warnings
fi
