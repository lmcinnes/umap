set -e

if [[ "$COVERAGE" == "true" ]]; then
    black --check $MODULE
fi

# Get into a temp directory to run test from the installed scikit learn and
# check if we do not leave artifacts
#mkdir -p $TEST_DIR

#cd $TEST_DIR

if [[ "$COVERAGE" == "true" ]]; then
    export NUMBA_DISABLE_JIT=1
    coverage run -m pytest --show-capture=no -v --disable-warnings --basetemp=$TEST_DIR
else
    pytest --show-capture=no -v --disable-warnings --basetemp=$TEST_DIR
fi
