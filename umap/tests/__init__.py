"""
Test Suite for UMAP to ensure things are working as expected.

The test suite comprises multiple testing modules,
including multiple test cases related to a specific
set of UMAP features under test.

Backend
-------
pytest is the reference backend for testing environment and execution,
also integrating with pre-existent nose-based tests

Shared Testing code
-------------------
Whenever needed, each module includes a set of
_utility_ functions that specify shared (and repeated)
testing operations.

Fixtures
--------
All data dependency has been implemented
as test fixtures (preferred to shared global variables).
All the fixtures shared by multiple test cases
are defined in the `conftest.py` module.

Fixtures allow the execution of each test module in isolation, as well
as within the whole test suite.

Modules in Tests (to keep up to date)
-------------------------------------
- conftest: pytrest fixtures
- test_plot: basic tests for umap.plot
- test_umap_df_validation_params:
    Tests on parameters validation for DataFrameUMAP
- test_umap_metrics:
    Tests for UMAP metrics - spatial, binary, and sparse
- test_umap_nn:
    Tests for NearestNeighbours
- test_umap_on_iris:
    Tests for UMAP on Iris Dataset
- test_umap_ops:
    Tests for general UMAP ops (e.g. clusterability, transform stability)
- test_umap_repeated_data:
    UMAP tests on repeated data (sparse|dense; spatial|binary)
- test_umap_trustworthiness:
    Tests on UMAP Trustworthiness
- test_umap_validation_params:
    Tests for fit parameters validation

"""
