"""
Tests for UMAP to ensure things are working as expected.

The test suite has been refactored to support PyTest.
This allow to prefer test fixtures over global variable.
All the fixtures are defined in the `conftest.py` module.

Moreover, test cases have been re-organised in different
sections (implemented as different modules)
according to the specific set of UMAP features under test.

Each test module/section include a set of utility functions -
defined on top of each section - which are meant to define the
core processing instructions required by (most of) the tests
so to avoid code clones (repetitions) as much as possible.
This is to make the testing code easier to maintain.

Moreover, the multiple testing sections/modules are well-integrated
each other, as data are now pytest fixtures and no more global variables.

Therefore:
- easy to run specific (subsets) of tests
- easy to add additional tests to specific sections
- avoiding code repetitions and multiple dependencies
    (pytest will handle fixtures deps auto-magically)

"""