"""Pytest configuration shared by unit tests and source doctests.

`DOCTEST_WARNING_FILTERS` suppresses expected warnings only for doctests in
the named source files. Add filters using pytest's `filterwarnings` syntax and
fully qualify custom warning classes. Keep this file at the repository root so
pytest discovers it when collecting both `qmcpy/` and `test/`.
"""

import pytest

# Maps a source filename to the expected-warning filter patterns its own
# doctest examples deliberately trigger, so those doctests read as passing
# examples of documented/intentional behavior rather than noisy warnings.
DOCTEST_WARNING_FILTERS = {
    "matern_gp.py": [
        "ignore:MaternGP.variance now returns.*:DeprecationWarning",
    ],
    "latin_hypercube.py": [
        "ignore:randomize=False only fixes.*:qmcpy.util.exceptions_warnings.ParameterWarning",
    ],
}


def pytest_collection_modifyitems(config, items):
    """Scope expected-warning filters to specific source files' doctests,
    without adding a pytest import to library code (pytest is a test-only
    dependency, not a runtime one)."""
    for item in items:
        for pattern in DOCTEST_WARNING_FILTERS.get(item.path.name, ()):
            item.add_marker(pytest.mark.filterwarnings(pattern))
