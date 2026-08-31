#!/usr/bin/env python3
"""Check ``test/test_*.py`` files against two suite conventions.

1. **Object class.** A test file should be written as a ``unittest.TestCase``
   subclass, not as bare ``def test_*`` pytest functions.  The numeric-correctness
   backbone (``test_tm_true_measures.py``, ``test_sc_stopping_criteria.py``,
   ``test_dd_discrete_distribs.py``, ...) already follows this; newer per-measure
   and tooling files do not.  The split is listed so it stays visible in review.

2. **Area prefix.** A test file should be named ``test_<area>_<rest>.py`` where
   ``<area>`` marks the ``qmcpy`` subpackage under test (or a cross-cutting
   bucket).  Recognized areas:

       dd    discrete_distribution    tm    true_measure
       ft    fast_transform           ut    util
       ig    integrand                ee    end-to-end / cross-cutting pipeline
       kn    kernel                   sr    scripts/ tooling, packaging, docs checks
       sc    stopping_criterion

Usage:
    python scripts/check_test_style.py [TEST_DIR] [--strict] [--quiet]

TEST_DIR defaults to ``test``.  With ``--strict`` the exit code is non-zero when
any file violates either convention (so it can gate CI); otherwise it is always
0 and the output is informational.
"""
import ast
import re
import sys
from pathlib import Path

AREA_PREFIXES = {
    "dd": "discrete_distribution",
    "ft": "fast_transform",
    "ig": "integrand",
    "kn": "kernel",
    "sc": "stopping_criterion",
    "tm": "true_measure",
    "ut": "util",
    "ee": "end-to-end / cross-cutting pipeline",
    "sr": "scripts/ tooling, packaging, docs checks",
}
AREA_RE = re.compile(r"^test_(?:" + "|".join(sorted(AREA_PREFIXES)) + r")_.+\.py$")


def _area_ok(path):
    """True if the filename starts with a recognized ``test_<area>_`` prefix."""
    return bool(AREA_RE.match(path.name))


def _subclasses_testcase(node):
    """True if a ClassDef lists ``TestCase`` / ``unittest.TestCase`` as a base."""
    for base in node.bases:
        if isinstance(base, ast.Attribute) and base.attr == "TestCase":
            return True
        if isinstance(base, ast.Name) and base.id == "TestCase":
            return True
    return False


def classify(path):
    """Return (has_testcase_class, has_test_callables)."""
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    nodes = list(ast.walk(tree))
    has_class = any(
        isinstance(n, ast.ClassDef) and _subclasses_testcase(n) for n in nodes
    )
    has_tests = any(
        isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
        and n.name.startswith("test_")
        for n in nodes
    )
    return has_class, has_tests


def main(argv):
    strict = "--strict" in argv
    quiet = "--quiet" in argv
    positional = [a for a in argv if not a.startswith("-")]
    test_dir = Path(positional[0]) if positional else Path("test")

    files = sorted(test_dir.glob("test_*.py"))
    if not files:
        print(f"no test_*.py files under {test_dir}/", file=sys.stderr)
        return 1

    class_based, function_based, no_tests = [], [], []
    for f in files:
        has_class, has_tests = classify(f)
        if has_class:
            class_based.append(f)
        elif has_tests:
            function_based.append(f)
        else:
            no_tests.append(f)

    misnamed = [f for f in files if not _area_ok(f)]

    if not quiet:
        print(f"{len(class_based)}/{len(files)} file(s) use a unittest.TestCase class")
    if function_based:
        print(
            f"{len(function_based)} file(s) use bare pytest functions "
            f"(no unittest.TestCase class):"
        )
        for f in function_based:
            print(f"  {f.as_posix()}")
    elif not quiet:
        print("  no bare-function test files found")
    if no_tests and not quiet:
        print(f"{len(no_tests)} file(s) define no test_* callables:")
        for f in no_tests:
            print(f"  {f.as_posix()}")

    if not quiet:
        print(
            f"{len(files) - len(misnamed)}/{len(files)} file(s) use a "
            f"test_<area>_ prefix ({', '.join(sorted(AREA_PREFIXES))})"
        )
    if misnamed:
        print(f"  {len(misnamed)} file(s) have no recognized test_<area>_ prefix:")
        for f in misnamed:
            print(f"  {f.as_posix()}")
    elif not quiet:
        print("  no misnamed test files found")

    return 1 if (strict and (function_based or misnamed)) else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
