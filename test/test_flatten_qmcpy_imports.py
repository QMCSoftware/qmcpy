import json
from pathlib import Path

from scripts.flatten_qmcpy_imports import (
    _load_qmcpy_public_names,
    flatten_imports,
    main,
)


def _nested_import(module, imported):
    return f"from {'qmcpy.' + module} import {imported}"


def test_flatten_imports_basic():
    source = (
        _nested_import("integrand", "Keister")
        + "\n"
        + _nested_import("discrete_distribution.lattice", "Lattice as LD")
        + "\nfrom qmcpy import DigitalNetB2\nimport qmcpy.util\n"
    ).encode()

    updated, count = flatten_imports(
        source, frozenset({"DigitalNetB2", "Keister", "Lattice"})
    )

    assert count == 3
    assert updated == (
        b"from qmcpy import DigitalNetB2, Keister, Lattice as LD\n"
        b"import qmcpy.util\n"
    )


def test_flatten_preserves_private():
    source = (
        _nested_import("_internal._helpers", "PublicHelper")
        + "\n"
        + _nested_import(
            "true_measure.uniform_triangle",
            "UniformTriangle, _UniformTriangleAdapter",
        )
        + "\n"
        + _nested_import(
            "true_measure.copula",
            "(\n    AbstractCopula,\n    _validate_dimension,\n)",
        )
        + "\n"
        + _nested_import("integrand", "Keister")
        + "\n"
    ).encode()

    updated, count = flatten_imports(source, frozenset({"Keister"}))

    assert count == 1
    assert updated == source.replace(
        _nested_import("integrand", "Keister").encode(),
        b"from qmcpy import Keister",
    )


def test_private_module_splits_groups():
    source = (
        b"from qmcpy import Zeta\n"
        b"from qmcpy._internal._helpers import PublicHelper\n"
        b"from qmcpy import Alpha\n"
    )

    updated, count = flatten_imports(source)

    assert count == 0
    assert updated == source


def test_flatten_preserves_util_imports():
    source = (
        b"from qmcpy.util import ParameterError\n"
        b"from qmcpy.util.transforms import tf_exp\n"
    )

    updated, count = flatten_imports(source, frozenset({"ParameterError", "tf_exp"}))

    assert (updated, count) == (source, 0)


def test_flatten_keeps_nonpublic_names():
    source = b"from qmcpy.stopping_criterion.pf_gp_ci import PFGPCIData\n"

    updated, count = flatten_imports(source, frozenset({"PFGPCI"}))

    assert (updated, count) == (source, 0)


def test_flatten_no_public_api_noop():
    source = (_nested_import("integrand", "Keister") + "\n").encode()

    updated, count = flatten_imports(source)

    assert (updated, count) == (source, 0)


def test_flatten_preserve_str_literals():
    source = b'text = """\nfrom qmcpy.integrand import Keister\n"""\n'

    updated, count = flatten_imports(source, frozenset({"Keister"}))

    assert count == 0
    assert updated == source


def test_python_string_protection_applies_to_every_rewrite_stage():
    string_body = (
        b'text = """\n'
        b"from qmcpy.integrand import Keister\n"
        b"from qmcpy import Zeta,Beta\n"
        b"from qmcpy import Alpha\n"
        b"from qmcpy import *\n"
        b"from qmcpy import *\n"
        b'"""\n'
    )
    source = string_body + b"from qmcpy.integrand import Keister\n"

    updated, count = flatten_imports(source, frozenset({"Keister"}))

    assert count == 1
    assert updated == string_body + b"from qmcpy import Keister\n"


def test_python_tokenize_failure_is_fail_closed():
    source = b'"""unterminated\nfrom qmcpy.integrand import Keister\n'

    assert flatten_imports(source, frozenset({"Keister"})) == (source, 0)


def test_flatten_skip_star_expansion():
    source = (
        b"from qmcpy import *\n\n"
        b"def f(Lattice):\n"
        b"    return Lattice\n\n"
        b"y = Keister(dimension=2)\n"
        b"x = Lattice(dimension=2)\n"
    )

    updated, count = flatten_imports(source, frozenset({"Keister", "Lattice"}))

    assert count == 0
    assert updated == source


def test_notebook_star_dedup():
    notebook = {
        "cells": [
            {
                "cell_type": "code",
                "source": [
                    _nested_import("integrand", "*") + "\n",
                    _nested_import("true_measure", "*"),
                ],
            }
        ]
    }
    source = json.dumps(notebook, indent=1).encode()

    updated, count = flatten_imports(source, frozenset({"Keister"}))

    assert count == 3
    assert json.loads(updated)["cells"][0]["source"] == ["from qmcpy import *"]


def test_named_imports_merge_sort():
    source = (
        b"from qmcpy import Zeta,Beta\n"
        b"from qmcpy import Alpha\n"
        b"\n"
        b"from qmcpy import Gamma\n"
    )

    updated, count = flatten_imports(source)

    assert count == 1
    assert updated == (
        b"from qmcpy import Alpha, Beta, Zeta\n"
        b"\n"
        b"from qmcpy import Gamma\n"
    )
    assert flatten_imports(updated) == (updated, 0)


def test_merge_paren_and_single_line():
    source = b"""from qmcpy import (
    KernelDigShiftInvar,
    KernelDigShiftInvarAdaptiveAlpha,
    KernelDigShiftInvarCombined,
    KernelShiftInvar,
    KernelShiftInvarCombined,
)
from qmcpy import tf_exp_eps, tf_exp_eps_inv
"""

    updated, count = flatten_imports(source)

    assert count == 1
    assert updated == b"""from qmcpy import (
    KernelDigShiftInvar,
    KernelDigShiftInvarAdaptiveAlpha,
    KernelDigShiftInvarCombined,
    KernelShiftInvar,
    KernelShiftInvarCombined,
    tf_exp_eps,
    tf_exp_eps_inv,
)
"""
    assert flatten_imports(updated) == (updated, 0)


def test_merge_same_scope_only():
    source = (
        b"if enabled:\n"
        b"    from qmcpy import Zeta\n"
        b"    from qmcpy import Alpha as First\n"
        b"else:\n"
        b"    from qmcpy import Beta\n"
        b"from qmcpy import _Private\n"
        b"from qmcpy import Gamma  # keep this comment\n"
    )

    updated, count = flatten_imports(source)

    assert count == 1
    assert updated == (
        b"if enabled:\n"
        b"    from qmcpy import Alpha as First, Zeta\n"
        b"else:\n"
        b"    from qmcpy import Beta\n"
        b"from qmcpy import _Private\n"
        b"from qmcpy import Gamma  # keep this comment\n"
    )


def test_notebook_named_merge():
    notebook = {
        "cells": [
            {
                "cell_type": "code",
                "source": [
                    "from qmcpy import Zeta\n",
                    "from qmcpy import Alpha,Beta\n",
                    "print(Alpha)\n",
                ],
            }
        ]
    }
    source = json.dumps(notebook, indent=1).encode()

    updated, count = flatten_imports(source)

    assert count == 1
    assert json.loads(updated)["cells"][0]["source"] == [
        "from qmcpy import Alpha, Beta, Zeta\n",
        "print(Alpha)\n",
    ]
    assert flatten_imports(updated) == (updated, 0)


def test_notebook_flattens_nested_imports_only_in_code_cells():
    nested_import = _nested_import("integrand", "Keister") + "\n"
    metadata_import = _nested_import("true_measure", "Gaussian") + "\n"
    string_literal = f'text = "{nested_import.rstrip()}"\n'
    multiline_string = ['text = """\n', nested_import, '"""\n']
    notebook = {
        "metadata": {"source": [metadata_import]},
        "cells": [
            {"cell_type": "markdown", "source": [nested_import]},
            {"cell_type": "code", "source": [nested_import]},
            {"cell_type": "code", "source": [string_literal]},
            {"cell_type": "code", "source": multiline_string},
        ]
    }
    source = json.dumps(notebook, indent=1).encode()

    updated, count = flatten_imports(source, frozenset({"Keister"}))

    cells = json.loads(updated)["cells"]
    assert count == 1
    assert json.loads(updated)["metadata"]["source"] == [metadata_import]
    assert cells[0]["source"] == [nested_import]
    assert cells[1]["source"] == ["from qmcpy import Keister\n"]
    assert cells[2]["source"] == [string_literal]
    assert cells[3]["source"] == multiline_string
    assert flatten_imports(updated, frozenset({"Keister"})) == (updated, 0)


def test_markdown_import_examples_are_flattened(tmp_path):
    path = tmp_path / "example.md"
    path.write_bytes(
        b'Example with unmatched prose delimiter: """\n\n'
        b"```python\n"
        b"from qmcpy.integrand import Keister\n"
        b"```\n"
    )

    assert main([str(path)]) == 0
    assert b"from qmcpy import Keister" in path.read_bytes()


def test_check_mode_no_write(tmp_path):
    path = tmp_path / "example.py"
    original = (_nested_import("true_measure", "Gaussian") + "\n").encode()
    path.write_bytes(original)

    assert main(["--check", str(path)]) == 1
    assert path.read_bytes() == original

    assert main([str(path)]) == 0
    assert path.read_bytes() == b"from qmcpy import Gaussian\n"
    assert main(["--check", str(path)]) == 0


def test_public_names_optional_free_stable():
    repository_root = Path(__file__).resolve().parent.parent
    names = _load_qmcpy_public_names(repository_root)

    assert names is not None
    assert "Gaussian" in names
    assert "Keister" in names
    # Optional dependencies are blocked in the probe context, so fallback
    # exports are part of the deterministic name set.
    assert "PFGPCI" in names
    # Helpers that are deliberately not part of the top-level API.
    assert "PFGPCIData" not in names
    assert "TriangularDistribution" not in names