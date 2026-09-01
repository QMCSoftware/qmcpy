import ast
import builtins
from pathlib import Path

import pytest


def _execute_optional_import(blocked_import):
    repository_root = Path(__file__).resolve().parent.parent
    init_path = repository_root / "qmcpy" / "__init__.py"
    init_tree = ast.parse(init_path.read_text())
    optional_import = next(
        node
        for node in init_tree.body
        if isinstance(node, ast.Try)
        and any(
            isinstance(statement, ast.ImportFrom)
            and statement.module == "discrete_distribution.mpmc"
            for statement in node.body
        )
    )

    import qmcpy

    real_import = builtins.__import__

    def guarded_import(name, globals=None, locals=None, fromlist=(), level=0):
        missing_module = blocked_import(name, fromlist, level)
        if missing_module is not None:
            raise ModuleNotFoundError(
                "blocked optional dependency",
                name=missing_module,
            )
        return real_import(name, globals, locals, fromlist, level)

    test_builtins = vars(builtins).copy()
    test_builtins["__import__"] = guarded_import
    namespace = {"__builtins__": test_builtins, "__package__": "qmcpy"}
    module = ast.Module(body=[optional_import], type_ignores=[])
    exec(compile(module, str(init_path), "exec"), namespace)
    return namespace


def test_mpmc_utils_remain_available_without_pyg():
    pytest.importorskip("torch")

    def block_pyg_models(name, fromlist, level):
        if level == 1 and name == "discrete_distribution.mpmc.models":
            return "torch_geometric"
        return None

    namespace = _execute_optional_import(block_pyg_models)

    import qmcpy

    assert namespace["mpmc_utils"] is qmcpy.mpmc_utils
    assert namespace["mpmc_utils"].__name__ == (
        "qmcpy.discrete_distribution.mpmc.utils"
    )
    assert "utils" not in namespace

    with pytest.raises(ModuleNotFoundError, match="MPMC_net.*torch_geometric") as error:
        namespace["MPMC_net"]()
    assert error.value.name == "torch_geometric"


def test_mpmc_placeholders_report_missing_torch():
    def block_torch_utils(name, fromlist, level):
        if (
            level == 1
            and name == "discrete_distribution.mpmc"
            and "utils" in fromlist
        ):
            return "torch"
        return None

    namespace = _execute_optional_import(block_torch_utils)

    with pytest.raises(ModuleNotFoundError, match="mpmc_utils.*torch") as error:
        namespace["mpmc_utils"].L2star
    assert error.value.name == "torch"

    with pytest.raises(ModuleNotFoundError, match="MPMC_net.*torch") as error:
        namespace["MPMC_net"]()
    assert error.value.name == "torch"


def test_mpmc_placeholder_missing_torch_scatter():
    pytest.importorskip("torch")

    def block_torch_scatter(name, fromlist, level):
        if level == 1 and name == "discrete_distribution.mpmc.models":
            return "torch_scatter"
        return None

    namespace = _execute_optional_import(block_torch_scatter)

    with pytest.raises(ModuleNotFoundError, match="MPMC_net.*torch_scatter") as error:
        namespace["MPMC_net"]()
    assert error.value.name == "torch_scatter"
