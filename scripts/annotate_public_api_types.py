#!/usr/bin/env python3
"""Add conservative public-API annotations from Google-style docstrings.

Only public module functions, public methods, and constructors of public
classes are considered. A docstring type is applied only when it is valid
Python annotation syntax and every referenced name is already bound by the
module or is a built-in type. Existing annotations are never overwritten;
conflicts are reported for review.
"""
from __future__ import annotations

import argparse
import ast
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import libcst as cst
from libcst.metadata import MetadataWrapper, PositionProvider

from scripts import add_docstring_arg_types as docstrings


BUILTIN_TYPE_NAMES = {
    "bool",
    "bytearray",
    "bytes",
    "complex",
    "dict",
    "float",
    "frozenset",
    "int",
    "list",
    "memoryview",
    "object",
    "range",
    "set",
    "slice",
    "str",
    "tuple",
    "type",
}


@dataclass(frozen=True)
class FunctionSpec:
    """Docstring-derived annotations for one public callable."""

    function: str
    line: int
    arguments: dict[str, str]
    rejected_arguments: dict[str, tuple[str, str]]
    return_type: str | None


@dataclass(frozen=True)
class Update:
    """One annotation inserted into a function signature."""

    path: Path
    line: int
    function: str
    slot: str
    annotation: str


@dataclass(frozen=True)
class Conflict:
    """A disagreement between an annotation and its docstring type."""

    path: Path
    line: int
    function: str
    slot: str
    signature_type: str
    docstring_type: str


@dataclass(frozen=True)
class Skip:
    """A docstring type that is unsafe to place in a signature."""

    path: Path
    line: int
    function: str
    slot: str
    docstring_type: str
    reason: str


@dataclass(frozen=True)
class UnsafeExistingAnnotation:
    """An existing annotation that repeats a rejected docstring type."""

    path: Path
    line: int
    function: str
    slot: str
    annotation: str
    reason: str


@dataclass(frozen=True)
class SourceResult:
    """Result of analyzing and transforming one source string."""

    source: str
    updates: tuple[Update, ...]
    conflicts: tuple[Conflict, ...]
    skips: tuple[Skip, ...]
    unsafe_existing: tuple[UnsafeExistingAnnotation, ...]


@dataclass(frozen=True)
class FileResult:
    """Result of inspecting one Python file."""

    path: Path
    updates: tuple[Update, ...]
    conflicts: tuple[Conflict, ...]
    skips: tuple[Skip, ...]
    unsafe_existing: tuple[UnsafeExistingAnnotation, ...]
    changed: bool


def _module_names(tree: ast.Module, before_line: int) -> set[str]:
    """Collect module names bound before a callable's definition."""
    names = set(BUILTIN_TYPE_NAMES)
    for node in tree.body:
        if getattr(node, "lineno", before_line) >= before_line:
            continue
        if isinstance(node, ast.Import):
            for alias in node.names:
                names.add(alias.asname or alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom):
            for alias in node.names:
                if alias.name != "*":
                    names.add(alias.asname or alias.name)
        elif isinstance(
            node,
            (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef),
        ):
            names.add(node.name)
        elif isinstance(node, (ast.Assign, ast.AnnAssign, ast.NamedExpr)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            for target in targets:
                if isinstance(target, ast.Name):
                    names.add(target.id)
    return names


def _annotation_expression(
    text: str,
    available_names: set[str],
) -> tuple[str | None, str | None]:
    """Validate and normalize a docstring type for runtime-safe insertion."""
    text = text.strip()
    try:
        expression = ast.parse(text, mode="eval").body
    except SyntaxError:
        return None, "not valid Python annotation syntax"

    if isinstance(expression, ast.Constant) and isinstance(expression.value, str):
        return None, "string descriptions are not inserted as annotations"
    unsafe = (
        ast.BoolOp,
        ast.Call,
        ast.Compare,
        ast.Dict,
        ast.DictComp,
        ast.GeneratorExp,
        ast.IfExp,
        ast.Lambda,
        ast.ListComp,
        ast.Set,
        ast.SetComp,
    )
    if any(isinstance(node, unsafe) for node in ast.walk(expression)):
        return None, "contains an expression that is unsafe in an annotation"
    if any(isinstance(node, ast.BinOp) for node in ast.walk(expression)):
        return None, "uses an operator that is not safe for Python 3.9 annotations"

    referenced_names = {
        node.id for node in ast.walk(expression) if isinstance(node, ast.Name)
    }
    missing = sorted(referenced_names - available_names)
    if missing:
        return None, f"name(s) not available in module: {', '.join(missing)}"

    normalized = ast.unparse(expression)
    try:
        cst.parse_expression(normalized)
    except cst.ParserSyntaxError:
        return None, "cannot be represented by LibCST"
    return normalized, None


def _argument_defaults(
    node: ast.FunctionDef | ast.AsyncFunctionDef,
) -> dict[str, ast.expr]:
    """Map parameter names to defaults present in the signature."""
    positional = list(node.args.posonlyargs) + list(node.args.args)
    defaults = {}
    if node.args.defaults:
        defaults.update(
            {
                argument.arg: default
                for argument, default in zip(
                    positional[-len(node.args.defaults):],
                    node.args.defaults,
                )
            }
        )
    defaults.update(
        {
            argument.arg: default
            for argument, default in zip(
                node.args.kwonlyargs,
                node.args.kw_defaults,
            )
            if default is not None
        }
    )
    return defaults


def _default_compatibility_reason(
    annotation: str,
    default: ast.expr | None,
) -> str | None:
    """Reject obvious contradictions between an annotation and a default."""
    if default is None:
        return None
    expression = ast.parse(annotation, mode="eval").body
    identifiers = {
        node.id for node in ast.walk(expression) if isinstance(node, ast.Name)
    }
    identifiers.update(
        node.attr for node in ast.walk(expression) if isinstance(node, ast.Attribute)
    )
    if identifiers & {"Any", "object"}:
        return None

    try:
        value = ast.literal_eval(default)
    except (ValueError, TypeError):
        return None

    if value is None:
        permits_none = "Optional" in identifiers or any(
            isinstance(node, ast.Constant) and node.value is None
            for node in ast.walk(expression)
        )
        if not permits_none:
            return "default is None but the documented type is not optional"
        return None

    if isinstance(value, bool):
        compatible = {"bool"}
    elif isinstance(value, int):
        compatible = {"complex", "float", "int", "Integral", "Number", "Real"}
    elif isinstance(value, float):
        compatible = {"complex", "float", "Number", "Real"}
    elif isinstance(value, str):
        compatible = {"str"}
    elif isinstance(value, bytes):
        compatible = {"bytes"}
    elif isinstance(value, list):
        compatible = {
            "Collection",
            "Iterable",
            "List",
            "MutableSequence",
            "Sequence",
            "list",
        }
    elif isinstance(value, tuple):
        compatible = {"Collection", "Iterable", "Sequence", "Tuple", "tuple"}
    elif isinstance(value, dict):
        compatible = {"Dict", "Mapping", "MutableMapping", "dict"}
    elif isinstance(value, (set, frozenset)):
        compatible = {
            "AbstractSet",
            "Collection",
            "FrozenSet",
            "Iterable",
            "Set",
            "frozenset",
            "set",
        }
    else:
        return None
    if identifiers & compatible:
        return None
    return (
        f"default value of type {type(value).__name__} conflicts with "
        "the documented type"
    )


def _docstring_argument_types(
    node: ast.FunctionDef | ast.AsyncFunctionDef,
    lines: list[str],
) -> dict[str, tuple[str, int]]:
    """Extract explicit Google-style argument types and their source lines."""
    dnode = docstrings.doc_node(node)
    if dnode is None or dnode.end_lineno is None:
        return {}
    section = docstrings.find_args_section(
        lines,
        dnode.lineno - 1,
        dnode.end_lineno - 1,
    )
    if section is None:
        return {}

    types = {}
    for i in range(section[0] + 1, section[1] + 1):
        content, _ = docstrings.line_without_ending(lines[i])
        match = docstrings.ARG_ENTRY.match(content)
        if match is None or match.group("type") is None:
            continue
        name = match.group("name").lstrip("*")
        types[name] = (match.group("type").strip(), i + 1)
    return types


def _docstring_output_type(
    node: ast.FunctionDef | ast.AsyncFunctionDef,
    lines: list[str],
) -> tuple[str, int] | None:
    """Extract an explicit aggregate type from an existing Returns section."""
    dnode = docstrings.doc_node(node)
    if dnode is None or dnode.end_lineno is None:
        return None
    section = docstrings.find_section(
        lines,
        dnode.lineno - 1,
        dnode.end_lineno - 1,
        "Returns",
    )
    if section is None:
        return None

    header, _ = docstrings.line_without_ending(lines[section[0]])
    header_indent = len(header) - len(header.lstrip())
    for i in range(section[0] + 1, section[1] + 1):
        content, _ = docstrings.line_without_ending(lines[i])
        if not content.strip():
            continue
        indent = len(content) - len(content.lstrip())
        if indent <= header_indent:
            continue
        match = docstrings.OUTPUT_ENTRY.match(content)
        if match is None:
            return None
        candidate = match.group("type").strip()
        if not docstrings.looks_like_type(candidate):
            return None
        return candidate, i + 1
    return None


def _collect_specs(
    source: str,
    path: Path,
) -> tuple[dict[tuple[int, str], FunctionSpec], list[Skip]]:
    """Collect validated docstring types for public functions and methods."""
    tree = ast.parse(source, filename=str(path))
    lines = source.splitlines(keepends=True)
    specs = {}
    skips = []

    for node, function in docstrings.iter_public_functions(tree):
        available_names = _module_names(tree, node.lineno)
        if "." in function:
            # A class is not bound to its module name until its body finishes.
            available_names.discard(function.split(".", maxsplit=1)[0])
        arguments = {}
        rejected_arguments = {}
        defaults = _argument_defaults(node)
        for name, (text, line) in _docstring_argument_types(node, lines).items():
            annotation, reason = _annotation_expression(text, available_names)
            if annotation is not None:
                reason = _default_compatibility_reason(
                    annotation,
                    defaults.get(name),
                )
                if reason is not None:
                    annotation = None
            if annotation is None:
                rejection_reason = reason or "unsafe"
                rejected_arguments[name] = (text, rejection_reason)
                skips.append(
                    Skip(path, line, function, name, text, rejection_reason)
                )
            else:
                arguments[name] = annotation

        return_type = "None" if node.name == "__init__" else None
        output = _docstring_output_type(node, lines)
        if output is not None and node.name != "__init__":
            text, line = output
            annotation, reason = _annotation_expression(text, available_names)
            if annotation is None:
                skips.append(
                    Skip(path, line, function, "return", text, reason or "unsafe")
                )
            else:
                return_type = annotation

        specs[(node.lineno, node.name)] = FunctionSpec(
            function=function,
            line=node.lineno,
            arguments=arguments,
            rejected_arguments=rejected_arguments,
            return_type=return_type,
        )
    return specs, skips


def _annotation_code(annotation: cst.Annotation) -> str:
    """Render one LibCST annotation expression without surrounding syntax."""
    return cst.Module(body=[]).code_for_node(annotation.annotation)


def _normalized_annotation(text: str) -> str:
    """Normalize annotations for conflict comparison."""
    try:
        expression = ast.parse(text, mode="eval").body
    except SyntaxError:
        return re.sub(r"\s+", "", text)
    if isinstance(expression, ast.Constant) and isinstance(expression.value, str):
        try:
            expression = ast.parse(expression.value, mode="eval").body
        except SyntaxError:
            return expression.value
    return ast.dump(expression, include_attributes=False)


class PublicAPIAnnotationTransformer(cst.CSTTransformer):
    """Insert validated docstring types into matching public signatures."""

    METADATA_DEPENDENCIES = (PositionProvider,)

    def __init__(self, path: Path, specs: dict[tuple[int, str], FunctionSpec]):
        self.path = path
        self.specs = specs
        self.updates: list[Update] = []
        self.conflicts: list[Conflict] = []
        self.unsafe_existing: list[UnsafeExistingAnnotation] = []

    def _update_param(
        self,
        original: cst.Param,
        updated: cst.Param,
        spec: FunctionSpec,
    ) -> cst.Param:
        """Annotate one parameter or report a signature/docstring conflict."""
        name = original.name.value
        desired = spec.arguments.get(name)
        line = self.get_metadata(PositionProvider, original.name).start.line
        rejected = spec.rejected_arguments.get(name)
        if desired is None:
            if original.annotation is not None and rejected is not None:
                existing = _annotation_code(original.annotation)
                rejected_type, reason = rejected
                if _normalized_annotation(existing) == _normalized_annotation(
                    rejected_type
                ):
                    self.unsafe_existing.append(
                        UnsafeExistingAnnotation(
                            self.path,
                            line,
                            spec.function,
                            name,
                            existing,
                            reason,
                        )
                    )
            return updated
        if original.annotation is not None:
            existing = _annotation_code(original.annotation)
            if _normalized_annotation(existing) != _normalized_annotation(desired):
                self.conflicts.append(
                    Conflict(
                        self.path,
                        line,
                        spec.function,
                        name,
                        existing,
                        desired,
                    )
                )
            return updated

        self.updates.append(Update(self.path, line, spec.function, name, desired))
        changes = {"annotation": cst.Annotation(cst.parse_expression(desired))}
        if updated.default is not None and isinstance(updated.equal, cst.AssignEqual):
            changes["equal"] = updated.equal.with_changes(
                whitespace_before=cst.SimpleWhitespace(" "),
                whitespace_after=cst.SimpleWhitespace(" "),
            )
        return updated.with_changes(**changes)

    def leave_FunctionDef(
        self,
        original_node: cst.FunctionDef,
        updated_node: cst.FunctionDef,
    ) -> cst.FunctionDef:
        """Update an eligible function or method signature."""
        line = self.get_metadata(PositionProvider, original_node.name).start.line
        spec = self.specs.get((line, original_node.name.value))
        if spec is None:
            return updated_node

        original_params = original_node.params
        updated_params = updated_node.params
        posonly_params = tuple(
            self._update_param(original, updated, spec)
            for original, updated in zip(
                original_params.posonly_params,
                updated_params.posonly_params,
            )
        )
        params = tuple(
            self._update_param(original, updated, spec)
            for original, updated in zip(original_params.params, updated_params.params)
        )
        kwonly_params = tuple(
            self._update_param(original, updated, spec)
            for original, updated in zip(
                original_params.kwonly_params,
                updated_params.kwonly_params,
            )
        )
        star_arg = updated_params.star_arg
        if isinstance(original_params.star_arg, cst.Param) and isinstance(
            updated_params.star_arg, cst.Param
        ):
            star_arg = self._update_param(
                original_params.star_arg,
                updated_params.star_arg,
                spec,
            )
        star_kwarg = updated_params.star_kwarg
        if original_params.star_kwarg is not None and star_kwarg is not None:
            star_kwarg = self._update_param(
                original_params.star_kwarg,
                star_kwarg,
                spec,
            )

        returns = updated_node.returns
        if spec.return_type is not None:
            if original_node.returns is None:
                self.updates.append(
                    Update(
                        self.path,
                        line,
                        spec.function,
                        "return",
                        spec.return_type,
                    )
                )
                returns = cst.Annotation(cst.parse_expression(spec.return_type))
            else:
                existing = _annotation_code(original_node.returns)
                if _normalized_annotation(existing) != _normalized_annotation(
                    spec.return_type
                ):
                    self.conflicts.append(
                        Conflict(
                            self.path,
                            line,
                            spec.function,
                            "return",
                            existing,
                            spec.return_type,
                        )
                    )

        return updated_node.with_changes(
            params=updated_params.with_changes(
                posonly_params=posonly_params,
                params=params,
                kwonly_params=kwonly_params,
                star_arg=star_arg,
                star_kwarg=star_kwarg,
            ),
            returns=returns,
        )


def transform_source(source: str, path: Path = Path("<memory>")) -> SourceResult:
    """Annotate one source string without writing it."""
    specs, skips = _collect_specs(source, path)
    module = cst.parse_module(source)
    transformer = PublicAPIAnnotationTransformer(path, specs)
    transformed = MetadataWrapper(module).visit(transformer)
    return SourceResult(
        source=transformed.code,
        updates=tuple(transformer.updates),
        conflicts=tuple(transformer.conflicts),
        skips=tuple(skips),
        unsafe_existing=tuple(transformer.unsafe_existing),
    )


def update_file(path: Path, check: bool = False) -> FileResult:
    """Annotate one Python file."""
    source = path.read_text(encoding="utf-8")
    result = transform_source(source, path=path)
    changed = result.source != source
    if changed and not check:
        path.write_text(result.source, encoding="utf-8")
    return FileResult(
        path=path,
        updates=result.updates,
        conflicts=result.conflicts,
        skips=result.skips,
        unsafe_existing=result.unsafe_existing,
        changed=changed,
    )


def _parse_args(argv: list[str]) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "paths",
        nargs="*",
        help="Python files or directories to update. Defaults to qmcpy.",
    )
    parser.add_argument(
        "--diff",
        metavar="REF",
        help="Use Python files reported by git diff REF.",
    )
    parser.add_argument(
        "--root",
        default="qmcpy",
        help="Restrict files selected by --diff. Defaults to qmcpy.",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Report annotations without writing files.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Only print the final summary.",
    )
    return parser.parse_args(argv)


def main(argv: list[str]) -> int:
    """Run the command-line interface."""
    args = _parse_args(argv)
    try:
        files = docstrings.python_files(args.paths, args.diff, root=args.root)
    except subprocess.CalledProcessError as exc:
        print(f"git diff failed: {exc}", file=sys.stderr)
        return 2

    if not files:
        print("No Python files to inspect.")
        return 0

    results = []
    had_parse_error = False
    for path in files:
        try:
            results.append(update_file(path, check=args.check))
        except (SyntaxError, cst.ParserSyntaxError) as exc:
            had_parse_error = True
            print(f"{path}: skipped syntax error: {exc}", file=sys.stderr)

    updates = [update for result in results for update in result.updates]
    conflicts = [conflict for result in results for conflict in result.conflicts]
    skips = [skip for result in results for skip in result.skips]
    unsafe_existing = [
        issue for result in results for issue in result.unsafe_existing
    ]
    if not args.quiet:
        action = "would annotate" if args.check else "annotated"
        for update in updates:
            print(
                f"{update.path}:{update.line}: {action} "
                f"{update.function}.{update.slot} as {update.annotation}"
            )
        for conflict in conflicts:
            print(
                f"{conflict.path}:{conflict.line}: conflict "
                f"{conflict.function}.{conflict.slot}: signature "
                f"`{conflict.signature_type}` != docstring "
                f"`{conflict.docstring_type}`"
            )
        for skip in skips:
            print(
                f"{skip.path}:{skip.line}: skipped {skip.function}.{skip.slot} "
                f"`{skip.docstring_type}`: {skip.reason}"
            )
        for issue in unsafe_existing:
            print(
                f"{issue.path}:{issue.line}: unsafe existing annotation "
                f"{issue.function}.{issue.slot} `{issue.annotation}`: "
                f"{issue.reason}"
            )

    changed_files = sum(result.changed for result in results)
    verb = "would change" if args.check else "changed"
    print(
        f"{len(files)} file(s) inspected; {len(updates)} signature update(s); "
        f"{len(conflicts)} conflict(s); {len(skips)} unsafe type(s) skipped; "
        f"{len(unsafe_existing)} unsafe existing annotation(s); "
        f"{changed_files} file(s) {verb}."
    )

    if had_parse_error:
        return 2
    if conflicts or unsafe_existing or (args.check and updates):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
