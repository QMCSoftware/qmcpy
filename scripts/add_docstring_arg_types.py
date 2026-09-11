#!/usr/bin/env python3
"""Synchronize Google-style docstring types from Python annotations.

This helper is intentionally conservative: it rewrites existing ``Args:``
entries for public functions and methods only when the corresponding argument
has an explicit annotation in the signature. With ``--include-outputs``, it
also updates existing ``Returns:`` and ``Yields:`` descriptions from return
annotations. It does not infer types from implementation code and it does not
invent missing descriptions or sections.
"""
from __future__ import annotations

import argparse
import ast
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


SECTION_HEADER = re.compile(r"^\s*[A-Z][A-Za-z]*(?: [A-Z][A-Za-z]*)*:\s*$")
ARG_ENTRY = re.compile(
    r"^(?P<indent>\s*)"
    r"(?P<name>\*{0,2}[A-Za-z_][A-Za-z0-9_]*)"
    r"\s*"
    r"(?:\((?P<type>[^)]*)\))?"
    r"\s*:\s*"
    r"(?P<description>.*)$"
)
OUTPUT_ENTRY = re.compile(
    r"^(?P<indent>\s*)(?P<type>[^:]+):\s*(?P<description>.*)$"
)
YIELD_CONTAINER_NAMES = {
    "AsyncGenerator",
    "AsyncIterator",
    "Generator",
    "Iterable",
    "Iterator",
}


@dataclass
class Update:
    path: Path
    line: int
    function: str
    argument: str
    annotation: str
    previous_type: str | None
    section: str = "Args"


@dataclass
class Skip:
    path: Path
    line: int
    function: str
    reason: str


@dataclass
class FileResult:
    path: Path
    updates: list[Update]
    skips: list[Skip]
    changed: bool


def doc_node(node: ast.AST) -> ast.Constant | None:
    """Return the string-literal node holding ``node``'s docstring, if any.

    Args:
        node (ast.AST): Node whose docstring literal is wanted.

    Returns:
        ast.Constant | None: The docstring node, or ``None`` when absent.
    """
    body = getattr(node, "body", None)
    if (
        body
        and isinstance(body[0], ast.Expr)
        and isinstance(body[0].value, ast.Constant)
        and isinstance(body[0].value.value, str)
    ):
        return body[0].value
    return None


def iter_public_functions(tree: ast.Module):
    """Yield public module functions and methods from public classes.

    Args:
        tree (ast.Module): Parsed module to walk.

    Yields:
        ast.FunctionDef | ast.AsyncFunctionDef: Each public function or method.
    """
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if not node.name.startswith("_"):
                yield node, node.name
        elif isinstance(node, ast.ClassDef) and not node.name.startswith("_"):
            for sub in node.body:
                if not isinstance(sub, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    continue
                if sub.name == "__init__" or not sub.name.startswith("_"):
                    yield sub, f"{node.name}.{sub.name}"


def _annotation_text(source: str, annotation: ast.AST | None) -> str | None:
    """Return the source spelling of a type annotation."""
    if annotation is None:
        return None
    if isinstance(annotation, ast.Constant) and isinstance(annotation.value, str):
        return annotation.value
    text = ast.get_source_segment(source, annotation)
    if text is not None:
        text = text.strip()
        if (
            len(text) >= 2
            and text[0] in {"'", '"'}
            and text[-1] == text[0]
        ):
            try:
                value = ast.literal_eval(text)
            except (SyntaxError, ValueError):
                return text
            if isinstance(value, str):
                return value
    # ``ast.unparse`` turns a multiline annotation into a safe, single-line
    # representation for a Google-style argument entry.
    return ast.unparse(annotation)


def _argument_annotations(node: ast.FunctionDef | ast.AsyncFunctionDef, source: str):
    """Map argument names to explicit annotation text."""
    annotations = {}
    args = (
        list(node.args.posonlyargs)
        + list(node.args.args)
        + list(node.args.kwonlyargs)
    )
    for arg in args:
        if arg.arg in {"self", "cls"}:
            continue
        annotation = _annotation_text(source, arg.annotation)
        if annotation is not None:
            annotations[arg.arg] = annotation
    if node.args.vararg is not None:
        annotation = _annotation_text(source, node.args.vararg.annotation)
        if annotation is not None:
            annotations[node.args.vararg.arg] = annotation
    if node.args.kwarg is not None:
        annotation = _annotation_text(source, node.args.kwarg.annotation)
        if annotation is not None:
            annotations[node.args.kwarg.arg] = annotation
    return annotations


def line_without_ending(line: str) -> tuple[str, str]:
    """Split a line into content and original line ending.

    Args:
        line (str): Source line, with or without a line ending.

    Returns:
        tuple[str, str]: The content and the line ending that was removed.
    """
    if line.endswith("\r\n"):
        return line[:-2], "\r\n"
    if line.endswith("\n"):
        return line[:-1], "\n"
    return line, ""


def find_section(
    lines: list[str], start: int, end: int, name: str
) -> tuple[int, int] | None:
    """Return the header and end indexes for a Google-style section.

    Args:
        lines (list[str]): Docstring lines to search.
        start (int): First index to consider.
        end (int): Index one past the last to consider.
        name (str): Section header to look for, such as ``"Args"``.

    Returns:
        tuple[int, int] | None: Header and end indexes, or ``None`` when the
        section is absent.
    """
    header_line = None
    header_indent = None
    for i in range(start, end + 1):
        content, _ = line_without_ending(lines[i])
        if content.strip() == f"{name}:":
            header_line = i
            header_indent = len(content) - len(content.lstrip())
            break
    if header_line is None or header_indent is None:
        return None

    section_end = end
    for i in range(header_line + 1, end + 1):
        content, _ = line_without_ending(lines[i])
        stripped = content.strip()
        if not stripped:
            continue
        indent = len(content) - len(content.lstrip())
        if indent <= header_indent and SECTION_HEADER.match(content):
            section_end = i - 1
            break
    return header_line, section_end


def find_args_section(
    lines: list[str], start: int, end: int
) -> tuple[int, int] | None:
    """Return ``(args_line, section_end)`` indexes for a Google Args section.

    Args:
        lines (list[str]): Docstring lines to search.
        start (int): First index to consider.
        end (int): Index one past the last to consider.

    Returns:
        tuple[int, int] | None: Header and end indexes, or ``None`` when there
        is no Args section.
    """
    return find_section(lines, start, end, "Args")


def _yield_annotation_text(source: str, annotation: ast.AST) -> str | None:
    """Extract the yielded item type from a standard iterator annotation."""
    if isinstance(annotation, ast.Constant) and isinstance(annotation.value, str):
        try:
            annotation = ast.parse(annotation.value, mode="eval").body
        except SyntaxError:
            return None
    if not isinstance(annotation, ast.Subscript):
        return None
    value = annotation.value
    if isinstance(value, ast.Name):
        container = value.id
    elif isinstance(value, ast.Attribute):
        container = value.attr
    else:
        return None
    if container not in YIELD_CONTAINER_NAMES:
        return None

    item = annotation.slice
    if container in {"Generator", "AsyncGenerator"} and isinstance(item, ast.Tuple):
        if not item.elts:
            return None
        item = item.elts[0]
    return _annotation_text(source, item)


def looks_like_type(text: str) -> bool:
    """Return whether text is syntactically usable as a type expression.

    Args:
        text (str): Candidate type expression.

    Returns:
        bool: Whether ``text`` parses as a Python expression.
    """
    try:
        ast.parse(text, mode="eval")
    except SyntaxError:
        return False
    return True


def _update_output_section(
    path: Path,
    lines: list[str],
    node: ast.FunctionDef | ast.AsyncFunctionDef,
    function: str,
    section_name: str,
    annotation: str,
    overwrite_existing: bool,
) -> tuple[list[Update], list[Skip]]:
    """Add a signature-derived type to an existing output description."""
    dnode = doc_node(node)
    if dnode is None or dnode.end_lineno is None:
        return [], [Skip(path, node.lineno, function, "missing docstring")]

    section = find_section(
        lines,
        dnode.lineno - 1,
        dnode.end_lineno - 1,
        section_name,
    )
    if section is None:
        return [], [
            Skip(
                path,
                dnode.lineno,
                function,
                f"missing {section_name} section for annotated output",
            )
        ]

    header_content, _ = line_without_ending(lines[section[0]])
    header_indent = len(header_content) - len(header_content.lstrip())
    for i in range(section[0] + 1, section[1] + 1):
        content, ending = line_without_ending(lines[i])
        if not content.strip():
            continue
        indent = len(content) - len(content.lstrip())
        if indent <= header_indent:
            continue

        match = OUTPUT_ENTRY.match(content)
        previous_type = None
        description = content.strip()
        entry_indent = content[:indent]
        if match is not None and looks_like_type(match.group("type").strip()):
            previous_type = match.group("type").strip()
            if not overwrite_existing:
                return [], []
            description = match.group("description").lstrip()
            entry_indent = match.group("indent")

        suffix = f" {description}" if description else ""
        replacement = f"{entry_indent}{annotation}:{suffix}{ending}"
        if replacement == lines[i]:
            return [], []
        lines[i] = replacement
        slot = "yield" if section_name == "Yields" else "return"
        return [
            Update(
                path=path,
                line=i + 1,
                function=function,
                argument=slot,
                annotation=annotation,
                previous_type=previous_type,
                section=section_name,
            )
        ], []

    return [], [
        Skip(path, node.lineno, function, f"empty {section_name} section")
    ]


def _update_args_section(
    path: Path,
    lines: list[str],
    node: ast.FunctionDef | ast.AsyncFunctionDef,
    function: str,
    annotations: dict[str, str],
    overwrite_existing: bool,
) -> tuple[list[Update], list[Skip]]:
    """Add annotation text to matching ``Args:`` entries."""
    dnode = doc_node(node)
    if dnode is None or dnode.end_lineno is None:
        return [], [Skip(path, node.lineno, function, "missing docstring")]

    section = find_args_section(lines, dnode.lineno - 1, dnode.end_lineno - 1)
    if section is None:
        return [], [Skip(path, dnode.lineno, function, "missing Args section")]

    updates = []
    seen = set()
    _, section_end = section
    for i in range(section[0] + 1, section_end + 1):
        content, ending = line_without_ending(lines[i])
        match = ARG_ENTRY.match(content)
        if match is None:
            continue
        display_name = match.group("name")
        argument = display_name.lstrip("*")
        if argument not in annotations:
            continue
        seen.add(argument)
        previous_type = match.group("type")
        if previous_type is not None and not overwrite_existing:
            continue
        annotation = annotations[argument]
        description = match.group("description").lstrip()
        suffix = f" {description}" if description else ""
        replacement = (
            f"{match.group('indent')}{display_name} ({annotation}):{suffix}{ending}"
        )
        if replacement == lines[i]:
            continue
        lines[i] = replacement
        updates.append(
            Update(
                path=path,
                line=i + 1,
                function=function,
                argument=argument,
                annotation=annotation,
                previous_type=previous_type,
            )
        )

    skips = [
        Skip(
            path,
            node.lineno,
            function,
            f"missing Args entry for annotated argument `{name}`",
        )
        for name in sorted(set(annotations) - seen)
    ]
    return updates, skips


def update_file(
    path: Path,
    check: bool = False,
    overwrite_existing: bool = False,
    include_outputs: bool = False,
) -> FileResult:
    """Update Google-style types in one Python file.

    Args:
        path (Path): Python file to update.
        check (bool): Report what would change without writing.
        overwrite_existing (bool): Replace types already present rather than only
            filling in missing ones.
        include_outputs (bool): Also update the Returns section.

    Returns:
        FileResult: Counts of updates made and entries skipped.
    """
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(path))
    lines = source.splitlines(keepends=True)
    updates = []
    skips = []

    for node, function in iter_public_functions(tree):
        annotations = _argument_annotations(node, source)
        if annotations:
            node_updates, node_skips = _update_args_section(
                path=path,
                lines=lines,
                node=node,
                function=function,
                annotations=annotations,
                overwrite_existing=overwrite_existing,
            )
            updates.extend(node_updates)
            skips.extend(node_skips)

        if not include_outputs or node.name == "__init__" or node.returns is None:
            continue
        return_annotation = _annotation_text(source, node.returns)
        if return_annotation in {None, "None", "NoneType"}:
            continue

        dnode = doc_node(node)
        if dnode is None or dnode.end_lineno is None:
            skips.append(Skip(path, node.lineno, function, "missing docstring"))
            continue
        doc_start = dnode.lineno - 1
        doc_end = dnode.end_lineno - 1
        yields_section = find_section(lines, doc_start, doc_end, "Yields")
        section_name = "Yields" if yields_section is not None else "Returns"
        output_annotation = return_annotation
        if section_name == "Yields":
            output_annotation = _yield_annotation_text(source, node.returns)
            if output_annotation is None:
                skips.append(
                    Skip(
                        path,
                        node.lineno,
                        function,
                        "cannot derive yielded item type from return annotation",
                    )
                )
                continue
        node_updates, node_skips = _update_output_section(
            path=path,
            lines=lines,
            node=node,
            function=function,
            section_name=section_name,
            annotation=output_annotation,
            overwrite_existing=overwrite_existing,
        )
        updates.extend(node_updates)
        skips.extend(node_skips)

    changed = bool(updates)
    if changed and not check:
        path.write_text("".join(lines), encoding="utf-8")
    return FileResult(path=path, updates=updates, skips=skips, changed=changed)


def _changed_files(ref: str) -> list[Path]:
    """Return Python files changed relative to ``ref`` using ``git diff``."""
    result = subprocess.run(
        ["git", "diff", "--name-only", "--diff-filter=ACMR", ref, "--", "*.py"],
        capture_output=True,
        text=True,
        check=True,
    )
    return [Path(name) for name in result.stdout.splitlines()]


def _is_under(path: Path, root: Path) -> bool:
    """Return whether a relative or absolute path is under root."""
    try:
        path.resolve().relative_to(root.resolve())
    except ValueError:
        return False
    return True


def python_files(
    paths: list[str], diff_ref: str | None, root: str | None = None
) -> list[Path]:
    """Collect Python files from paths, or from ``git diff`` when requested.

    Args:
        paths (list[str]): Files or directories to collect from.
        diff_ref (str | None): Git ref to diff against instead of using ``paths``.
        root (str | None): Repository root for the diff; defaults to the cwd.

    Returns:
        list[Path]: Python files to process, in sorted order.
    """
    if diff_ref is not None:
        candidates = _changed_files(diff_ref)
    else:
        candidates = [Path(p) for p in (paths or ["qmcpy"])]

    if root is not None and diff_ref is not None:
        root_path = Path(root)
        candidates = [path for path in candidates if _is_under(path, root_path)]

    files = []
    for path in candidates:
        if path.is_dir():
            files.extend(sorted(path.rglob("*.py")))
        elif path.suffix == ".py" and path.exists():
            files.append(path)
    return sorted(dict.fromkeys(files))


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
        help="Update Python files reported by `git diff --name-only REF -- '*.py'`.",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Report files that would change without writing them.",
    )
    parser.add_argument(
        "--overwrite-existing",
        action="store_true",
        help="Replace existing Google Args types with signature annotations.",
    )
    parser.add_argument(
        "--include-outputs",
        action="store_true",
        help="Also update existing Returns and Yields descriptions.",
    )
    parser.add_argument(
        "--root",
        help="Restrict files selected by --diff to this directory.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Only print the final summary.",
    )
    return parser.parse_args(argv)


def main(argv: list[str]) -> int:
    """Run the command-line interface.

    Args:
        argv (list[str]): Command-line arguments, excluding the program name.

    Returns:
        int: Process exit status; ``0`` on success.
    """
    args = _parse_args(argv)
    try:
        files = python_files(args.paths, args.diff, root=args.root)
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
            result = update_file(
                path,
                check=args.check,
                overwrite_existing=args.overwrite_existing,
                include_outputs=args.include_outputs,
            )
        except SyntaxError as exc:
            had_parse_error = True
            print(f"{path}: skipped syntax error: {exc}", file=sys.stderr)
            continue
        results.append(result)

    updates = [update for result in results for update in result.updates]
    skips = [skip for result in results for skip in result.skips]
    if not args.quiet and (updates or skips):
        print()
        for update in updates:
            action = "would update" if args.check else "updated"
            old = (
                ""
                if update.previous_type is None
                else f" replacing `{update.previous_type}`"
            )
            print(
                f"  - {update.path}:{update.line}: {action} "
                f"{update.function}.{update.argument} ({update.annotation}){old}"
            )
        for skip in skips:
            print(f"  - {skip.path}:{skip.line}: skipped {skip.function}: {skip.reason}")

    args_updates = [update for update in updates if update.section == "Args"]
    output_updates = [update for update in updates if update.section != "Args"]
    changed_files = sum(1 for result in results if result.changed)
    verb = "would change" if args.check else "changed"
    print(
        f"  - {len(files)} file(s) inspected; {len(args_updates)} Args type update(s); "
        f"{len(output_updates)} output type update(s); "
        f"{changed_files} file(s) {verb}."
    )

    if changed_files == 0:
        print(f"clean  (0 of {len(files)} files)")
    elif args.check:
        print(f"ERROR: {changed_files} would change  ({changed_files} of {len(files)} files)")
    else:
        print(f"{changed_files} changed  ({changed_files} of {len(files)} files)")

    if args.check and updates:
        return 1
    return 2 if had_parse_error else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
