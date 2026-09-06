#!/usr/bin/env python3
"""Convert Python assertions to explicit exception raises.

The codemod preserves formatting and comments with LibCST. By default it
converts ``assert condition, message`` to an explicit ``AssertionError`` so
the validation is not removed by ``python -O``. A developer may select a
different exception that is already in scope, but the tool deliberately does
not guess domain-specific exception classes.
"""
from __future__ import annotations

import argparse
import re
import subprocess
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import libcst as cst
from libcst.metadata import MetadataWrapper, PositionProvider


EXCEPTION_NAME = re.compile(
    r"^[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)*$"
)


@dataclass(frozen=True)
class SourceResult:
    """Result of transforming one source string."""

    source: str
    converted_lines: tuple[int, ...]
    skipped_lines: tuple[int, ...]


@dataclass(frozen=True)
class FileResult:
    """Result of inspecting one Python file."""

    path: Path
    converted_lines: tuple[int, ...]
    skipped_lines: tuple[int, ...]
    changed: bool


def _parenthesize(expression: cst.BaseExpression) -> cst.BaseExpression:
    """Parenthesize an expression unless it is already parenthesized."""
    if expression.lpar:
        return expression
    return expression.with_changes(
        lpar=(cst.LeftParen(),),
        rpar=(cst.RightParen(),),
    )


def _exception_call(
    exception: cst.BaseExpression,
    message: cst.BaseExpression,
) -> cst.Call:
    """Build an exception call while reusing message-parenthesis whitespace."""
    if (
        not message.lpar
        or not message.rpar
        or isinstance(message, (cst.Tuple, cst.Yield))
    ):
        return cst.Call(func=exception, args=[cst.Arg(message)])

    opening = message.lpar[0]
    closing = message.rpar[-1]
    unwrapped_message = message.with_changes(
        lpar=message.lpar[1:],
        rpar=message.rpar[:-1],
    )
    return cst.Call(
        func=exception,
        args=[
            cst.Arg(
                unwrapped_message,
                whitespace_after_arg=closing.whitespace_before,
            )
        ],
        whitespace_before_args=opening.whitespace_after,
    )


class ConvertAssertTransformer(cst.CSTTransformer):
    """Rewrite standalone assertion statements as explicit conditional raises."""

    METADATA_DEPENDENCIES = (PositionProvider,)

    def __init__(self, exception: str):
        self.exception = cst.parse_expression(exception)
        self.seen_lines = []
        self.converted_lines = []

    def visit_Assert(self, node: cst.Assert) -> None:
        """Record every assertion, including forms that cannot be rewritten."""
        position = self.get_metadata(PositionProvider, node)
        self.seen_lines.append(position.start.line)

    def leave_SimpleStatementLine(
        self,
        original_node: cst.SimpleStatementLine,
        updated_node: cst.SimpleStatementLine,
    ) -> cst.BaseStatement:
        """Rewrite an assert when it is the line's only small statement."""
        if len(updated_node.body) != 1:
            return updated_node
        assertion = updated_node.body[0]
        if not isinstance(assertion, cst.Assert):
            return updated_node

        condition = cst.UnaryOperation(
            operator=cst.Not(whitespace_after=cst.SimpleWhitespace(" ")),
            expression=_parenthesize(assertion.test),
        )
        exception = self.exception.deep_clone()
        if assertion.msg is None:
            raised_exception = exception
        else:
            raised_exception = _exception_call(exception, assertion.msg)

        position = self.get_metadata(PositionProvider, original_node)
        self.converted_lines.append(position.start.line)
        return cst.If(
            test=condition,
            body=cst.IndentedBlock(
                header=updated_node.trailing_whitespace,
                body=[
                    cst.SimpleStatementLine(
                        body=[cst.Raise(exc=raised_exception)]
                    )
                ],
            ),
            leading_lines=updated_node.leading_lines,
        )


def transform_source(source: str, exception: str = "AssertionError") -> SourceResult:
    """Transform standalone assertions in a Python source string."""
    _validate_exception(exception)
    module = cst.parse_module(source)
    transformer = ConvertAssertTransformer(exception)
    updated = MetadataWrapper(module).visit(transformer)

    skipped = Counter(transformer.seen_lines)
    skipped.subtract(transformer.converted_lines)
    skipped_lines = tuple(
        line
        for line, count in sorted(skipped.items())
        for _ in range(max(count, 0))
    )
    return SourceResult(
        source=updated.code,
        converted_lines=tuple(transformer.converted_lines),
        skipped_lines=skipped_lines,
    )


def convert_file(
    path: Path,
    exception: str = "AssertionError",
    check: bool = False,
) -> FileResult:
    """Convert assertions in one Python file."""
    source = path.read_text(encoding="utf-8")
    result = transform_source(source, exception=exception)
    changed = result.source != source
    if changed and not check:
        path.write_text(result.source, encoding="utf-8")
    return FileResult(
        path=path,
        converted_lines=result.converted_lines,
        skipped_lines=result.skipped_lines,
        changed=changed,
    )


def _validate_exception(exception: str) -> None:
    """Require a simple or dotted exception name, not arbitrary code."""
    if not EXCEPTION_NAME.fullmatch(exception):
        raise ValueError(
            "exception must be a name already in scope, such as "
            "AssertionError, ValueError, or qmcpy.util.ParameterError"
        )


def _changed_files(ref: str) -> list[Path]:
    """Return changed production Python files relative to ``ref``."""
    result = subprocess.run(
        [
            "git",
            "diff",
            "--name-only",
            "--diff-filter=ACMR",
            ref,
            "--",
            "qmcpy/*.py",
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    return [Path(name) for name in result.stdout.splitlines()]


def _python_files(paths: list[str], diff_ref: str | None) -> list[Path]:
    """Collect Python files from paths or a production-code diff."""
    if diff_ref is not None:
        candidates = _changed_files(diff_ref)
    else:
        candidates = [Path(path) for path in (paths or ["qmcpy"])]

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
        help="Use changed qmcpy/*.py files reported by git diff REF.",
    )
    parser.add_argument(
        "--exception",
        default="AssertionError",
        help=(
            "Exception name already in scope for every selected file. "
            "Defaults to AssertionError."
        ),
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Report convertible assertions without writing files.",
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
        _validate_exception(args.exception)
        files = _python_files(args.paths, args.diff)
    except (ValueError, subprocess.CalledProcessError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2

    if not files:
        print("No Python files to inspect.")
        return 0

    results = []
    had_parse_error = False
    for path in files:
        try:
            result = convert_file(
                path,
                exception=args.exception,
                check=args.check,
            )
        except (cst.ParserSyntaxError, UnicodeError) as error:
            had_parse_error = True
            print(f"{path}: skipped parse error: {error}", file=sys.stderr)
            continue
        results.append(result)

    if not args.quiet:
        action = "would convert" if args.check else "converted"
        for result in results:
            for line in result.converted_lines:
                print(
                    f"{result.path}:{line}: {action} assert to "
                    f"explicit {args.exception}"
                )
            for line in result.skipped_lines:
                print(
                    f"{result.path}:{line}: skipped assert in a compound "
                    "one-line statement"
                )

    converted = sum(len(result.converted_lines) for result in results)
    skipped = sum(len(result.skipped_lines) for result in results)
    changed_files = sum(result.changed for result in results)
    verb = "would change" if args.check else "changed"
    print(
        f"{len(files)} file(s) inspected; {converted} assert(s) converted; "
        f"{skipped} assert(s) skipped; {changed_files} file(s) {verb}."
    )

    if args.check and converted:
        return 1
    return 2 if had_parse_error else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
