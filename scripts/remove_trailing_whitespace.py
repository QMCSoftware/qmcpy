#!/usr/bin/env python3
"""Remove trailing whitespace from tracked and untracked source/config files."""

from __future__ import annotations

import argparse
import io
from pathlib import Path
import re
import subprocess
import tokenize


SUPPORTED_NAMES = {"Dockerfile", "Makefile", "makefile"}
SUPPORTED_SUFFIXES = {
    ".bib",
    ".c",
    ".cfg",
    ".cpp",
    ".h",
    ".hpp",
    ".ini",
    ".json",
    ".mk",
    ".ps1",
    ".py",
    ".pyi",
    ".rst",
    ".sh",
    ".tex",
    ".toml",
    ".yaml",
    ".yml",
    ".zsh",
}
TRAILING_WHITESPACE_RE = re.compile(rb"[ \t]+(?=\r?$)", re.MULTILINE)
TRAILING_TEXT_RE = re.compile(r"[ \t]+(?=\r?$)", re.MULTILINE)


def iter_source_files(paths: list[str]) -> list[Path]:
    command = [
        "git",
        "ls-files",
        "-z",
        "--cached",
        "--others",
        "--exclude-standard",
        "--",
        *paths,
    ]
    output = subprocess.run(command, check=True, capture_output=True).stdout
    candidates = (Path(raw) for raw in output.decode().split("\0") if raw)
    return sorted(
        path
        for path in candidates
        if not path.is_symlink()
        and path.is_file()
        and (path.name in SUPPORTED_NAMES or path.suffix.lower() in SUPPORTED_SUFFIXES)
    )


def _python_string_spans(text: str) -> dict[int, list[tuple[int, int | None]]]:
    spans: dict[int, list[tuple[int, int | None]]] = {}
    tokens = tokenize.generate_tokens(io.StringIO(text).readline)
    for token in tokens:
        token_name = tokenize.tok_name.get(token.type, "")
        if token.type != tokenize.STRING and token_name not in {
            "FSTRING_START",
            "FSTRING_MIDDLE",
            "FSTRING_END",
        }:
            continue
        (start_row, start_col), (end_row, end_col) = token.start, token.end
        if start_row == end_row:
            spans.setdefault(start_row, []).append((start_col, end_col))
            continue
        spans.setdefault(start_row, []).append((start_col, None))
        for row in range(start_row + 1, end_row):
            spans.setdefault(row, []).append((0, None))
        spans.setdefault(end_row, []).append((0, end_col))
    return spans


def _strip_python_source(original: bytes) -> bytes:
    encoding, _ = tokenize.detect_encoding(io.BytesIO(original).readline)
    text = original.decode(encoding)
    try:
        string_spans = _python_string_spans(text)
    except (IndentationError, SyntaxError, tokenize.TokenError):
        return original

    updated_lines = []
    for row, line in enumerate(text.splitlines(keepends=True), start=1):
        match = TRAILING_TEXT_RE.search(line)
        if match is None:
            updated_lines.append(line)
            continue
        trailing_start = match.start()
        inside_string = any(
            start <= trailing_start and (end is None or trailing_start < end)
            for start, end in string_spans.get(row, [])
        )
        updated_lines.append(line if inside_string else line[:trailing_start] + line[match.end():])
    return "".join(updated_lines).encode(encoding)


def remove_trailing_whitespace(path: Path, check: bool) -> bool:
    original = path.read_bytes()
    if b"\0" in original:
        return False
    updated = (
        _strip_python_source(original)
        if path.suffix.lower() in {".py", ".pyi"}
        else TRAILING_WHITESPACE_RE.sub(b"", original)
    )
    changed = updated != original
    if changed and not check:
        path.write_bytes(updated)
    return changed


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="fail if any file would change")
    parser.add_argument("paths", nargs="+", help="tracked files or directories to process")
    args = parser.parse_args()

    changed = [
        path for path in iter_source_files(args.paths) if remove_trailing_whitespace(path, args.check)
    ]
    action = "would update" if args.check else "updated"
    print(f"trailing whitespace {action}: {len(changed)} file(s)")
    return int(args.check and bool(changed))


if __name__ == "__main__":
    raise SystemExit(main())
