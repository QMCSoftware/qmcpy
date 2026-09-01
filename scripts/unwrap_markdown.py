#!/usr/bin/env python3
"""Unwrap hard-wrapped Markdown prose in .md files and notebook markdown cells."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import sys

SUPPORTED_SUFFIXES = {".md", ".ipynb"}
FENCE_RE = re.compile(r"^\s*([`~]{3,})")
LIST_RE = re.compile(r"^\s*(?:[-+*]|\d+[.)])\s+")
REFERENCE_DEF_RE = re.compile(r"^\s*\[[^\]]+\]:\s+\S")
SETEXT_RE = re.compile(r"^\s{0,3}(?:=+|-+)\s*$")
HR_RE = re.compile(r"^\s{0,3}(?:[-*_]\s*){3,}$")
LATEX_ENV_BEGIN_RE = re.compile(r"^\s*\\begin\{([A-Za-z*]+)\}")
LATEX_ENV_END_RE = re.compile(r"^\s*\\end\{([A-Za-z*]+)\}")
LATEX_HINT_RE = re.compile(
    r"(?<!\\)\$\$|(?<!\\)\$(?=\S)|(?<=\S)(?<!\\)\$|\\\(|\\\)|\\\[|\\\]|\\begin\{[A-Za-z*]+\}|\\end\{[A-Za-z*]+\}|\\[A-Za-z]+",
)
HTML_TAG_RE = re.compile(r"^</?[A-Za-z]")


def iter_targets(paths: list[str]) -> tuple[list[Path], list[str]]:
    files: list[Path] = []
    errors: list[str] = []
    for raw_path in paths:
        path = Path(raw_path)
        if not path.exists():
            errors.append(f"path not found: {raw_path}")
            continue
        if path.is_file():
            if path.suffix.lower() not in SUPPORTED_SUFFIXES:
                errors.append(f"unsupported file type: {raw_path}")
                continue
            files.append(path)
            continue
        for child in sorted(path.rglob("*")):
            if child.is_file() and child.suffix.lower() in SUPPORTED_SUFFIXES:
                files.append(child)
    return files, errors


def _closing_fence(line: str, marker: str) -> bool:
    stripped = line.lstrip()
    return stripped.startswith(marker[0] * len(marker))


def _is_structural_line(line: str) -> bool:
    stripped = line.strip()
    if not stripped:
        return True
    if line.startswith("    ") or line.startswith("\t"):
        return True
    if stripped.startswith(("#", ">", "|", "<!--", ":::", "!!!", "???", "[![")):
        return True
    if HTML_TAG_RE.match(stripped):
        return True
    if REFERENCE_DEF_RE.match(line):
        return True
    if LIST_RE.match(line):
        return True
    if SETEXT_RE.match(line) or HR_RE.match(line):
        return True
    return False


def _unwrap_paragraph(lines: list[str]) -> str:
    if len(lines) <= 1:
        return lines[0]
    return " ".join([lines[0].rstrip(), *(line.strip() for line in lines[1:])])


def _relative_to_list_item(line: str, paragraph: list[str]) -> str:
    """Return a continuation line relative to its active list-item marker."""
    if not paragraph:
        return line
    list_match = LIST_RE.match(paragraph[0])
    if list_match is None:
        return line
    content_indent = list_match.end()
    leading_whitespace = len(line) - len(line.lstrip())
    if leading_whitespace < content_indent:
        return line
    return line[content_indent:]


def _paragraph_has_latex(lines: list[str]) -> bool:
    return bool(LATEX_HINT_RE.search("\n".join(lines)))


def unwrap_markdown_text(text: str, *, preserve_latex: bool = False) -> str:
    if not text:
        return text

    newline = "\r\n" if "\r\n" in text else "\n"
    had_final_newline = text.endswith(("\n", "\r"))
    lines = text.splitlines()
    out: list[str] = []
    paragraph: list[str] = []
    fence_marker: str | None = None
    math_fence_end: str | None = None
    latex_env_name: str | None = None
    in_comment = False
    in_html_block = False
    in_front_matter = bool(lines and lines[0].strip() == "---")

    def flush_paragraph() -> None:
        nonlocal paragraph
        if paragraph:
            if preserve_latex and _paragraph_has_latex(paragraph):
                out.extend(paragraph)
            else:
                out.append(_unwrap_paragraph(paragraph))
            paragraph = []

    for index, line in enumerate(lines):
        stripped = line.strip()

        if in_front_matter:
            out.append(line)
            if index > 0 and stripped in {"---", "..."}:
                in_front_matter = False
            continue

        if fence_marker is not None:
            out.append(line)
            if _closing_fence(line, fence_marker):
                fence_marker = None
            continue

        if math_fence_end is not None:
            out.append(line)
            if stripped == math_fence_end:
                math_fence_end = None
            continue

        if latex_env_name is not None:
            out.append(line)
            end_match = LATEX_ENV_END_RE.match(line)
            if end_match and end_match.group(1) == latex_env_name:
                latex_env_name = None
            continue

        if in_comment:
            out.append(line)
            if "-->" in line:
                in_comment = False
            continue

        if in_html_block:
            out.append(line)
            if not stripped:
                in_html_block = False
            continue

        fence_match = FENCE_RE.match(line)
        if fence_match:
            flush_paragraph()
            fence_marker = fence_match.group(1)
            out.append(line)
            continue

        if stripped in {"$$", "\\[", "\\("}:
            flush_paragraph()
            if stripped == "$$":
                math_fence_end = "$$"
            elif stripped == "\\[":
                math_fence_end = "\\]"
            else:
                math_fence_end = "\\)"
            out.append(line)
            continue
        if stripped in {"\\]", "\\)"}:
            flush_paragraph()
            out.append(line)
            continue

        if preserve_latex:
            begin_match = LATEX_ENV_BEGIN_RE.match(line)
            if begin_match:
                flush_paragraph()
                latex_env_name = begin_match.group(1)
                out.append(line)
                end_match = LATEX_ENV_END_RE.match(line)
                if end_match and end_match.group(1) == latex_env_name:
                    latex_env_name = None
                continue

        if not stripped:
            flush_paragraph()
            out.append(line)
            continue

        list_match = LIST_RE.match(line)
        active_list_item = bool(paragraph and LIST_RE.match(paragraph[0]))
        indented_as_code = line.startswith("    ") or line.startswith("\t")
        if list_match and not HR_RE.match(line) and (
            active_list_item or not indented_as_code
        ):
            flush_paragraph()
            paragraph.append(line)
            continue

        structural_line = _relative_to_list_item(line, paragraph)
        if _is_structural_line(structural_line):
            flush_paragraph()
            out.append(line)
            if stripped.startswith("<!--") and "-->" not in stripped:
                in_comment = True
            elif HTML_TAG_RE.match(stripped):
                in_html_block = True
            continue

        if paragraph and (paragraph[-1].endswith("  ") or paragraph[-1].endswith("\\")):
            flush_paragraph()
        paragraph.append(line)

    flush_paragraph()
    result = newline.join(out)
    if had_final_newline:
        result += newline
    return result


def _split_notebook_source(text: str) -> list[str]:
    if not text:
        return []
    return text.splitlines(keepends=True)


def process_markdown_file(path: Path, check: bool) -> bool:
    original = path.read_text(encoding="utf-8")
    updated = unwrap_markdown_text(original, preserve_latex=True)
    changed = updated != original
    if changed and not check:
        path.write_text(updated, encoding="utf-8")
    return changed


def process_notebook(path: Path, check: bool) -> tuple[bool, int]:
    with path.open(encoding="utf-8") as handle:
        notebook = json.load(handle)

    changed = False
    changed_cells = 0
    for cell in notebook.get("cells", []):
        if cell.get("cell_type") != "markdown":
            continue
        source = cell.get("source", [])
        source_text = source if isinstance(source, str) else "".join(source)
        updated = unwrap_markdown_text(source_text, preserve_latex=True)
        if updated == source_text:
            continue
        changed = True
        changed_cells += 1
        cell["source"] = updated if isinstance(source, str) else _split_notebook_source(updated)

    if changed and not check:
        with path.open("w", encoding="utf-8") as handle:
            json.dump(notebook, handle, ensure_ascii=False, indent=1)
            handle.write("\n")
    return changed, changed_cells


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="fail if any file would change")
    parser.add_argument("paths", nargs="+", help="Markdown file, notebook, or directory to process")
    args = parser.parse_args()

    targets, errors = iter_targets(args.paths)
    if errors:
        for message in errors:
            print(f"error: {message}", file=sys.stderr)
        return 2
    if not targets:
        print("error: no .md or .ipynb files found", file=sys.stderr)
        return 2

    changed_files = 0
    changed_cells = 0
    for path in targets:
        suffix = path.suffix.lower()
        if suffix == ".md":
            changed = process_markdown_file(path, args.check)
            changed_files += int(changed)
        elif suffix == ".ipynb":
            changed, cell_count = process_notebook(path, args.check)
            changed_files += int(changed)
            changed_cells += cell_count

    mode = "would update" if args.check else "updated"
    print(
        f"markdown unwrap {mode}: {changed_files} file(s), {changed_cells} markdown cell(s)",
    )
    return 1 if args.check and changed_files else 0


if __name__ == "__main__":
    raise SystemExit(main())
