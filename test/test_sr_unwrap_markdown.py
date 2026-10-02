from contextlib import redirect_stderr
from io import StringIO
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

from scripts.unwrap_markdown import iter_targets, main, unwrap_markdown_text


class TestUnwrapMarkdown(unittest.TestCase):

    def test_unwraps_list_continuations(self):
        cases = [
            (
                "- unordered first\n  unordered second\n",
                "- unordered first unordered second\n",
            ),
            (
                "- [ ] task first\n  task second\n",
                "- [ ] task first task second\n",
            ),
            (
                "10. ordered first\n    ordered second\n",
                "10. ordered first ordered second\n",
            ),
        ]
        for source, expected in cases:
            with self.subTest(source=source):
                updated = unwrap_markdown_text(source)

                self.assertEqual(updated, expected)
                self.assertEqual(unwrap_markdown_text(updated), updated)

    def test_unwraps_nested_lists_separately(self):
        source = (
            "- parent first\n"
            "  parent second\n"
            "  - child first\n"
            "    child second\n"
            "- sibling first\n"
            "  sibling second\n"
        )

        self.assertEqual(
            unwrap_markdown_text(source),
            (
                "- parent first parent second\n"
                "  - child first child second\n"
                "- sibling first sibling second\n"
            ),
        )

    def test_preserves_list_blocks_and_hard_breaks(self):
        source = (
            "- first paragraph\n"
            "  continuation\n"
            "\n"
            "  second paragraph\n"
            "  continuation\n"
            "\n"
            "- item before code\n"
            "      indented code\n"
            "\n"
            "- explicit hard break  \n"
            "  remains separate\n"
        )

        self.assertEqual(
            unwrap_markdown_text(source),
            (
                "- first paragraph continuation\n"
                "\n"
                "  second paragraph continuation\n"
                "\n"
                "- item before code\n"
                "      indented code\n"
                "\n"
                "- explicit hard break  \n"
                "  remains separate\n"
            ),
        )

    def test_unwraps_paragraphs(self):
        self.assertEqual(
            unwrap_markdown_text("first line\nsecond line\n"),
            "first line second line\n",
        )

    def test_preserves_horizontal_rules(self):
        for rule in ["- - -", "* * *", "_ _ _"]:
            with self.subTest(rule=rule):
                source = f"{rule}\nfollowing paragraph\n"

                self.assertEqual(unwrap_markdown_text(source), source)

    def test_preserves_code_like_a_list(self):
        source = "    - code first\n      code second\n"

        self.assertEqual(unwrap_markdown_text(source), source)

    def test_keeps_ieee_references_separate(self):
        # Without IEEE_REFERENCE_RE, two adjacent [N] entries with no blank
        # line between them would be joined into one run-on paragraph --
        # exactly the case check_ref_style.py's own parser also treats as
        # two separate entries even with no blank line required.
        source = (
            '[1] Author One, "Title One," Venue, 2020.\n'
            '[2] Author Two, "Title Two," Venue, 2021.\n'
        )

        self.assertEqual(unwrap_markdown_text(source), source)

    def test_unwraps_wrapped_ieee_reference(self):
        source = (
            "## References\n\n"
            '[1] Author One, "A Very Long Title Hard-Wrapped Across\n'
            'Several Lines," Venue, 2020.\n\n'
            '[2] Author Two, "Title Two," Venue, 2021.\n'
        )

        self.assertEqual(
            unwrap_markdown_text(source),
            (
                "## References\n\n"
                '[1] Author One, "A Very Long Title Hard-Wrapped Across '
                'Several Lines," Venue, 2020.\n\n'
                '[2] Author Two, "Title Two," Venue, 2021.\n'
            ),
        )

    def test_preserves_code_like_a_reference(self):
        source = "    [1] not a real citation, just code\n"

        self.assertEqual(unwrap_markdown_text(source), source)

    def test_skips_generated_notebook_dirs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            kept = root / "kept.md"
            kept.write_text("kept\n", encoding="utf-8")
            for dirname in (".ipynb_checkpoints", ".virtual_documents"):
                generated = root / dirname
                generated.mkdir()
                (generated / "ignored.md").write_text("ignored\n", encoding="utf-8")

            targets, errors = iter_targets([str(root)])

        self.assertEqual(errors, [])
        self.assertEqual(targets, [kept])

    def test_invalid_notebook_prevents_partial_rewrite(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            markdown = root / "a.md"
            original = "first line\nsecond line\n"
            markdown.write_text(original, encoding="utf-8")
            (root / "z.ipynb").write_text("not json", encoding="utf-8")

            stderr = StringIO()
            with patch.object(sys, "argv", ["unwrap_markdown.py", str(root)]):
                with redirect_stderr(stderr):
                    status = main()

            self.assertEqual(status, 2)
            self.assertIn("invalid notebook", stderr.getvalue())
            self.assertEqual(markdown.read_text(encoding="utf-8"), original)


if __name__ == "__main__":
    unittest.main()
