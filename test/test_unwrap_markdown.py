import pytest

from scripts.unwrap_markdown import unwrap_markdown_text


@pytest.mark.parametrize(
    ("source", "expected"),
    [
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
    ],
)
def test_unwraps_list_item_continuations(source, expected):
    updated = unwrap_markdown_text(source)

    assert updated == expected
    assert unwrap_markdown_text(updated) == updated


def test_unwraps_adjacent_and_nested_list_items_separately():
    source = (
        "- parent first\n"
        "  parent second\n"
        "  - child first\n"
        "    child second\n"
        "- sibling first\n"
        "  sibling second\n"
    )

    assert unwrap_markdown_text(source) == (
        "- parent first parent second\n"
        "  - child first child second\n"
        "- sibling first sibling second\n"
    )


def test_preserves_list_item_blocks_and_explicit_hard_breaks():
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

    assert unwrap_markdown_text(source) == (
        "- first paragraph continuation\n"
        "\n"
        "  second paragraph continuation\n"
        "\n"
        "- item before code\n"
        "      indented code\n"
        "\n"
        "- explicit hard break  \n"
        "  remains separate\n"
    )


def test_unwraps_ordinary_paragraphs():
    assert unwrap_markdown_text("first line\nsecond line\n") == "first line second line\n"


@pytest.mark.parametrize("rule", ["- - -", "* * *", "_ _ _"])
def test_preserves_horizontal_rules(rule):
    source = f"{rule}\nfollowing paragraph\n"

    assert unwrap_markdown_text(source) == source


def test_preserves_indented_code_that_looks_like_a_list():
    source = "    - code first\n      code second\n"

    assert unwrap_markdown_text(source) == source
