"""Tests for Markdown compressor."""
from tokenmin.strategies.text import MarkdownCompressor


def test_markdown_table_compression():
    md = """
    # Results Summary

    | Column 1             | Column 2                          | Status    |
    |----------------------|-----------------------------------|-----------|
    | Alpha Item           | This is a long description cell   | Passed    |
    | Beta Item            | Another test result cell          | Failed    |

    End of summary.
    """
    compressor = MarkdownCompressor()
    result = compressor.compress(md)

    assert result.metrics.saved_tokens > 0
    # Inside table, padded spaces should be minified
    assert "|Column 1|Column 2|Status|" in result.text
    assert "|Alpha Item|This is a long description cell|Passed|" in result.text


def test_horizontal_rules():
    md = "Section A\n\n-------------------------------\n\nSection B"
    compressor = MarkdownCompressor()
    result = compressor.compress(md)

    assert "---" in result.text
    assert "-------------------------------" not in result.text

