"""
Markdown compressor.
Minifies markdown tables by removing whitespace padding, tightens list items,
and condenses decorative markdown elements while preserving valid GFM rendering.
"""
from __future__ import annotations

import re
from typing import Any, List, Tuple

from tokenmin.models import CompressionLevel
from tokenmin.strategies.base import BaseCompressor


def _compress_markdown_table(table_text: str) -> str:
    """
    Minify a markdown table by stripping interior space padding from each cell.
    | Header 1       | Header 2       | -> |Header 1|Header 2|
    |----------------|----------------| -> |--|--|
    """
    lines = table_text.strip().splitlines()
    compressed_lines = []
    for line in lines:
        stripped = line.strip()
        if not stripped.startswith("|") and not stripped.endswith("|"):
            compressed_lines.append(stripped)
            continue

        # Split by pipe
        raw_cells = stripped.split("|")
        # Leading and trailing splits may be empty because line starts/ends with |
        if raw_cells and raw_cells[0] == "":
            raw_cells = raw_cells[1:]
        if raw_cells and raw_cells[-1] == "":
            raw_cells = raw_cells[:-1]

        compressed_cells = []
        for cell in raw_cells:
            cell_content = cell.strip()
            # If it's a delimiter row (e.g. ---, :---, ---:, :---:)
            if re.match(r"^:?-+:?$", cell_content):
                # Minify delimiter: preserve alignment colons
                left_colon = ":" if cell_content.startswith(":") else ""
                right_colon = ":" if cell_content.endswith(":") else ""
                min_delim = f"{left_colon}--{right_colon}"
                compressed_cells.append(min_delim)
            else:
                compressed_cells.append(cell_content)

        compressed_lines.append("|" + "|".join(compressed_cells) + "|")

    return "\n".join(compressed_lines)


class MarkdownCompressor(BaseCompressor):
    """Compresses Markdown documents and tables without breaking rendering or semantics."""

    name = "markdown"
    description = "Minifies Markdown tables, tightens list spacing, and strips decorative formatting"

    def run_compression(
        self,
        text: str,
        level: CompressionLevel = CompressionLevel.MODERATE,
        **kwargs: Any
    ) -> Tuple[str, List[str]]:
        applied_steps: List[str] = []
        cleaned = text

        # 1. Minify tables: identify contiguous blocks of table lines
        table_regex = re.compile(r"((?:^[ \t]*\|.*\|[ \t]*\n?)+)", re.MULTILINE)
        if table_regex.search(cleaned):
            def table_sub(match: re.Match) -> str:
                return _compress_markdown_table(match.group(1)) + "\n"

            before = cleaned
            cleaned = table_regex.sub(table_sub, cleaned)
            if cleaned != before:
                applied_steps.append("minify_markdown_tables")

        # 2. Normalize horizontal rules: '----------', '***', '____' -> '---'
        before = cleaned
        cleaned = re.sub(r"^[ \t]*[-*_]{3,}[ \t]*$", "---", cleaned, flags=re.MULTILINE)
        if cleaned != before:
            applied_steps.append("normalize_horizontal_rules")

        # 3. Tighten list spacing: collapse empty lines between list items
        before = cleaned
        cleaned = re.sub(r"(^[ \t]*[-*+\d.]+\s+[^\n]+)\n\s*\n([ \t]*[-*+\d.]+\s+)", r"\1\n\2", cleaned, flags=re.MULTILINE)
        if cleaned != before:
            applied_steps.append("tighten_list_spacing")

        # 4. Collapse multiple blank lines to at most 1 blank line
        cleaned = re.sub(r"\n\s*\n\s*\n+", "\n\n", cleaned)
        applied_steps.append("collapse_blank_lines")

        return cleaned.strip(), applied_steps

