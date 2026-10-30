"""
Generic code compressor for languages like JavaScript, TypeScript, C/C++, Java, Go, Rust, and SQL.
Safely strips comments, collapse blank lines, and normalizes indentation.
"""
from __future__ import annotations

import re
from typing import Any, List, Tuple

from tokenmin.models import CompressionLevel
from tokenmin.strategies.base import BaseCompressor


def _strip_c_style_comments(code: str) -> str:
    """
    Remove // comments and /* */ comments without touching string literals.
    Handles double quotes, single quotes, and backticks.
    """
    pattern = r"""
        (                               # Group 1: strings
            "(?:\\.|[^"\\])*"          # Double-quoted string
          | '(?:\\.|[^'\\])*'          # Single-quoted string
          | `(?:\\.|[^`\\])*`          # Template literal
        )
      | (/\*.*?\*/)                    # Group 2: Block comment
      | (//[^\r\n]*)                   # Group 3: Line comment
    """
    regex = re.compile(pattern, re.VERBOSE | re.DOTALL)

    def replacer(match: re.Match) -> str:
        # If group 1 matched (a string), return it unchanged
        if match.group(1):
            return match.group(1)
        # Otherwise it was a comment, remove it
        return ""

    return regex.sub(replacer, code)


def _strip_hash_comments(code: str) -> str:
    """Strip # comments for Shell, Ruby, Perl, YAML, etc., avoiding strings."""
    pattern = r"""
        (
            "(?:\\.|[^"\\])*"
          | '(?:\\.|[^'\\])*'
        )
      | (\#[^\r\n]*)
    """
    regex = re.compile(pattern, re.VERBOSE | re.DOTALL)

    def replacer(match: re.Match) -> str:
        if match.group(1):
            return match.group(1)
        return ""

    return regex.sub(replacer, code)


def _strip_sql_comments(code: str) -> str:
    """Strip -- comments and /* */ comments for SQL."""
    pattern = r"""
        (
            '(?:''|[^'])*'
          | "(?:""|[^"])*"
        )
      | (/\*.*?\*/)
      | (--[^\r\n]*)
    """
    regex = re.compile(pattern, re.VERBOSE | re.DOTALL)

    def replacer(match: re.Match) -> str:
        if match.group(1):
            return match.group(1)
        return ""

    return regex.sub(replacer, code)


def _strip_html_comments(code: str) -> str:
    """Strip <!-- --> comments in HTML/XML."""
    return re.sub(r"<!--(?:(?!-->).)*-->", "", code, flags=re.DOTALL)


class GenericCodeCompressor(BaseCompressor):
    """Multi-language code compressor preserving syntax while eliminating comments and blank lines."""

    name = "code"
    description = "Compresses generic code (JS/TS, C/C++, Go, Rust, Java, SQL, HTML) by stripping comments and excess spacing"

    def run_compression(
        self,
        text: str,
        level: CompressionLevel = CompressionLevel.MODERATE,
        lang: str = "generic",
        **kwargs: Any
    ) -> Tuple[str, List[str]]:
        applied_steps: List[str] = []
        code = text
        lang_lower = lang.lower()

        # Comment removal based on language family
        if lang_lower in ("sql",):
            code = _strip_sql_comments(code)
            applied_steps.append("strip_sql_comments")
        elif lang_lower in ("html", "xml", "svg"):
            code = _strip_html_comments(code)
            applied_steps.append("strip_html_comments")
        elif lang_lower in ("sh", "bash", "zsh", "yaml", "yml", "ruby", "rb"):
            code = _strip_hash_comments(code)
            applied_steps.append("strip_hash_comments")
        else:
            # Default C-style comments (JS, TS, C, C++, Java, Rust, Go, C#)
            code = _strip_c_style_comments(code)
            applied_steps.append("strip_c_style_comments")

        # Normalize blank lines
        code = re.sub(r"\n\s*\n\s*\n+", "\n\n", code)
        lines = [line.rstrip() for line in code.splitlines()]
        code = "\n".join(lines).strip() + "\n"
        applied_steps.append("collapse_blank_lines")

        # In aggressive mode: collapse 4-space indentation to 2-space indentation
        if level == CompressionLevel.AGGRESSIVE:
            condensed_lines = []
            for line in code.splitlines():
                leading_spaces = len(line) - len(line.lstrip(" "))
                if leading_spaces > 0 and leading_spaces % 4 == 0:
                    new_indent = "  " * (leading_spaces // 4)
                    condensed_lines.append(new_indent + line.lstrip(" "))
                else:
                    condensed_lines.append(line)
            code = "\n".join(condensed_lines).strip() + "\n"
            applied_steps.append("condense_indentation")

        return code, applied_steps

