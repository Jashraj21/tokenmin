"""
Python AST and lexical code compressor.
Strips docstrings, comments, redundant blank lines, and provides skeletonization (interface outline).
"""
from __future__ import annotations

import ast
import re
from typing import Any, List, Optional, Set, Tuple

from tokenmin.models import CompressionLevel
from tokenmin.strategies.base import BaseCompressor


class DocstringStripper(ast.NodeTransformer):
    """AST transformer that removes or condenses docstrings."""

    def __init__(self, keep_first_line: bool = False):
        super().__init__()
        self.keep_first_line = keep_first_line

    def _strip_docstring(self, node: ast.AST) -> None:
        if not hasattr(node, "body") or not node.body:
            return

        first = node.body[0]
        if isinstance(first, ast.Expr) and isinstance(first.value, ast.Constant) and isinstance(first.value.value, str):
            doc = first.value.value
            if self.keep_first_line and doc.strip():
                first_line = doc.strip().splitlines()[0].strip()
                first.value.value = first_line
            else:
                node.body.pop(0)
                if not node.body:
                    node.body.append(ast.Pass())

    def visit_Module(self, node: ast.Module) -> ast.AST:
        self._strip_docstring(node)
        self.generic_visit(node)
        return node

    def visit_ClassDef(self, node: ast.ClassDef) -> ast.AST:
        self._strip_docstring(node)
        self.generic_visit(node)
        return node

    def visit_FunctionDef(self, node: ast.FunctionDef) -> ast.AST:
        self._strip_docstring(node)
        self.generic_visit(node)
        return node

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> ast.AST:
        self._strip_docstring(node)
        self.generic_visit(node)
        return node


class ASTSkeletonizer(ast.NodeTransformer):
    """
    Transforms code into an interface outline:
    keeps classes, methods, signatures, and type annotations,
    replaces function bodies with `...`.
    """

    def visit_FunctionDef(self, node: ast.FunctionDef) -> ast.AST:
        node.body = [ast.Expr(value=ast.Constant(value=...))]
        return node

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> ast.AST:
        node.body = [ast.Expr(value=ast.Constant(value=...))]
        return node


def _strip_fallback_comments(source: str) -> str:
    """Regex fallback to remove comments when AST is unavailable."""
    cleaned_lines = []
    for line in source.splitlines():
        # Only strip # if not inside string
        if "#" in line:
            stripped = re.sub(r"(?<!['\"])#.*$", "", line)
            cleaned_lines.append(stripped)
        else:
            cleaned_lines.append(line)
    return "\n".join(cleaned_lines)


def _collapse_blank_lines(source: str) -> str:
    """Collapse 2 or more blank lines into a single blank line."""
    cleaned = re.sub(r"\n\s*\n\s*\n+", "\n\n", source)
    lines = [line.rstrip() for line in cleaned.splitlines()]
    return "\n".join(lines).strip() + "\n"


class PythonCompressor(BaseCompressor):
    """Compresses Python code via AST transformations, comment removal, and skeletonization."""

    name = "python"
    description = "Optimizes Python source code using AST pruning, comment removal, and signature extraction"

    def run_compression(
        self,
        text: str,
        level: CompressionLevel = CompressionLevel.MODERATE,
        skeleton: bool = False,
        keep_comments: bool = False,
        **kwargs: Any
    ) -> Tuple[str, List[str]]:
        applied_steps: List[str] = []
        code = text

        # Try AST-based transformations
        try:
            tree = ast.parse(code)
            ast_success = True
        except SyntaxError:
            ast_success = False

        if ast_success:
            if skeleton:
                tree = ASTSkeletonizer().visit(tree)
                applied_steps.append("skeletonize_signatures")

            if level == CompressionLevel.CONSERVATIVE:
                tree = DocstringStripper(keep_first_line=True).visit(tree)
                applied_steps.append("condense_docstrings")
            else:
                tree = DocstringStripper(keep_first_line=False).visit(tree)
                applied_steps.append("strip_docstrings")

            ast.fix_missing_locations(tree)
            try:
                # ast.unparse removes all comments and produces clean standard formatting
                code = ast.unparse(tree)
                applied_steps.append("ast_unparse")
                applied_steps.append("strip_comments")
            except Exception:
                code = text
        else:
            # Fallback for code fragments / syntax errors
            if not keep_comments:
                code = _strip_fallback_comments(code)
                applied_steps.append("strip_comments_fallback")

        # Whitespace normalization
        code = _collapse_blank_lines(code)
        applied_steps.append("collapse_blank_lines")

        return code, applied_steps
