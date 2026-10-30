"""Tests for Python AST and code compressor."""
from tokenmin.models import CompressionLevel
from tokenmin.strategies.code import PythonCompressor

SAMPLE_PYTHON = '''"""Module docstring explaining the module."""

import math


def calculate_area(radius: float) -> float:
    """Calculate the area of a circle given radius.
    
    This function uses math.pi and squares the radius.
    """
    # Verify that radius is non-negative
    if radius < 0:
        raise ValueError("radius cannot be negative")
    # Return calculated area
    return math.pi * (radius ** 2)


class ShapeProcessor:
    """A processor for various geometric shapes."""

    def __init__(self, name: str):
        """Initialize shape processor."""
        self.name = name

    def execute(self) -> bool:
        """Execute processing logic."""
        # Print status message
        msg = "Processing with # in string"
        print(msg)
        return True
'''


def test_python_docstring_removal():
    compressor = PythonCompressor()
    result = compressor.compress(SAMPLE_PYTHON, level=CompressionLevel.MODERATE)

    assert result.metrics.saved_tokens > 0
    assert "Calculate the area of a circle" not in result.text
    assert "calculate_area" in result.text
    assert "ShapeProcessor" in result.text
    # Comments should be stripped
    assert "# Verify that radius" not in result.text
    # String containing '#' must be preserved
    assert "Processing with # in string" in result.text


def test_python_skeletonization():
    compressor = PythonCompressor()
    result = compressor.compress(SAMPLE_PYTHON, skeleton=True)

    assert result.metrics.saved_tokens > 0
    assert "def calculate_area(radius: float) -> float:" in result.text
    assert "class ShapeProcessor:" in result.text
    assert "def execute(self) -> bool:" in result.text
    # Function bodies replaced with Ellipsis
    assert "..." in result.text
    assert "math.pi * (radius ** 2)" not in result.text


def test_python_conservative_preserves_first_line():
    compressor = PythonCompressor()
    result = compressor.compress(SAMPLE_PYTHON, level=CompressionLevel.CONSERVATIVE)

    # First line of docstring kept, detailed body stripped
    assert "Calculate the area of a circle" in result.text
    assert "This function uses math.pi and squares the radius" not in result.text


def test_python_syntax_error_fallback():
    broken_code = "def broken(x: \n  return x + # broken comment\n"
    compressor = PythonCompressor()
    result = compressor.compress(broken_code)

    assert "def broken" in result.text

