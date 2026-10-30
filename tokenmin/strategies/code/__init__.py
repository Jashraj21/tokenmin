"""Code compression strategies."""
from tokenmin.strategies.code.generic_code_compressor import GenericCodeCompressor
from tokenmin.strategies.code.python_compressor import PythonCompressor

__all__ = ["PythonCompressor", "GenericCodeCompressor"]
