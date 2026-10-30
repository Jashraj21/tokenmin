"""
TokenMin: High-performance token minimizer and prompt optimizer for AI models.
"""
from tokenmin.detector import detect_content_type
from tokenmin.metrics import display_rich_report, format_summary_table
from tokenmin.models import (
    MODEL_INPUT_PRICES_PER_M,
    CompressionLevel,
    ContentType,
    MinifyResult,
    TokenStats,
)
from tokenmin.pipeline import MinimizerPipeline, minimize
from tokenmin.tokenizer import Tokenizer, count_tokens

__version__ = "0.1.0"

__all__ = [
    "minimize",
    "count_tokens",
    "Tokenizer",
    "MinimizerPipeline",
    "CompressionLevel",
    "ContentType",
    "TokenStats",
    "MinifyResult",
    "detect_content_type",
    "format_summary_table",
    "display_rich_report",
    "MODEL_INPUT_PRICES_PER_M",
    "__version__",
]
