"""
TokenMin: High-performance token minimizer and prompt optimizer for latest AI models.
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
from tokenmin.models_registry import LATEST_MODELS, ModelInfo, list_supported_models, resolve_model
from tokenmin.pipeline import MinimizerPipeline, minimize
from tokenmin.tokenizer import Tokenizer, count_tokens

__version__ = "0.2.0"

__all__ = [
    "minimize",
    "count_tokens",
    "Tokenizer",
    "MinimizerPipeline",
    "CompressionLevel",
    "ContentType",
    "TokenStats",
    "MinifyResult",
    "ModelInfo",
    "LATEST_MODELS",
    "resolve_model",
    "list_supported_models",
    "detect_content_type",
    "format_summary_table",
    "display_rich_report",
    "MODEL_INPUT_PRICES_PER_M",
    "__version__",
]
