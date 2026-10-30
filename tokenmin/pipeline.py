"""
Pipeline orchestrator for token minimization.
Routes content to the optimal strategy, auto-detects formats, and applies budget constraints.
"""
from __future__ import annotations

import re
from typing import Any, Dict, Optional, Union

from tokenmin.detector import detect_content_type
from tokenmin.models import CompressionLevel, ContentType, MinifyResult
from tokenmin.models_registry import resolve_model
from tokenmin.strategies.base import BaseCompressor
from tokenmin.strategies.cache import CacheOptimizer
from tokenmin.strategies.chat import ChatCompressor
from tokenmin.strategies.code import GenericCodeCompressor, PythonCompressor
from tokenmin.strategies.data import (
    DeltaCompressor,
    JsonCompressor,
    LogCompressor,
)
from tokenmin.strategies.text import (
    MarkdownCompressor,
    PromptCompressor,
    ReasoningCompressor,
)


class MinimizerPipeline:
    """Central compression engine managing strategies and routing."""

    def __init__(self):
        self.compressors: Dict[str, BaseCompressor] = {
            "python": PythonCompressor(),
            "code": GenericCodeCompressor(),
            "prompt": PromptCompressor(),
            "reasoning": ReasoningCompressor(),
            "cache": CacheOptimizer(),
            "delta": DeltaCompressor(),
            "text": PromptCompressor(),
            "markdown": MarkdownCompressor(),
            "json": JsonCompressor(),
            "log": LogCompressor(),
            "chat": ChatCompressor(),
        }

    def register_compressor(self, name: str, compressor: BaseCompressor) -> None:
        """Register a custom compressor strategy."""
        self.compressors[name.lower()] = compressor

    def minimize(
        self,
        text: str,
        content_type: Union[ContentType, str] = ContentType.AUTO,
        level: Union[CompressionLevel, str] = CompressionLevel.MODERATE,
        target_tokens: Optional[int] = None,
        model: str = "gpt-4o",
        reasoning_mode: bool = False,
        cache_friendly: bool = False,
        **kwargs: Any
    ) -> MinifyResult:
        """
        Compress text using the most appropriate or specified strategy.
        """
        model_info = resolve_model(model)

        # Normalize level
        if isinstance(level, str):
            level = CompressionLevel(level.lower())

        # Resolve content type
        if isinstance(content_type, str):
            try:
                c_type = ContentType(content_type.lower())
            except ValueError:
                c_type = ContentType.AUTO
        else:
            c_type = content_type

        # 1. If AUTO, check for streaming delta frames (Astra/Gemini 4) or detect content format
        if c_type == ContentType.AUTO:
            if re.search(r"^\[(?:Frame\s+\d+|\d{2}:\d{2})\]", text.strip(), re.MULTILINE):
                strategy_key = "delta"
            else:
                c_type = detect_content_type(text)
                strategy_key = c_type.value.lower()
        else:
            strategy_key = c_type.value.lower()

        # 2. Route specialized modes
        if cache_friendly or c_type == ContentType.CACHE:
            strategy_key = "cache"
        elif reasoning_mode or c_type == ContentType.REASONING or (model_info.is_reasoning_model and strategy_key in ("prompt", "text")):
            strategy_key = "reasoning"

        compressor = self.compressors.get(strategy_key, self.compressors["prompt"])

        result = compressor.compress(
            text,
            level=level,
            target_tokens=target_tokens,
            model=model_info.name,
            **kwargs
        )
        result.content_type = strategy_key
        return result


# Singleton global pipeline instance
_default_pipeline = MinimizerPipeline()


def minimize(
    text: str,
    content_type: Union[ContentType, str] = ContentType.AUTO,
    level: Union[CompressionLevel, str] = CompressionLevel.MODERATE,
    target_tokens: Optional[int] = None,
    model: str = "gpt-4o",
    reasoning_mode: bool = False,
    cache_friendly: bool = False,
    **kwargs: Any
) -> MinifyResult:
    """Convenience function to minimize text using the default pipeline."""
    return _default_pipeline.minimize(
        text=text,
        content_type=content_type,
        level=level,
        target_tokens=target_tokens,
        model=model,
        reasoning_mode=reasoning_mode,
        cache_friendly=cache_friendly,
        **kwargs
    )
