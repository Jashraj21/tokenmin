"""
Pipeline orchestrator for token minimization.
Routes content to the optimal strategy, auto-detects formats, and applies budget constraints.
"""
from __future__ import annotations

from typing import Any, Dict, Optional, Union

from tokenmin.detector import detect_content_type
from tokenmin.models import CompressionLevel, ContentType, MinifyResult
from tokenmin.strategies.base import BaseCompressor
from tokenmin.strategies.chat import ChatCompressor
from tokenmin.strategies.code import GenericCodeCompressor, PythonCompressor
from tokenmin.strategies.data import JsonCompressor, LogCompressor
from tokenmin.strategies.text import MarkdownCompressor, PromptCompressor


class MinimizerPipeline:
    """Central compression engine managing strategies and routing."""

    def __init__(self):
        self.compressors: Dict[str, BaseCompressor] = {
            "python": PythonCompressor(),
            "code": GenericCodeCompressor(),
            "prompt": PromptCompressor(),
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
        **kwargs: Any
    ) -> MinifyResult:
        """
        Compress text using the most appropriate or specified strategy.

        Args:
            text: Input string to compress.
            content_type: Content type (e.g. auto, prompt, python, code, json, log, chat, markdown).
            level: Compression aggressiveness (conservative, moderate, aggressive).
            target_tokens: Optional hard token cap.
            model: Target LLM model name for accurate token counting.
            **kwargs: Additional strategy-specific arguments (e.g., skeleton=True for python).

        Returns:
            MinifyResult containing compressed text and detailed metrics.
        """
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

        if c_type == ContentType.AUTO:
            detected = detect_content_type(text)
            c_type = detected

        strategy_key = c_type.value.lower()
        compressor = self.compressors.get(strategy_key, self.compressors["prompt"])

        result = compressor.compress(
            text,
            level=level,
            target_tokens=target_tokens,
            model=model,
            **kwargs
        )
        result.content_type = c_type.value
        return result


# Singleton global pipeline instance
_default_pipeline = MinimizerPipeline()


def minimize(
    text: str,
    content_type: Union[ContentType, str] = ContentType.AUTO,
    level: Union[CompressionLevel, str] = CompressionLevel.MODERATE,
    target_tokens: Optional[int] = None,
    model: str = "gpt-4o",
    **kwargs: Any
) -> MinifyResult:
    """Convenience function to minimize text using the default pipeline."""
    return _default_pipeline.minimize(
        text=text,
        content_type=content_type,
        level=level,
        target_tokens=target_tokens,
        model=model,
        **kwargs
    )
