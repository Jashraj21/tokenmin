"""
Base interface and abstractions for all token compression strategies.
"""
from __future__ import annotations

import time
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple

from tokenmin.models import CompressionLevel, MinifyResult, TokenStats
from tokenmin.tokenizer import Tokenizer


class BaseCompressor(ABC):
    """Abstract base class for all token minimizers."""

    name: str = "base"
    description: str = "Base compressor"

    @abstractmethod
    def run_compression(
        self,
        text: str,
        level: CompressionLevel = CompressionLevel.MODERATE,
        **kwargs: Any
    ) -> Tuple[str, List[str]]:
        """
        Execute core compression transformations.

        Returns:
            Tuple of (compressed_text, list_of_applied_step_names)
        """
        pass

    def compress(
        self,
        text: str,
        level: CompressionLevel = CompressionLevel.MODERATE,
        target_tokens: Optional[int] = None,
        model: str = "gpt-4o",
        **kwargs: Any
    ) -> MinifyResult:
        """
        Compress text and return MinifyResult with complete metrics.
        """
        if not text:
            stats = TokenStats(
                original_tokens=0,
                minimized_tokens=0,
                original_chars=0,
                minimized_chars=0,
                elapsed_ms=0.0,
                model=model,
            )
            return MinifyResult(
                text="",
                original="",
                content_type=self.name,
                strategy_used=self.name,
                metrics=stats,
                applied_steps=[]
            )

        start_time = time.perf_counter()
        orig_tokens = Tokenizer.count(text, model=model)
        orig_chars = len(text)

        # Run strategy-specific compression
        min_text, applied_steps = self.run_compression(text, level=level, **kwargs)

        # If a target token budget was specified, apply safe truncation if still over budget
        if target_tokens is not None and target_tokens > 0:
            current_tokens = Tokenizer.count(min_text, model=model)
            if current_tokens > target_tokens:
                min_text = Tokenizer.truncate_to_budget(min_text, target_tokens, model=model)
                applied_steps.append(f"budget_cap_{target_tokens}_tokens")

        elapsed_ms = (time.perf_counter() - start_time) * 1000.0
        final_tokens = Tokenizer.count(min_text, model=model)
        final_chars = len(min_text)

        stats = TokenStats(
            original_tokens=orig_tokens,
            minimized_tokens=final_tokens,
            original_chars=orig_chars,
            minimized_chars=final_chars,
            elapsed_ms=round(elapsed_ms, 3),
            model=model,
        )

        return MinifyResult(
            text=min_text,
            original=text,
            content_type=self.name,
            strategy_used=self.name,
            metrics=stats,
            applied_steps=applied_steps
        )

