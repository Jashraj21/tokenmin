"""
Data models and configurations for TokenMin.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


class CompressionLevel(str, Enum):
    """Aggressiveness level for compression."""
    CONSERVATIVE = "conservative"
    MODERATE = "moderate"
    AGGRESSIVE = "aggressive"


class ContentType(str, Enum):
    """Type of content to compress."""
    AUTO = "auto"
    TEXT = "text"
    PROMPT = "prompt"
    PYTHON = "python"
    CODE = "code"
    JSON = "json"
    LOG = "log"
    CHAT = "chat"
    MARKDOWN = "markdown"


# Estimated input pricing per 1,000,000 tokens (USD) as of standard LLM pricing
MODEL_INPUT_PRICES_PER_M: Dict[str, float] = {
    "gpt-4o": 2.50,
    "gpt-4o-mini": 0.15,
    "claude-3-5-sonnet": 3.00,
    "claude-3-haiku": 0.25,
    "gemini-1.5-pro": 1.25,
    "gemini-1.5-flash": 0.075,
    "gemini-2.0-flash": 0.10,
}


@dataclass
class TokenStats:
    """Metrics and statistics comparing original vs minimized content."""
    original_tokens: int
    minimized_tokens: int
    original_chars: int
    minimized_chars: int
    elapsed_ms: float = 0.0
    model: str = "gpt-4o"

    @property
    def saved_tokens(self) -> int:
        return max(0, self.original_tokens - self.minimized_tokens)

    @property
    def reduction_percentage(self) -> float:
        """Percentage of tokens saved (0.0% to 100.0%)."""
        if self.original_tokens <= 0:
            return 0.0
        return round((self.saved_tokens / self.original_tokens) * 100.0, 2)

    @property
    def compression_ratio(self) -> float:
        """Ratio of minimized to original tokens (e.g., 0.40 = 40% of original size)."""
        if self.original_tokens <= 0:
            return 1.0
        return round(self.minimized_tokens / self.original_tokens, 4)

    @property
    def char_reduction_percentage(self) -> float:
        """Percentage of characters saved."""
        if self.original_chars <= 0:
            return 0.0
        saved = max(0, self.original_chars - self.minimized_chars)
        return round((saved / self.original_chars) * 100.0, 2)

    def estimated_savings(self, requests_count: int = 1_000_000) -> Dict[str, float]:
        """Estimate dollars saved across major models for a given number of API calls."""
        savings: Dict[str, float] = {}
        for model_name, price_per_m in MODEL_INPUT_PRICES_PER_M.items():
            saved_dollars = (self.saved_tokens * requests_count / 1_000_000.0) * price_per_m
            savings[model_name] = round(saved_dollars, 4)
        return savings

    def summary(self) -> str:
        """Human-readable summary of token savings."""
        return (
            f"Tokens: {self.original_tokens} -> {self.minimized_tokens} "
            f"(-{self.saved_tokens} tokens, {self.reduction_percentage}% reduction) "
            f"in {self.elapsed_ms:.2f}ms"
        )


@dataclass
class MinifyResult:
    """Result of a token minimization operation."""
    text: str
    original: str
    content_type: str
    strategy_used: str
    metrics: TokenStats
    applied_steps: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __str__(self) -> str:
        return self.text

    def diff(self) -> str:
        """Return a unified diff between original and minimized text."""
        import difflib
        original_lines = self.original.splitlines(keepends=True)
        minimized_lines = self.text.splitlines(keepends=True)
        diff = difflib.unified_diff(
            original_lines,
            minimized_lines,
            fromfile="original",
            tofile="minimized",
            n=2
        )
        return "".join(diff)
