"""
Prompt Caching Optimizer for modern models (Claude 3.5 Sonnet, Gemini 1.5/2.0, DeepSeek V3, GPT-4o).
Stabilizes static prefixes to prevent cache-busting and isolates dynamic suffixes.
"""
from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Tuple

from tokenmin.models import CompressionLevel
from tokenmin.strategies.base import BaseCompressor
from tokenmin.tokenizer import Tokenizer


# Provider cache threshold minimums in tokens
CACHE_THRESHOLDS = {
    "claude-3-5-sonnet": 1024,
    "claude-3-5-haiku": 2048,
    "gemini-1.5-pro": 32768,
    "gemini-2.0-flash": 32768,
    "deepseek-v3": 1024,
    "gpt-4o": 1024,
}


def _strip_cache_busting_artifacts(prefix: str) -> Tuple[str, List[str]]:
    """Remove dynamic elements like timestamps, dates, or UUIDs that break prefix cache hits."""
    applied = []
    cleaned = prefix

    # Remove dynamic ISO timestamps: 2026-10-30T15:20:00Z
    iso_pattern = r"\b\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:Z|[+-]\d{2}:\d{2})?\b"
    if re.search(iso_pattern, cleaned):
        cleaned = re.sub(iso_pattern, "[TIMESTAMP_CACHED]", cleaned)
        applied.append("neutralize_dynamic_timestamps")

    # Remove random UUIDs: 4161c708-1d5a-4614-9307-99ab7e5b8a37
    uuid_pattern = r"\b[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}\b"
    if re.search(uuid_pattern, cleaned):
        cleaned = re.sub(uuid_pattern, "[SESSION_ID_CACHED]", cleaned)
        applied.append("neutralize_dynamic_uuids")

    # Deterministic line endings and trailing whitespace stripping
    lines = [line.rstrip() for line in cleaned.splitlines()]
    cleaned = "\n".join(lines).strip() + "\n"
    applied.append("canonical_prefix_whitespace")

    return cleaned, applied


class CacheOptimizer(BaseCompressor):
    """
    Optimizes prompts to maximize Prompt Caching hit rates on Claude 3.5, Gemini, DeepSeek, and GPT-4o.
    Stabilizes static prefixes and separates dynamic user queries.
    """

    name = "cache"
    description = "Optimizes prompts for prompt caching (Claude 3.5, Gemini 2.0, DeepSeek V3, GPT-4o) to achieve 50-90% discount"

    def run_compression(
        self,
        text: str,
        level: CompressionLevel = CompressionLevel.MODERATE,
        delimiter: str = "\n---\n",
        model: str = "claude-3-5-sonnet",
        **kwargs: Any
    ) -> Tuple[str, List[str]]:
        applied_steps: List[str] = []

        # If delimiter exists, treat first part as static cache prefix, second as dynamic query
        if delimiter in text:
            parts = text.split(delimiter, 1)
            prefix, suffix = parts[0], parts[1]

            stable_prefix, steps = _strip_cache_busting_artifacts(prefix)
            applied_steps.extend(steps)

            prefix_tokens = Tokenizer.count(stable_prefix, model=model)
            threshold = CACHE_THRESHOLDS.get(model, 1024)

            if prefix_tokens >= threshold:
                applied_steps.append(f"cache_eligible_{prefix_tokens}_tokens_exceeds_{threshold}_threshold")
            else:
                applied_steps.append(f"cache_below_threshold_{prefix_tokens}_vs_{threshold}_tokens")

            # Compress dynamic suffix using prompt compressor
            from tokenmin.strategies.text.prompt_compressor import PromptCompressor
            compressed_suffix, suffix_steps = PromptCompressor().run_compression(suffix, level=level)
            applied_steps.extend([f"suffix_{s}" for s in suffix_steps])

            result = f"{stable_prefix.strip()}{delimiter}{compressed_suffix.strip()}"
            return result, applied_steps

        # If no delimiter, stabilize the entire text for prefix caching
        stable, steps = _strip_cache_busting_artifacts(text)
        applied_steps.extend(steps)
        return stable.strip(), applied_steps
