"""
Registry of latest frontier AI models with context window limits, pricing, and token calibration.
Includes OpenAI (GPT-4o, o1, o3-mini), Anthropic (Claude 3.5 Sonnet/Haiku), Google (Gemini 2.0 Flash/Pro),
DeepSeek (V3, R1), and Meta (Llama 3.3).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional


@dataclass(frozen=True)
class ModelInfo:
    """Specification and pricing for an AI model."""
    name: str
    display_name: str
    provider: str
    context_window: int
    input_price_per_m: float          # Standard input rate per 1M tokens (USD)
    cached_input_price_per_m: float   # Cached input rate per 1M tokens (USD)
    tokenizer_calibration: float       # Ratio relative to standard o200k token count
    is_reasoning_model: bool = False
    supports_caching: bool = True
    notes: str = ""


# Registry of the latest models
LATEST_MODELS: Dict[str, ModelInfo] = {
    # OpenAI
    "gpt-4o": ModelInfo(
        name="gpt-4o",
        display_name="OpenAI GPT-4o",
        provider="OpenAI",
        context_window=128_000,
        input_price_per_m=2.50,
        cached_input_price_per_m=1.25,
        tokenizer_calibration=1.0,
        is_reasoning_model=False,
        supports_caching=True,
    ),
    "gpt-4o-mini": ModelInfo(
        name="gpt-4o-mini",
        display_name="OpenAI GPT-4o mini",
        provider="OpenAI",
        context_window=128_000,
        input_price_per_m=0.15,
        cached_input_price_per_m=0.075,
        tokenizer_calibration=1.0,
        is_reasoning_model=False,
        supports_caching=True,
    ),
    "o1": ModelInfo(
        name="o1",
        display_name="OpenAI o1",
        provider="OpenAI",
        context_window=200_000,
        input_price_per_m=15.00,
        cached_input_price_per_m=7.50,
        tokenizer_calibration=1.0,
        is_reasoning_model=True,
        supports_caching=True,
        notes="High-depth reasoning model"
    ),
    "o3-mini": ModelInfo(
        name="o3-mini",
        display_name="OpenAI o3-mini",
        provider="OpenAI",
        context_window=200_000,
        input_price_per_m=1.10,
        cached_input_price_per_m=0.55,
        tokenizer_calibration=1.0,
        is_reasoning_model=True,
        supports_caching=True,
        notes="Fast STEM & coding reasoning model"
    ),

    # Anthropic
    "claude-3-5-sonnet": ModelInfo(
        name="claude-3-5-sonnet",
        display_name="Claude 3.5 Sonnet",
        provider="Anthropic",
        context_window=200_000,
        input_price_per_m=3.00,
        cached_input_price_per_m=0.30,
        tokenizer_calibration=1.06,
        is_reasoning_model=False,
        supports_caching=True,
        notes="Prompt caching gives 90% discount"
    ),
    "claude-3-5-haiku": ModelInfo(
        name="claude-3-5-haiku",
        display_name="Claude 3.5 Haiku",
        provider="Anthropic",
        context_window=200_000,
        input_price_per_m=0.80,
        cached_input_price_per_m=0.08,
        tokenizer_calibration=1.06,
        is_reasoning_model=False,
        supports_caching=True,
    ),

    # Google Gemini
    "gemini-2.0-flash": ModelInfo(
        name="gemini-2.0-flash",
        display_name="Gemini 2.0 Flash",
        provider="Google",
        context_window=1_048_576,
        input_price_per_m=0.10,
        cached_input_price_per_m=0.025,
        tokenizer_calibration=0.96,
        is_reasoning_model=False,
        supports_caching=True,
        notes="High-speed multimodal, 1M context"
    ),
    "gemini-2.0-flash-thinking": ModelInfo(
        name="gemini-2.0-flash-thinking",
        display_name="Gemini 2.0 Flash Thinking",
        provider="Google",
        context_window=1_048_576,
        input_price_per_m=0.10,
        cached_input_price_per_m=0.025,
        tokenizer_calibration=0.96,
        is_reasoning_model=True,
        supports_caching=True,
        notes="Reasoning with visible thoughts"
    ),
    "gemini-1.5-pro": ModelInfo(
        name="gemini-1.5-pro",
        display_name="Gemini 1.5 Pro",
        provider="Google",
        context_window=2_097_152,
        input_price_per_m=1.25,
        cached_input_price_per_m=0.3125,
        tokenizer_calibration=0.96,
        is_reasoning_model=False,
        supports_caching=True,
        notes="2M token context window"
    ),
    "gemini-1.5-flash": ModelInfo(
        name="gemini-1.5-flash",
        display_name="Gemini 1.5 Flash",
        provider="Google",
        context_window=1_048_576,
        input_price_per_m=0.075,
        cached_input_price_per_m=0.01875,
        tokenizer_calibration=0.96,
        is_reasoning_model=False,
        supports_caching=True,
    ),

    # DeepSeek
    "deepseek-v3": ModelInfo(
        name="deepseek-v3",
        display_name="DeepSeek V3",
        provider="DeepSeek",
        context_window=128_000,
        input_price_per_m=0.14,
        cached_input_price_per_m=0.014,
        tokenizer_calibration=1.01,
        is_reasoning_model=False,
        supports_caching=True,
        notes="671B MoE frontier model"
    ),
    "deepseek-r1": ModelInfo(
        name="deepseek-r1",
        display_name="DeepSeek R1",
        provider="DeepSeek",
        context_window=128_000,
        input_price_per_m=0.55,
        cached_input_price_per_m=0.14,
        tokenizer_calibration=1.01,
        is_reasoning_model=True,
        supports_caching=True,
        notes="Open-weights reasoning model"
    ),

    # Meta Llama
    "llama-3.3-70b": ModelInfo(
        name="llama-3.3-70b",
        display_name="Llama 3.3 70B",
        provider="Meta",
        context_window=128_000,
        input_price_per_m=0.59,
        cached_input_price_per_m=0.30,
        tokenizer_calibration=1.03,
        is_reasoning_model=False,
        supports_caching=False,
        notes="Industry-standard open model"
    ),
    "llama-3.1-405b": ModelInfo(
        name="llama-3.1-405b",
        display_name="Llama 3.1 405B",
        provider="Meta",
        context_window=128_000,
        input_price_per_m=2.00,
        cached_input_price_per_m=1.00,
        tokenizer_calibration=1.03,
        is_reasoning_model=False,
        supports_caching=False,
    ),
    "qwen-2.5-coder-32b": ModelInfo(
        name="qwen-2.5-coder-32b",
        display_name="Qwen 2.5 Coder 32B",
        provider="Alibaba",
        context_window=128_000,
        input_price_per_m=0.30,
        cached_input_price_per_m=0.15,
        tokenizer_calibration=1.02,
        is_reasoning_model=False,
        supports_caching=False,
        notes="State of the art open code model"
    ),
}


def resolve_model(query: str) -> ModelInfo:
    """
    Resolve model string to ModelInfo with fuzzy matching for version abbreviations.
    E.g., 'sonnet', 'claude-3.5', 'r1', 'o3', 'flash-2.0', 'gemini' all resolve correctly.
    """
    q = query.lower().strip()
    if q in LATEST_MODELS:
        return LATEST_MODELS[q]

    # Alias / fuzzy mappings
    if "o3" in q:
        return LATEST_MODELS["o3-mini"]
    if "o1" in q:
        return LATEST_MODELS["o1"]
    if "r1" in q or "deepseek-r" in q:
        return LATEST_MODELS["deepseek-r1"]
    if "deepseek" in q or "v3" in q:
        return LATEST_MODELS["deepseek-v3"]
    if "sonnet" in q:
        return LATEST_MODELS["claude-3-5-sonnet"]
    if "haiku" in q:
        return LATEST_MODELS["claude-3-5-haiku"]
    if "2.0-flash" in q or "gemini-2" in q:
        return LATEST_MODELS["gemini-2.0-flash"]
    if "thinking" in q:
        return LATEST_MODELS["gemini-2.0-flash-thinking"]
    if "1.5-pro" in q:
        return LATEST_MODELS["gemini-1.5-pro"]
    if "gemini" in q or "flash" in q:
        return LATEST_MODELS["gemini-2.0-flash"]
    if "llama-3.3" in q or "llama-3" in q:
        return LATEST_MODELS["llama-3.3-70b"]
    if "qwen" in q:
        return LATEST_MODELS["qwen-2.5-coder-32b"]
    if "4o-mini" in q:
        return LATEST_MODELS["gpt-4o-mini"]
    if "4o" in q or "gpt-4" in q:
        return LATEST_MODELS["gpt-4o"]

    # Fallback to GPT-4o
    return LATEST_MODELS["gpt-4o"]


def list_supported_models() -> List[str]:
    """Return all canonical supported model identifiers."""
    return list(LATEST_MODELS.keys())
