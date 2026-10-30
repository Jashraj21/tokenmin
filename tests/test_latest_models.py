"""Tests for latest AI models support, reasoning optimizer, and prompt caching."""
from tokenmin.models_registry import LATEST_MODELS, resolve_model
from tokenmin.pipeline import minimize
from tokenmin.strategies.cache import CacheOptimizer
from tokenmin.strategies.text import ReasoningCompressor
from tokenmin.tokenizer import Tokenizer, count_tokens


def test_resolve_latest_models():
    # OpenAI Reasoning models
    o3 = resolve_model("o3-mini")
    assert o3.name == "o3-mini"
    assert o3.is_reasoning_model is True
    assert o3.context_window == 200_000

    o1 = resolve_model("o1")
    assert o1.name == "o1"
    assert o1.is_reasoning_model is True

    # Anthropic
    claude = resolve_model("claude-3-5-sonnet")
    assert claude.name == "claude-3-5-sonnet"
    assert claude.context_window == 200_000
    assert claude.cached_input_price_per_m < claude.input_price_per_m

    # Google Gemini 2.0
    gemini = resolve_model("gemini-2.0-flash")
    assert gemini.name == "gemini-2.0-flash"
    assert gemini.context_window >= 1_000_000

    # DeepSeek
    r1 = resolve_model("deepseek-r1")
    assert r1.name == "deepseek-r1"
    assert r1.is_reasoning_model is True

    v3 = resolve_model("deepseek-v3")
    assert v3.name == "deepseek-v3"


def test_reasoning_compressor_strips_thinking_loops():
    prompt = (
        "Please take your time and think step by step. "
        "What do you think about implementing a distributed lock with Redis? "
        "Make sure you don't make any mistakes."
    )
    compressor = ReasoningCompressor()
    result = compressor.compress(prompt)

    assert "think step by step" not in result.text
    assert "take your time" not in result.text
    assert "make sure you don't make any mistakes" not in result.text
    assert result.metrics.saved_tokens > 0


def test_cache_optimizer_stabilization():
    raw_prompt = (
        "System prompt with dynamic date: 2026-10-30T15:00:00Z and session id 4161c708-1d5a-4614-9307-99ab7e5b8a37.\n"
        "---\n"
        "User question: Please can you summarize the findings?"
    )
    compressor = CacheOptimizer()
    result = compressor.compress(raw_prompt)

    assert "2026-10-30T15:00:00Z" not in result.text
    assert "4161c708-1d5a-4614-9307-99ab7e5b8a37" not in result.text
    assert "[TIMESTAMP_CACHED]" in result.text
    assert "[SESSION_ID_CACHED]" in result.text
    assert "Please can you" not in result.text


def test_pipeline_with_latest_reasoning_model():
    text = "Please could you kindly think step by step and solve this equation: 3x + 5 = 20."
    # With o3-mini target model, pipeline automatically activates reasoning optimization
    res = minimize(text, model="o3-mini")

    assert "think step by step" not in res.text
    assert "Please could you kindly" not in res.text
    assert "o3-mini" in res.metrics.model
    assert res.metrics.model_info.is_reasoning_model is True


def test_context_utilization_calculation():
    text = "Short prompt for testing context."
    res = minimize(text, model="gemini-2.0-flash")

    # In 1M context, a short prompt is < 0.01%
    assert res.metrics.context_usage_percentage() < 0.01
