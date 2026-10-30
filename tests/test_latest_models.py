"""Tests for latest AI models support, reasoning optimizer, prompt caching, and next-gen architectures."""
from tokenmin.models_registry import LATEST_MODELS, resolve_model
from tokenmin.pipeline import minimize
from tokenmin.strategies.cache import CacheOptimizer
from tokenmin.strategies.data import DeltaCompressor
from tokenmin.strategies.text import ReasoningCompressor
from tokenmin.tokenizer import Tokenizer, count_tokens


def test_resolve_latest_and_next_gen_models():
    # Next-Gen models
    g4 = resolve_model("gemini 4")
    assert g4.name == "gemini-4"
    assert g4.context_window == 10_000_000

    astra = resolve_model("astra")
    assert astra.name == "gemini-astra"
    assert astra.context_window == 2_000_000

    gpt6 = resolve_model("gpt 6")
    assert gpt6.name == "gpt-6"
    assert gpt6.context_window == 1_000_000
    assert gpt6.is_reasoning_model is True

    fable = resolve_model("claude fable 5")
    assert fable.name == "claude-5-fable"
    assert fable.context_window == 500_000
    assert fable.is_reasoning_model is True

    # Current flagship models
    o3 = resolve_model("o3-mini")
    assert o3.name == "o3-mini"
    assert o3.is_reasoning_model is True
    assert o3.context_window == 200_000

    claude = resolve_model("claude-3-5-sonnet")
    assert claude.name == "claude-3-5-sonnet"

    gemini = resolve_model("gemini-2.0-flash")
    assert gemini.name == "gemini-2.0-flash"


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


def test_delta_compressor_streaming_frames():
    stream_input = """[Frame 101] Camera: Workspace desk with keyboard and coffee mug
[Frame 102] Camera: Workspace desk with keyboard and coffee mug
[Frame 103] Camera: Workspace desk with keyboard and coffee mug
[Frame 104] Camera: User hand reaches for coffee mug
[pause 0.2s] [pause 0.4s] User says: What is next on my schedule?
"""
    compressor = DeltaCompressor()
    result = compressor.compress(stream_input)

    assert "[Static across 3 frames]" in result.text
    assert "User hand reaches for coffee mug" in result.text
    assert result.metrics.saved_tokens > 0


def test_pipeline_with_gpt6_and_gemini4():
    text = "Please could you kindly think step by step and explain quantum entanglement."
    res_gpt6 = minimize(text, model="gpt 6")
    assert res_gpt6.metrics.model_info.name == "gpt-6"
    assert "think step by step" not in res_gpt6.text

    res_g4 = minimize(text, model="gemini 4")
    assert res_g4.metrics.model_info.name == "gemini-4"
    assert res_g4.metrics.model_info.context_window == 10_000_000
