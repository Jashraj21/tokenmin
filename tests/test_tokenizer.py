"""Tests for tokenizer and budget truncation."""
from tokenmin.tokenizer import Tokenizer, count_tokens


def test_tokenizer_empty():
    assert count_tokens("") == 0
    assert Tokenizer.truncate_to_budget("", 10) == ""


def test_tokenizer_basic_counting():
    text = "The quick brown fox jumps over the lazy dog."
    tokens = count_tokens(text, model="gpt-4o")
    assert tokens > 0
    assert isinstance(tokens, int)


def test_tokenizer_model_calibrations():
    text = "Explain the architectural differences between transformers and state space models."
    gpt4o = Tokenizer.count(text, model="gpt-4o")
    claude = Tokenizer.count(text, model="claude-3-5-sonnet")
    gemini = Tokenizer.count(text, model="gemini-1.5-pro")

    assert gpt4o > 0
    assert claude > 0
    assert gemini > 0


def test_tokenizer_truncate_to_budget():
    text = "Line 1: Alpha\nLine 2: Beta\nLine 3: Gamma\nLine 4: Delta\nLine 5: Epsilon"
    orig_tokens = count_tokens(text)
    budget = orig_tokens // 2

    truncated = Tokenizer.truncate_to_budget(text, max_tokens=budget, model="gpt-4o")
    trunc_tokens = count_tokens(truncated)
    assert trunc_tokens <= budget
    assert "Line 1" in truncated


def test_tokenizer_truncate_from_end():
    text = "Old turn 1\nOld turn 2\nRecent turn 3\nRecent turn 4"
    orig_tokens = count_tokens(text)
    budget = orig_tokens // 2

    truncated = Tokenizer.truncate_to_budget(text, max_tokens=budget, from_end=True)
    trunc_tokens = count_tokens(truncated)
    assert trunc_tokens <= budget
    assert "Recent" in truncated
