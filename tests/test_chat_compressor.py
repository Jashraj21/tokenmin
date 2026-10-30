"""Tests for chat and conversational dialogue compressor."""
import json

from tokenmin.models import CompressionLevel
from tokenmin.strategies.chat import ChatCompressor


def test_json_chat_compression():
    messages = [
        {"role": "system", "content": "You are a senior python developer."},
        {"role": "user", "content": "How do I reverse a string?"},
        {
            "role": "assistant",
            "content": "To reverse a string in Python, you can use slice notation text[::-1]. " * 10
        },
        {"role": "user", "content": "What about reversing words in a sentence?"},
        {"role": "assistant", "content": "You can split words and reverse the list: ' '.join(s.split()[::-1])"},
        {"role": "user", "content": "Can you benchmark that?"},
        {"role": "assistant", "content": "Sure, here is the timeit comparison."},
    ]
    raw = json.dumps(messages)
    compressor = ChatCompressor()
    result = compressor.compress(raw, keep_recent_turns=2)

    parsed = json.loads(result.text)
    assert parsed[0]["role"] == "system"
    assert parsed[0]["content"] == "You are a senior python developer."
    # Older assistant message should be condensed
    assert "Condensed" in parsed[2]["content"]
    # Recent turns must be preserved intact
    assert "Sure, here is the timeit comparison." == parsed[-1]["content"]


def test_text_chat_compression():
    chat_text = """User: Hello there
Assistant: Welcome! How can I assist you with your project today? Let me know all your requirements in detail.
User: Can you explain recursion?
Assistant: Recursion is when a function calls itself until reaching a base condition. It has two main parts: the base case and the recursive step.
User: Give me a simple example.
Assistant: def factorial(n): return 1 if n <= 1 else n * factorial(n - 1)
"""
    compressor = ChatCompressor()
    result = compressor.compress(chat_text, keep_recent_turns=2)

    assert result.metrics.saved_tokens >= 0
    assert "def factorial(n):" in result.text

