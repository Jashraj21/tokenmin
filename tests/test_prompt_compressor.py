"""Tests for prompt and natural language compressor."""
from tokenmin.models import CompressionLevel
from tokenmin.strategies.text import PromptCompressor


def test_prompt_fluff_removal():
    prompt = (
        "Please can you kindly write a python function to parse CSV files. "
        "I would like you to make sure that you handle file not found exceptions. "
        "Please note that you should return a list of dictionaries."
    )
    compressor = PromptCompressor()
    result = compressor.compress(prompt)

    assert result.metrics.saved_tokens > 0
    assert "Please can you kindly" not in result.text
    assert "I would like you to" not in result.text
    assert "parse CSV files" in result.text
    assert "return a list of dictionaries" in result.text


def test_wordy_phrase_replacements():
    text = (
        "In order to compile the application, due to the fact that dependencies have updated, "
        "run the build script prior to execution. In the event that errors occur, report them."
    )
    compressor = PromptCompressor()
    result = compressor.compress(text)

    # Replaced: "In order to" -> "To", "due to the fact that" -> "because", "prior to" -> "before", "in the event that" -> "if"
    assert "due to the fact that" not in result.text
    assert "because" in result.text
    assert "before" in result.text
    assert "if" in result.text.lower()


def test_whitespace_and_punctuation_cleanup():
    text = "Here is an instruction   with   excessive   spaces  , and multiple marks??? Really!!!"
    compressor = PromptCompressor()
    result = compressor.compress(text)

    assert "   " not in result.text
    assert "spaces," in result.text
    assert "???" not in result.text
    assert "!!!" not in result.text
