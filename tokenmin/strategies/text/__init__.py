"""Text and prompt compression strategies."""
from tokenmin.strategies.text.markdown_compressor import MarkdownCompressor
from tokenmin.strategies.text.prompt_compressor import PromptCompressor

__all__ = ["PromptCompressor", "MarkdownCompressor"]
