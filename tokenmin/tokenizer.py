"""
Tokenizer abstraction supporting tiktoken, calibrated counting for latest models,
and context window limit validation.
"""
from __future__ import annotations

import re
from typing import List, Optional

from tokenmin.models_registry import ModelInfo, resolve_model

try:
    import tiktoken
    _TIKTOKEN_AVAILABLE = True
except ImportError:
    _TIKTOKEN_AVAILABLE = False


def _simple_heuristic_count(text: str) -> int:
    """Accurate fallback heuristic token counter using subword-like regex."""
    if not text:
        return 0
    tokens = re.findall(r"\w+|[^\w\s]|\n|\r\n", text)
    count = 0
    for tok in tokens:
        if len(tok) > 6:
            count += max(1, (len(tok) + 3) // 4)
        else:
            count += 1
    return max(1, count) if text.strip() else count


class Tokenizer:
    """Multi-model token counter and text truncator calibrated for latest AI models."""

    _encoders: dict = {}

    @classmethod
    def _get_tiktoken_encoding(cls, encoding_name: str):
        if not _TIKTOKEN_AVAILABLE:
            return None
        if encoding_name not in cls._encoders:
            try:
                cls._encoders[encoding_name] = tiktoken.get_encoding(encoding_name)
            except Exception:
                try:
                    cls._encoders[encoding_name] = tiktoken.encoding_for_model(encoding_name)
                except Exception:
                    return None
        return cls._encoders.get(encoding_name)

    @classmethod
    def count(cls, text: str, model: str = "gpt-4o") -> int:
        """
        Count tokens in text calibrated for the target model.
        Supports OpenAI (o1, o3-mini, GPT-4o), Claude 3.5, Gemini 2.0, DeepSeek V3/R1, Llama 3.3.
        """
        if not text:
            return 0

        info = resolve_model(model)
        model_name = info.name

        if _TIKTOKEN_AVAILABLE:
            # Modern frontier models use o200k or equivalent large vocabularies
            if any(k in model_name for k in ("gpt-4o", "o1", "o3", "gemini", "claude", "deepseek", "llama")):
                enc = cls._get_tiktoken_encoding("o200k_base") or cls._get_tiktoken_encoding("cl100k_base")
            else:
                enc = cls._get_tiktoken_encoding("cl100k_base") or cls._get_tiktoken_encoding("o200k_base")

            if enc is not None:
                raw_count = len(enc.encode(text, disallowed_special=()))
                calibrated = int(raw_count * info.tokenizer_calibration)
                return max(1, calibrated)

        # Fallback heuristic
        raw = _simple_heuristic_count(text)
        return max(1, int(raw * info.tokenizer_calibration))

    @classmethod
    def get_context_window(cls, model: str = "gpt-4o") -> int:
        """Return the maximum context window in tokens for the model."""
        return resolve_model(model).context_window

    @classmethod
    def truncate_to_budget(
        cls,
        text: str,
        max_tokens: int,
        model: str = "gpt-4o",
        from_end: bool = False
    ) -> str:
        """
        Truncate text so it strictly fits within max_tokens for the given model.
        """
        if max_tokens <= 0:
            return ""
        current_tokens = cls.count(text, model=model)
        if current_tokens <= max_tokens:
            return text

        if _TIKTOKEN_AVAILABLE:
            enc = cls._get_tiktoken_encoding("o200k_base") or cls._get_tiktoken_encoding("cl100k_base")
            if enc is not None:
                token_ids = enc.encode(text, disallowed_special=())
                if len(token_ids) > max_tokens:
                    if from_end:
                        token_ids = token_ids[-max_tokens:]
                    else:
                        token_ids = token_ids[:max_tokens]

                    decoded = enc.decode(token_ids)
                    while token_ids and cls.count(decoded, model=model) > max_tokens:
                        token_ids = token_ids[1:] if from_end else token_ids[:-1]
                        decoded = enc.decode(token_ids)

                    return decoded

        # Fallback binary search over lines
        lines = text.splitlines(keepends=True)
        if not lines:
            return text[:max_tokens * 3]

        low, high = 0, len(lines)
        best_text = ""
        while low <= high:
            mid = (low + high) // 2
            cand = "".join(lines[-mid:]) if from_end else "".join(lines[:mid])
            cand_tokens = cls.count(cand, model=model)
            if cand_tokens <= max_tokens:
                best_text = cand
                low = mid + 1
            else:
                high = mid - 1

        return best_text if best_text else text[:max_tokens * 3]


def count_tokens(text: str, model: str = "gpt-4o") -> int:
    """Convenience function to count tokens in text."""
    return Tokenizer.count(text, model=model)
