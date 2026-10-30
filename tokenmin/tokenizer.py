"""
Tokenizer abstraction supporting tiktoken and calibrated multi-model counting.
"""
from __future__ import annotations

import re
from typing import List, Optional

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
    """Multi-model token counter and text truncator."""

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
        """Count tokens in text for a given model target."""
        if not text:
            return 0

        model_lower = model.lower()

        if _TIKTOKEN_AVAILABLE:
            if "gpt-4o" in model_lower or "o1" in model_lower or "o3" in model_lower:
                enc = cls._get_tiktoken_encoding("o200k_base")
            elif "gpt-4" in model_lower or "gpt-3.5" in model_lower:
                enc = cls._get_tiktoken_encoding("cl100k_base")
            else:
                enc = cls._get_tiktoken_encoding("o200k_base") or cls._get_tiktoken_encoding("cl100k_base")

            if enc is not None:
                raw_count = len(enc.encode(text, disallowed_special=()))

                if "claude" in model_lower:
                    return max(1, int(raw_count * 1.05))
                elif "gemini" in model_lower:
                    return max(1, int(raw_count * 0.98))
                elif "llama" in model_lower:
                    return max(1, int(raw_count * 1.04))
                return raw_count

        raw = _simple_heuristic_count(text)
        if "claude" in model_lower:
            return max(1, int(raw * 1.05))
        elif "gemini" in model_lower:
            return max(1, int(raw * 0.98))
        return raw

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

                    # Step back slightly if subword re-encoding at the boundary exceeded max_tokens
                    decoded = enc.decode(token_ids)
                    while token_ids and cls.count(decoded, model=model) > max_tokens:
                        token_ids = token_ids[1:] if from_end else token_ids[:-1]
                        decoded = enc.decode(token_ids)

                    return decoded

        # Fallback line/word binary search
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
