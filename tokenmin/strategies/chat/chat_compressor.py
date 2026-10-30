"""
Chat and multi-turn conversation compressor.
Optimizes message history (JSON message list or formatted text) by preserving
recent dialogue, condensing older assistant turns, and deduplicating context.
"""
from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional, Tuple, Union

from tokenmin.models import CompressionLevel
from tokenmin.strategies.base import BaseCompressor


def _compress_message_list(
    messages: List[Dict[str, Any]],
    keep_recent_turns: int = 4,
    level: CompressionLevel = CompressionLevel.MODERATE
) -> Tuple[List[Dict[str, Any]], List[str]]:
    """Compress a list of OpenAI/Anthropic format message dicts."""
    applied_steps = []
    if len(messages) <= keep_recent_turns + 1:
        return messages, applied_steps

    compressed = []
    total_messages = len(messages)
    split_index = max(1, total_messages - keep_recent_turns)

    # 1. Preserve system prompt (if first message)
    start_idx = 0
    if messages and messages[0].get("role") == "system":
        compressed.append(messages[0])
        start_idx = 1

    # 2. Older messages: condense assistant and tool responses
    older_messages = messages[start_idx:split_index]
    condensed_count = 0

    for msg in older_messages:
        role = msg.get("role", "")
        content = msg.get("content", "")

        if isinstance(content, str):
            # If assistant or tool message is long, condense it
            if role in ("assistant", "tool") and len(content) > 200:
                condensed_count += 1
                lines = [line.strip() for line in content.splitlines() if line.strip()]
                # Keep first line and condensed note
                first_line = lines[0] if lines else ""
                shortened = f"{first_line} ... [Condensed {len(content)} chars]"
                new_msg = dict(msg)
                new_msg["content"] = shortened
                compressed.append(new_msg)
            else:
                compressed.append(msg)
        else:
            compressed.append(msg)

    if condensed_count > 0:
        applied_steps.append(f"condensed_{condensed_count}_older_turn_messages")

    # 3. Recent messages: keep full fidelity
    compressed.extend(messages[split_index:])
    applied_steps.append(f"preserved_last_{keep_recent_turns}_turns")

    return compressed, applied_steps


class ChatCompressor(BaseCompressor):
    """Compresses conversational dialogue and agent context history."""

    name = "chat"
    description = "Optimizes chat history and agent memory by preserving recent context and condensing older turns"

    def run_compression(
        self,
        text: str,
        level: CompressionLevel = CompressionLevel.MODERATE,
        keep_recent_turns: int = 4,
        **kwargs: Any
    ) -> Tuple[str, List[str]]:
        applied_steps: List[str] = []

        # Check if text is JSON message array: [{"role": "...", "content": "..."}]
        stripped = text.strip()
        if stripped.startswith("[") and stripped.endswith("]"):
            try:
                parsed = json.loads(stripped)
                if isinstance(parsed, list) and all(isinstance(item, dict) and "role" in item for item in parsed):
                    compressed_msgs, steps = _compress_message_list(
                        parsed,
                        keep_recent_turns=keep_recent_turns,
                        level=level
                    )
                    applied_steps.extend(steps)
                    applied_steps.append("compact_json_chat_output")
                    return json.dumps(compressed_msgs, separators=(",", ":"), ensure_ascii=False), applied_steps
            except Exception:
                pass

        # Text format chat parsing (e.g. "User: ... \nAssistant: ...")
        turn_pattern = re.compile(r"(?=(?:^|\n)(?:User|Assistant|System|Human|AI):\s*)", re.MULTILINE)
        raw_turns = [t.strip() for t in turn_pattern.split(text) if t.strip()]

        if len(raw_turns) > keep_recent_turns:
            preserved_recent = raw_turns[-keep_recent_turns:]
            older_turns = raw_turns[:-keep_recent_turns]

            condensed_older = []
            for turn in older_turns:
                # Condense older AI turns if verbose
                if re.match(r"^(?:Assistant|AI):", turn, re.IGNORECASE) and len(turn) > 200:
                    lines = turn.splitlines()
                    short_turn = f"{lines[0]} ... [Prior response condensed]"
                    condensed_older.append(short_turn)
                else:
                    condensed_older.append(turn)

            applied_steps.append(f"condense_text_chat_{len(older_turns)}_older_turns")
            applied_steps.append(f"preserve_last_{keep_recent_turns}_turns")
            result = "\n\n".join(condensed_older + preserved_recent)
            return result, applied_steps

        # If few turns, normalize whitespace
        cleaned = re.sub(r"\n\s*\n\s*\n+", "\n\n", text).strip()
        applied_steps.append("normalize_whitespace")
        return cleaned, applied_steps

