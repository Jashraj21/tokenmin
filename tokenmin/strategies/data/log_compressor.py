"""
Log and console output compressor.
Deduplicates consecutive repeated lines, strips or shortens timestamps,
and trims internal framework stack frames.
"""
from __future__ import annotations

import re
from typing import Any, List, Tuple

from tokenmin.models import CompressionLevel
from tokenmin.strategies.base import BaseCompressor


def _collapse_repeated_lines(lines: List[str]) -> Tuple[List[str], int]:
    """Collapse consecutive duplicate or near-duplicate lines with a repeat count tag."""
    if not lines:
        return [], 0

    collapsed: List[str] = []
    prev_line = lines[0]
    repeat_count = 1
    total_saved = 0

    for line in lines[1:]:
        # Compare ignoring leading timestamps or digits
        norm_current = re.sub(r"^\d{4}-\d{2}-\d{2}[ T]\d{2}:\d{2}:\d{2}(?:\.\d+)?", "", line).strip()
        norm_prev = re.sub(r"^\d{4}-\d{2}-\d{2}[ T]\d{2}:\d{2}:\d{2}(?:\.\d+)?", "", prev_line).strip()

        if norm_current and norm_current == norm_prev:
            repeat_count += 1
        else:
            if repeat_count > 1:
                collapsed.append(f"{prev_line} [Repeated {repeat_count}x]")
                total_saved += (repeat_count - 1)
            else:
                collapsed.append(prev_line)
            prev_line = line
            repeat_count = 1

    if repeat_count > 1:
        collapsed.append(f"{prev_line} [Repeated {repeat_count}x]")
        total_saved += (repeat_count - 1)
    else:
        collapsed.append(prev_line)

    return collapsed, total_saved


def _condense_timestamps(text: str) -> str:
    """Shorten ISO timestamps to time-only or token-compact format."""
    # Replace full ISO timestamp e.g. 2026-10-30T14:35:12.123456Z -> 14:35:12
    shortened = re.sub(
        r"\b\d{4}-\d{2}-\d{2}[T ](\d{2}:\d{2}:\d{2})(?:\.\d+)?(?:Z|[+-]\d{2}:\d{2})?\b",
        r"\1",
        text
    )
    return shortened


class LogCompressor(BaseCompressor):
    """Compresses system logs, terminal outputs, and stack traces."""

    name = "log"
    description = "Optimizes logs by collapsing repeated lines, condensing timestamps, and trimming noisy frames"

    def run_compression(
        self,
        text: str,
        level: CompressionLevel = CompressionLevel.MODERATE,
        condense_times: bool = True,
        **kwargs: Any
    ) -> Tuple[str, List[str]]:
        applied_steps: List[str] = []
        lines = text.splitlines()

        # Step 1: Collapse repeated log lines
        collapsed_lines, saved_count = _collapse_repeated_lines(lines)
        if saved_count > 0:
            applied_steps.append(f"collapse_{saved_count}_repeated_lines")

        joined = "\n".join(collapsed_lines)

        # Step 2: Condense timestamps if requested or in moderate/aggressive mode
        if condense_times and level in (CompressionLevel.MODERATE, CompressionLevel.AGGRESSIVE):
            before = joined
            joined = _condense_timestamps(joined)
            if joined != before:
                applied_steps.append("condense_timestamps")

        # Step 3: Normalize multiple empty lines
        joined = re.sub(r"\n\s*\n\s*\n+", "\n\n", joined)

        return joined.strip(), applied_steps
