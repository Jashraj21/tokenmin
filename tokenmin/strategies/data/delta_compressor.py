"""
Multimodal Streaming and Perceptual Delta Compressor.
Designed for real-time perceptual agents like Google Project Astra and Gemini 4.
Compresses continuous camera frames, screen updates, sensory telemetry,
and real-time audio transcripts by pruning static frames and deduplicating visual deltas.
"""
from __future__ import annotations

import re
from typing import Any, List, Tuple

from tokenmin.models import CompressionLevel
from tokenmin.strategies.base import BaseCompressor


def _collapse_streaming_frames(lines: List[str]) -> Tuple[List[str], int]:
    """Collapse consecutive near-identical camera/video frame perceptions."""
    if not lines:
        return [], 0

    collapsed: List[str] = []
    prev_line = lines[0]
    repeat_count = 1
    total_saved = 0

    for line in lines[1:]:
        # Strip frame ID/timestamp e.g. [Frame 102] or 00:01:23.400
        norm_current = re.sub(r"^\[(?:Frame\s+\d+|\d{2}:\d{2}(?::\d{2})?(?:\.\d+)?)\]\s*", "", line).strip()
        norm_prev = re.sub(r"^\[(?:Frame\s+\d+|\d{2}:\d{2}(?::\d{2})?(?:\.\d+)?)\]\s*", "", prev_line).strip()

        # Check if identical scene or flagged unchanged
        is_same = (norm_current == norm_prev) or ("unchanged" in line.lower() and norm_current.replace("[unchanged]", "").strip() == norm_prev.replace("[unchanged]", "").strip())

        if is_same and norm_current:
            repeat_count += 1
        else:
            if repeat_count > 1:
                collapsed.append(f"{prev_line} [Static across {repeat_count} frames]")
                total_saved += (repeat_count - 1)
            else:
                collapsed.append(prev_line)
            prev_line = line
            repeat_count = 1

    if repeat_count > 1:
        collapsed.append(f"{prev_line} [Static across {repeat_count} frames]")
        total_saved += (repeat_count - 1)
    else:
        collapsed.append(prev_line)

    return collapsed, total_saved


def _clean_audio_pauses(text: str) -> str:
    """Normalize repeated audio pauses or filler transcription tokens from realtime audio streams."""
    # Collapse multiple [pause 0.2s] [pause 0.3s] into single pause
    cleaned = re.sub(r"(?:\[pause(?:\s+\d+(?:\.\d+)?s)?\]\s*){2,}", "[pause] ", text)
    # Collapse repeated filler sounds: 'uh', 'um', 'mm'
    cleaned = re.sub(r"\b(?:uh|um|mm|ah)\b(?:\s*,\s*|\s+)+(?=(?:uh|um|mm|ah)\b)", "", cleaned, flags=re.IGNORECASE)
    return cleaned


class DeltaCompressor(BaseCompressor):
    """
    Compresses real-time streaming multimodal inputs for Google Astra and Gemini 4.
    Prunes static video frames, screen telemetry, and audio stream repetitions.
    """

    name = "delta"
    description = "Optimizes real-time multimodal streams (Project Astra, Gemini 4) by pruning static frame telemetry"

    def run_compression(
        self,
        text: str,
        level: CompressionLevel = CompressionLevel.MODERATE,
        **kwargs: Any
    ) -> Tuple[str, List[str]]:
        applied_steps: List[str] = []
        lines = text.splitlines()

        # 1. Collapse consecutive static frames
        collapsed_lines, saved_frames = _collapse_streaming_frames(lines)
        if saved_frames > 0:
            applied_steps.append(f"pruned_{saved_frames}_static_perceptual_frames")

        joined = "\n".join(collapsed_lines)

        # 2. Clean real-time audio pauses and stutter artifacts
        before_audio = joined
        joined = _clean_audio_pauses(joined)
        if joined != before_audio:
            applied_steps.append("condense_streaming_audio_pauses")

        # 3. Collapse multiple blank lines
        joined = re.sub(r"\n\s*\n\s*\n+", "\n\n", joined)

        return joined.strip(), applied_steps

