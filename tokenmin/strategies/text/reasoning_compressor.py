"""
Reasoning Model Prompt Optimizer.
Designed specifically for reasoning LLMs: OpenAI o1, o3-mini, DeepSeek-R1,
and Gemini 2.0 Flash Thinking.

Removes conversational ambiguity, pleasantries, and open-ended drift that cause
reasoning models to generate excessive internal thought tokens. Formats prompts
into high-density constraints and clear output specifications.
"""
from __future__ import annotations

import re
from typing import Any, List, Tuple

from tokenmin.models import CompressionLevel
from tokenmin.strategies.base import BaseCompressor


# Phrases that trigger unnecessary reasoning loops or overthinking
OVERTHINK_TRIGGERS = [
    (r"(?i)\bi\s+am\s+not\s+sure\s+if\s+", ""),
    (r"(?i)\bwhat\s+do\s+you\s+think\s+about\s+", "Analyze: "),
    (r"(?i)\bcan\s+you\s+give\s+me\s+your\s+thoughts\s+on\s+", "Evaluate: "),
    (r"(?i)\bi\s+would\s+be\s+really\s+curious\s+to\s+know\s+", ""),
    (r"(?i)\btake\s+your\s+time(?:\s+and)?(?:\s+think\s+carefully)?\s*", ""),
    (r"(?i)\bthink\s+step\s+by\s+step\b", ""),
    (r"(?i)\bmake\s+sure\s+you\s+don't\s+make\s+any\s+mistakes\b", ""),
    (r"(?i)\bexplain\s+everything\s+in\s+detail\b", "Provide precise explanation"),
    (r"(?i)\bas\s+thoroughly\s+as\s+possible\b", "precisely"),
]


class ReasoningCompressor(BaseCompressor):
    """
    Optimizes prompts for reasoning models (o1, o3-mini, DeepSeek-R1, Gemini Thinking).
    Eliminates internal reasoning token bloat by transforming ambiguous prompts
    into crisp, bounded constraint specifications.
    """

    name = "reasoning"
    description = "Optimizes prompts for o1/o3-mini/R1 by removing prompt bloat that triggers runaway thinking tokens"

    def run_compression(
        self,
        text: str,
        level: CompressionLevel = CompressionLevel.MODERATE,
        **kwargs: Any
    ) -> Tuple[str, List[str]]:
        applied_steps: List[str] = []
        cleaned = text

        # 1. Strip polite fluff and conversational preambles first
        fluff_regex = re.compile(
            r"(?i)\b(?:please\s+)?(?:could\s+you|can\s+you|would\s+you)\s+(?:kindly\s+)?(?:help\s+me\s+)?(?:to\s+)?|(?:please\s+)"
        )
        if fluff_regex.search(cleaned):
            cleaned = fluff_regex.sub("", cleaned)
            applied_steps.append("strip_polite_fluff")

        # 2. Strip redundant reasoning triggers ('think step by step', 'be careful', etc.)
        for pattern, replacement in OVERTHINK_TRIGGERS:
            before = cleaned
            cleaned = re.sub(pattern, replacement, cleaned)
            if cleaned != before:
                applied_steps.append("strip_overthink_triggers")

        # 3. Collapse multiple spaces and blank lines
        cleaned = re.sub(r"[ \t]+", " ", cleaned)
        cleaned = re.sub(r"\n\s*\n\s*\n+", "\n\n", cleaned)
        cleaned = re.sub(r"\s+([,.:;!?])", r"\1", cleaned)

        # 4. Clean orphan 'and.' or punctuation artifacts
        cleaned = re.sub(r"\band\s*[.,;:]\s*", "", cleaned, flags=re.IGNORECASE)

        # 5. In moderate or aggressive mode, ensure imperative direct tone
        lines = [line.strip() for line in cleaned.splitlines() if line.strip()]
        capped_lines = []
        for line in lines:
            if line and line[0].islower():
                capped_lines.append(line[0].upper() + line[1:])
            else:
                capped_lines.append(line)

        cleaned = "\n".join(capped_lines)
        applied_steps.append("normalize_reasoning_prompt_structure")

        return cleaned.strip(), applied_steps
