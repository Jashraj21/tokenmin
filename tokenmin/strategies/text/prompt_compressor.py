"""
Prompt and natural language compressor.
Eliminates conversational fluff, replaces wordy transitions, normalizes spacing,
and prunes low-entropy filler phrases while strictly preserving instructions.
"""
from __future__ import annotations

import math
import re
from collections import Counter
from typing import Any, List, Optional, Set, Tuple

from tokenmin.models import CompressionLevel
from tokenmin.strategies.base import BaseCompressor


# Conversational polite fluff to remove or condense at the beginning of sentences/prompts
FLUFF_PATTERNS = [
    (r"(?i)\b(?:please\s+)?(?:can\s+you|could\s+you|would\s+you)\s+(?:kindly\s+)?(?:please\s+)?", ""),
    (r"(?i)\bplease\s+(?:kindly\s+)?", ""),
    (r"(?i)\bkindly\s+", ""),
    (r"(?i)\bi\s+(?:would\s+like|want)\s+you\s+to\s+", ""),
    (r"(?i)\bi\s+need\s+you\s+to\s+", ""),
    (r"(?i)\b(?:it\s+is\s+important\s+to\s+note\s+that|please\s+note\s+that)\s*", "Note: "),
    (r"(?i)\bmake\s+sure\s+(?:that\s+)?you\s+", "Ensure you "),
    (r"(?i)\bbe\s+sure\s+to\s+", "Ensure to "),
    (r"(?i)\bi\s+am\s+looking\s+for\s+(?:a|an)\s+", ""),
    (r"(?i)\bfeel\s+free\s+to\s+", ""),
    (r"(?i)\bas\s+an\s+ai(?:\s+assistant|\s+language\s+model)?,\s*", ""),
    (r"(?i)\bcan\s+you\s+help\s+me\s+(?:with\s+)?", ""),
]

# Verbose multi-word phrases mapped to concise equivalents
WORDY_REPLACEMENTS = [
    (r"(?i)\bin\s+order\s+to\b", "to"),
    (r"(?i)\bdue\s+to\s+the\s+fact\s+that\b", "because"),
    (r"(?i)\bat\s+this\s+point\s+in\s+time\b", "now"),
    (r"(?i)\bat\s+the\s+present\s+time\b", "currently"),
    (r"(?i)\bin\s+the\s+event\s+that\b", "if"),
    (r"(?i)\bfor\s+the\s+purpose\s+of\b", "for"),
    (r"(?i)\btake\s+into\s+consideration\b", "consider"),
    (r"(?i)\bas\s+well\s+as\b", "and"),
    (r"(?i)\ba\s+variety\s+of\b", "various"),
    (r"(?i)\ba\s+large\s+number\s+of\b", "many"),
    (r"(?i)\bhas\s+the\s+ability\s+to\b", "can"),
    (r"(?i)\bis\s+able\s+to\b", "can"),
    (r"(?i)\bwith\s+regard(?:s)?\s+to\b", "regarding"),
    (r"(?i)\bin\s+spite\s+of\s+the\s+fact\s+that\b", "although"),
    (r"(?i)\bprior\s+to\b", "before"),
    (r"(?i)\bsubsequent\s+to\b", "after"),
    (r"(?i)\buntil\s+such\s+time\s+as\b", "until"),
    (r"(?i)\bin\s+accordance\s+with\b", "per"),
    (r"(?i)\bwith\s+the\s+exception\s+of\b", "except"),
    (r"(?i)\bon\s+a\s+daily\s+basis\b", "daily"),
    (r"(?i)\ba\s+majority\s+of\b", "most"),
    (r"(?i)\bneedless\s+to\s+say,\s*", ""),
    (r"(?i)\bas\s+a\s+matter\s+of\s+fact,\s*", ""),
    (r"(?i)\bit\s+goes\s+without\s+saying\s+that,\s*", ""),
]

# Common English stopwords used for lexical density calculation
STOPWORDS = {
    "a", "about", "above", "after", "again", "against", "all", "am", "an", "and", "any", "are",
    "aren't", "as", "at", "be", "because", "been", "before", "being", "below", "between", "both",
    "but", "by", "can't", "cannot", "could", "couldn't", "did", "didn't", "do", "does", "doesn't",
    "doing", "don't", "down", "during", "each", "few", "for", "from", "further", "had", "hadn't",
    "has", "hasn't", "have", "haven't", "having", "he", "he'd", "he'll", "he's", "her", "here",
    "here's", "hers", "herself", "him", "himself", "his", "how", "how's", "i", "i'd", "i'll", "i'm",
    "i've", "if", "in", "into", "is", "isn't", "it", "it's", "its", "itself", "let's", "me", "more",
    "most", "mustn't", "my", "myself", "no", "nor", "not", "of", "off", "on", "once", "only", "or",
    "other", "ought", "our", "ours", "ourselves", "out", "over", "own", "same", "shan't", "she",
    "she'd", "she'll", "she's", "should", "shouldn't", "so", "some", "such", "than", "that",
    "that's", "the", "their", "theirs", "them", "themselves", "then", "there", "there's", "these",
    "they", "they'd", "they'll", "they're", "they've", "this", "those", "through", "to", "too",
    "under", "until", "up", "very", "was", "wasn't", "we", "we'd", "we'll", "we're", "we've", "were",
    "weren't", "what", "what's", "when", "when's", "where", "where's", "which", "while", "who",
    "who's", "whom", "why", "why's", "with", "won't", "would", "wouldn't", "you", "you'd", "you'll",
    "you're", "you've", "your", "yours", "yourself", "yourselves"
}


def _calculate_sentence_information_density(sentence: str) -> float:
    """Calculate the information density score of a sentence based on non-stopword lexical variety."""
    words = re.findall(r"\b\w+\b", sentence.lower())
    if not words:
        return 0.0

    content_words = [w for w in words if w not in STOPWORDS]
    if not content_words:
        return 0.1

    # Ratio of informative content words to total words
    content_ratio = len(content_words) / len(words)
    # Unique informative words ratio
    unique_content = len(set(content_words)) / len(content_words)

    # Length penalty for extremely short or extremely long fragments
    length_factor = min(1.0, len(words) / 5.0)

    return content_ratio * unique_content * length_factor


def _extractive_sentence_prune(text: str, target_ratio: float = 0.7) -> str:
    """
    Keep highest-information sentences while preserving narrative order
    and always retaining first and last sentences (intent + constraints).
    """
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    if len(sentences) <= 3:
        return text

    target_count = max(2, int(len(sentences) * target_ratio))
    if target_count >= len(sentences):
        return text

    scored: List[Tuple[int, float, str]] = []
    for idx, sent in enumerate(sentences):
        # Anchor first and last sentences with high scores
        if idx == 0 or idx == len(sentences) - 1:
            score = 999.0
        else:
            score = _calculate_sentence_information_density(sent)
        scored.append((idx, score, sent))

    # Pick top N sentences by score
    scored_sorted = sorted(scored, key=lambda x: x[1], reverse=True)
    selected_indices = sorted([item[0] for item in scored_sorted[:target_count]])

    result = [sentences[i] for i in selected_indices]
    return " ".join(result)


class PromptCompressor(BaseCompressor):
    """Compresses prompts and instructions while preserving all constraints and task directives."""

    name = "prompt"
    description = "Optimizes natural language prompts by pruning filler, wordy transitions, and redundant phrasing"

    def run_compression(
        self,
        text: str,
        level: CompressionLevel = CompressionLevel.MODERATE,
        preserve_lists: bool = True,
        **kwargs: Any
    ) -> Tuple[str, List[str]]:
        applied_steps: List[str] = []
        cleaned = text

        # Step 1: Remove conversational fluff
        original_len = len(cleaned)
        for pattern, replacement in FLUFF_PATTERNS:
            cleaned = re.sub(pattern, replacement, cleaned)
        if len(cleaned) < original_len:
            applied_steps.append("strip_conversational_fluff")

        # Step 2: Replace wordy filler phrases
        original_len = len(cleaned)
        for pattern, replacement in WORDY_REPLACEMENTS:
            cleaned = re.sub(pattern, replacement, cleaned)
        if len(cleaned) < original_len:
            applied_steps.append("condense_wordy_phrases")

        # Step 3: Whitespace and punctuation normalization
        # Normalize multiple spaces (preserving newlines)
        cleaned = re.sub(r"[ \t]+", " ", cleaned)
        # Collapse 3 or more newlines to 2
        cleaned = re.sub(r"\n\s*\n\s*\n+", "\n\n", cleaned)
        # Strip space before punctuation
        cleaned = re.sub(r"\s+([,.:;!?])", r"\1", cleaned)
        # Normalize repeated punctuation (e.g. '???', '!!!')
        cleaned = re.sub(r"!{2,}", "!", cleaned)
        cleaned = re.sub(r"\?{2,}", "?", cleaned)
        applied_steps.append("normalize_whitespace_and_punctuation")

        # Step 4: Aggressive extractive pruning if requested
        if level == CompressionLevel.AGGRESSIVE and len(cleaned.split()) > 40:
            before_prune = cleaned
            cleaned = _extractive_sentence_prune(cleaned, target_ratio=0.75)
            if cleaned != before_prune:
                applied_steps.append("extractive_density_prune")

        # Ensure sentence beginnings are capitalized cleanly if leading fluff was stripped
        lines = cleaned.splitlines()
        capped_lines = []
        for line in lines:
            line = line.strip()
            if line and line[0].islower():
                capped_lines.append(line[0].upper() + line[1:])
            else:
                capped_lines.append(line)
        cleaned = "\n".join(capped_lines)

        return cleaned.strip(), applied_steps
