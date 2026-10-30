"""
Automatic content type detector for incoming text.
"""
from __future__ import annotations

import ast
import json
import re

from tokenmin.models import ContentType


def detect_content_type(text: str) -> ContentType:
    """
    Detect the likely content type of the given text (JSON, Python, Code, Markdown, Log, Chat, Prompt).
    """
    stripped = text.strip()
    if not stripped:
        return ContentType.TEXT

    # 1. Check for JSON / Structured Data
    if (stripped.startswith("{") and stripped.endswith("}")) or (stripped.startswith("[") and stripped.endswith("]")):
        try:
            parsed = json.loads(stripped)
            # If it's a list of LLM chat messages: must have both 'role' and 'content'
            if isinstance(parsed, list) and parsed and isinstance(parsed[0], dict) and "role" in parsed[0] and "content" in parsed[0]:
                return ContentType.CHAT
            return ContentType.JSON
        except Exception:
            pass

    # 2. Check for Chat / Dialogue transcript
    chat_markers = re.findall(r"(?:^|\n)(?:User|Assistant|System|Human|AI):\s*", stripped, re.IGNORECASE)
    if len(chat_markers) >= 2:
        return ContentType.CHAT

    # 3. Check for Logs
    log_markers = re.findall(
        r"(?:^|\n)(?:\d{4}-\d{2}-\d{2}|\b(?:INFO|DEBUG|WARN|WARNING|ERROR|FATAL|TRACE)\b)",
        stripped
    )
    if len(log_markers) >= 3 and len(stripped.splitlines()) >= 3:
        return ContentType.LOG

    # 4. Check for Python
    python_keywords = len(re.findall(r"\b(?:def\s+\w+|class\s+\w+|import\s+\w+|from\s+\w+\s+import|elif\s+|lambda\s+)\b", stripped))
    if python_keywords >= 2:
        try:
            ast.parse(stripped)
            return ContentType.PYTHON
        except SyntaxError:
            return ContentType.PYTHON

    # 5. Check for Generic Code (JS/TS, Go, Rust, Java, C++)
    code_keywords = len(re.findall(r"\b(?:function\s+|const\s+|let\s+|var\s+|interface\s+|type\s+\w+\s*=|pub\s+fn|package\s+\w+|#include|public\s+class)\b", stripped))
    if code_keywords >= 2 or re.search(r"[{};]\s*$", stripped, re.MULTILINE):
        if code_keywords >= 2:
            return ContentType.CODE

    # 6. Check for Markdown
    md_elements = 0
    if re.search(r"^#{1,6}\s+\w+", stripped, re.MULTILINE):
        md_elements += 1
    if re.search(r"^\|.*\|", stripped, re.MULTILINE):
        md_elements += 1
    if re.search(r"\[.+?\]\(.+?\)", stripped):
        md_elements += 1
    if re.search(r"^[-*+]\s+\w+", stripped, re.MULTILINE):
        md_elements += 1

    if md_elements >= 2:
        return ContentType.MARKDOWN

    # Default to prompt / natural language text
    return ContentType.PROMPT
