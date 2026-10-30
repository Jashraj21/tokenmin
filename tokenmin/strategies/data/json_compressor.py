"""
JSON and structured data compressor.
Minifies whitespace, prunes empty/null values, rounds excess float precision,
and packs uniform object arrays into schema-tabular form.
"""
from __future__ import annotations

import json
from typing import Any, Dict, List, Optional, Tuple

from tokenmin.models import CompressionLevel
from tokenmin.strategies.base import BaseCompressor


def _prune_empty_and_null(data: Any, prune_empty_strings: bool = False) -> Any:
    """Recursively prune None/null and empty lists/dicts."""
    if isinstance(data, dict):
        cleaned_dict = {}
        for k, v in data.items():
            cleaned_val = _prune_empty_and_null(v, prune_empty_strings)
            if cleaned_val is None:
                continue
            if isinstance(cleaned_val, (dict, list)) and len(cleaned_val) == 0:
                continue
            if prune_empty_strings and isinstance(cleaned_val, str) and cleaned_val == "":
                continue
            cleaned_dict[k] = cleaned_val
        return cleaned_dict
    elif isinstance(data, list):
        cleaned_list = []
        for item in data:
            cleaned_item = _prune_empty_and_null(item, prune_empty_strings)
            if cleaned_item is None:
                continue
            if isinstance(cleaned_item, (dict, list)) and len(cleaned_item) == 0:
                continue
            if prune_empty_strings and isinstance(cleaned_item, str) and cleaned_item == "":
                continue
            cleaned_list.append(cleaned_item)
        return cleaned_list
    elif isinstance(data, float):
        # Round float precision to 4 decimals
        return round(data, 4)
    return data


def _is_uniform_record_list(data: Any) -> bool:
    """Check if data is a list of dicts sharing the same keys."""
    if not isinstance(data, list) or len(data) < 3:
        return False
    first = data[0]
    if not isinstance(first, dict) or not first:
        return False
    keys = list(first.keys())
    # Ensure all items are dicts with the same keys
    for item in data[1:]:
        if not isinstance(item, dict) or list(item.keys()) != keys:
            return False
    return True


def _convert_to_schema_tabular(data: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Convert list of uniform dicts to {_schema: [...], _rows: [[...], ...]}."""
    keys = list(data[0].keys())
    rows = [[item.get(k) for k in keys] for item in data]
    return {"_schema": keys, "_rows": rows}


class JsonCompressor(BaseCompressor):
    """Compresses JSON payloads for LLM context optimization."""

    name = "json"
    description = "Minifies JSON by stripping whitespace, pruning null/empty fields, and tabularizing uniform lists"

    def run_compression(
        self,
        text: str,
        level: CompressionLevel = CompressionLevel.MODERATE,
        tabularize: Optional[bool] = None,
        prune_nulls: bool = True,
        **kwargs: Any
    ) -> Tuple[str, List[str]]:
        applied_steps: List[str] = []

        try:
            parsed = json.loads(text.strip())
        except Exception:
            # If not valid JSON, perform safe whitespace minification
            minified = " ".join(text.split())
            applied_steps.append("whitespace_minify_fallback")
            return minified, applied_steps

        # 1. Prune null and empty values
        if prune_nulls and level in (CompressionLevel.MODERATE, CompressionLevel.AGGRESSIVE):
            prune_empty_str = (level == CompressionLevel.AGGRESSIVE)
            parsed = _prune_empty_and_null(parsed, prune_empty_strings=prune_empty_str)
            applied_steps.append("prune_null_and_empty_fields")

        # 2. Check for uniform record list to tabularize
        should_tabularize = tabularize if tabularize is not None else (level == CompressionLevel.AGGRESSIVE)
        if should_tabularize:
            if _is_uniform_record_list(parsed):
                parsed = _convert_to_schema_tabular(parsed)
                applied_steps.append("convert_to_schema_tabular")
            elif isinstance(parsed, dict):
                # Check top-level values for uniform lists
                for k, v in list(parsed.items()):
                    if _is_uniform_record_list(v):
                        parsed[k] = _convert_to_schema_tabular(v)
                        applied_steps.append(f"convert_{k}_to_schema_tabular")

        # 3. Compact serialization
        minified = json.dumps(parsed, separators=(",", ":"), ensure_ascii=False)
        applied_steps.append("compact_json_serialization")

        return minified, applied_steps

