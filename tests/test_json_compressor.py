"""Tests for JSON and structured data compressor."""
import json

from tokenmin.models import CompressionLevel
from tokenmin.strategies.data import JsonCompressor


def test_json_whitespace_and_null_pruning():
    data = {
        "id": 101,
        "name": "Project Alpha",
        "empty_list": [],
        "null_field": None,
        "nested": {
            "valid": True,
            "useless": None,
            "empty_dict": {}
        }
    }
    raw_json = json.dumps(data, indent=4)
    compressor = JsonCompressor()
    result = compressor.compress(raw_json, level=CompressionLevel.MODERATE)

    assert result.metrics.saved_tokens > 0
    # Null and empty fields pruned
    parsed = json.loads(result.text)
    assert "null_field" not in parsed
    assert "empty_list" not in parsed
    assert parsed["nested"] == {"valid": True}


def test_json_uniform_records_to_tabular():
    records = [
        {"id": 1, "name": "Alice", "role": "admin"},
        {"id": 2, "name": "Bob", "role": "user"},
        {"id": 3, "name": "Charlie", "role": "editor"},
        {"id": 4, "name": "David", "role": "viewer"}
    ]
    raw_json = json.dumps(records, indent=2)
    compressor = JsonCompressor()
    result = compressor.compress(raw_json, level=CompressionLevel.AGGRESSIVE)

    parsed = json.loads(result.text)
    assert "_schema" in parsed
    assert "_rows" in parsed
    assert parsed["_schema"] == ["id", "name", "role"]
    assert len(parsed["_rows"]) == 4
    assert result.metrics.reduction_percentage > 35.0
