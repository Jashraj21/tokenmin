"""Tests for log and console output compressor."""
from tokenmin.models import CompressionLevel
from tokenmin.strategies.data import LogCompressor


def test_log_repeated_lines_collapsing():
    logs = """2026-10-30 14:00:01 INFO Server started on port 8080
2026-10-30 14:00:02 WARN Database connection retry
2026-10-30 14:00:03 WARN Database connection retry
2026-10-30 14:00:04 WARN Database connection retry
2026-10-30 14:00:05 WARN Database connection retry
2026-10-30 14:00:06 INFO Database connected successfully
"""
    compressor = LogCompressor()
    result = compressor.compress(logs, level=CompressionLevel.MODERATE)

    assert result.metrics.saved_tokens > 0
    assert "[Repeated 4x]" in result.text
    assert "Database connected successfully" in result.text
