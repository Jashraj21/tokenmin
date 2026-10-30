"""Tests for central minimizer pipeline, auto-detection, and budget capping."""
from tokenmin.detector import detect_content_type
from tokenmin.models import ContentType
from tokenmin.pipeline import minimize


def test_auto_detection():
    # Python detection
    py_code = "import os\ndef test_fn():\n    return os.getcwd()"
    assert detect_content_type(py_code) == ContentType.PYTHON

    # JSON detection
    json_data = '{"name": "test", "active": true}'
    assert detect_content_type(json_data) == ContentType.JSON

    # Markdown detection
    md = "# Title\n\n| Col 1 | Col 2 |\n|---|---|\n| A | B |\n\n- item 1\n- item 2"
    assert detect_content_type(md) == ContentType.MARKDOWN

    # Log detection
    logs = (
        "2026-10-30 INFO Starting\n"
        "2026-10-30 DEBUG Initializing subsystem\n"
        "2026-10-30 ERROR Timeout connecting to port 5432"
    )
    assert detect_content_type(logs) == ContentType.LOG


def test_pipeline_target_budget_cap():
    long_text = "This is a detailed sentence. " * 50
    result = minimize(long_text, target_tokens=20)

    assert result.metrics.minimized_tokens <= 20
    assert "budget_cap_20_tokens" in result.applied_steps


def test_pipeline_diff_generation():
    text = "Please kindly check the code in order to fix bugs."
    result = minimize(text)

    diff_str = result.diff()
    assert isinstance(diff_str, str)
