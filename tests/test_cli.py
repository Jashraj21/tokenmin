"""Tests for CLI entrypoint."""
import subprocess
import sys


def test_cli_help():
    result = subprocess.run(
        [sys.executable, "-m", "tokenmin.cli.main", "--help"],
        capture_output=True,
        text=True
    )
    assert result.returncode == 0
    assert "⚡ TokenMin" in result.stdout


def test_cli_minimize_stdin():
    input_text = "Please kindly optimize this prompt in order to reduce tokens."
    result = subprocess.run(
        [sys.executable, "-m", "tokenmin.cli.main", "minimize", "-"],
        input=input_text,
        capture_output=True,
        text=True
    )
    assert result.returncode == 0
    assert "Please kindly" not in result.stdout
    assert "reduce tokens" in result.stdout
