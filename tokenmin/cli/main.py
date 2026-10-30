"""
Command Line Interface for TokenMin.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional

from tokenmin.metrics import display_rich_report, format_summary_table
from tokenmin.models import CompressionLevel, ContentType
from tokenmin.pipeline import minimize
from tokenmin.tokenizer import Tokenizer, count_tokens


def _read_input(path_or_dash: str) -> str:
    """Read input text from a file or standard input."""
    if path_or_dash == "-":
        return sys.stdin.read()
    path = Path(path_or_dash)
    if not path.is_file():
        sys.stderr.write(f"Error: File not found: {path_or_dash}\n")
        sys.exit(1)
    return path.read_text(encoding="utf-8")


def handle_minimize(args: argparse.Namespace) -> None:
    """Handle `tokenmin minimize` command."""
    text = _read_input(args.file)

    result = minimize(
        text,
        content_type=args.mode,
        level=args.level,
        target_tokens=args.budget,
        model=args.model,
        skeleton=args.skeleton,
    )

    # Output minimized text
    if args.output:
        Path(args.output).write_text(result.text, encoding="utf-8")
        sys.stderr.write(f"Wrote minimized output to {args.output}\n")
    elif not args.stats and not args.diff:
        sys.stdout.write(result.text)
        if not result.text.endswith("\n"):
            sys.stdout.write("\n")

    # If diff requested
    if args.diff:
        sys.stderr.write("\n=== Diff (Original vs Minimized) ===\n")
        sys.stderr.write(result.diff())
        sys.stderr.write("\n")

    # If stats requested
    if args.stats or (not args.output and sys.stdout.isatty() and not args.diff):
        display_rich_report(result)


def handle_stats(args: argparse.Namespace) -> None:
    """Handle `tokenmin stats` command."""
    text = _read_input(args.file)

    gpt4o_tokens = Tokenizer.count(text, model="gpt-4o")
    claude_tokens = Tokenizer.count(text, model="claude-3-5-sonnet")
    gemini_tokens = Tokenizer.count(text, model="gemini-1.5-pro")
    char_count = len(text)
    word_count = len(text.split())
    line_count = len(text.splitlines())

    try:
        from rich.console import Console
        from rich.table import Table

        console = Console()
        table = Table(title=f"📊 Token & Size Analysis: {args.file}", title_style="bold cyan")
        table.add_column("Property", style="bold")
        table.add_column("Value", style="green")

        table.add_row("Lines", f"{line_count:,}")
        table.add_row("Words", f"{word_count:,}")
        table.add_row("Characters", f"{char_count:,}")
        table.add_row("OpenAI GPT-4o Tokens", f"{gpt4o_tokens:,}")
        table.add_row("Claude 3.5 Sonnet Tokens (est.)", f"{claude_tokens:,}")
        table.add_row("Gemini 1.5/2.0 Tokens (est.)", f"{gemini_tokens:,}")
        console.print(table)
    except ImportError:
        print(f"File: {args.file}")
        print(f"Lines: {line_count} | Words: {word_count} | Chars: {char_count}")
        print(f"GPT-4o Tokens: {gpt4o_tokens}")
        print(f"Claude 3.5 Tokens: {claude_tokens}")
        print(f"Gemini Tokens: {gemini_tokens}")


def handle_benchmark(args: argparse.Namespace) -> None:
    """Handle `tokenmin bench` command to evaluate conservative, moderate, aggressive."""
    text = _read_input(args.file)

    levels = [
        CompressionLevel.CONSERVATIVE,
        CompressionLevel.MODERATE,
        CompressionLevel.AGGRESSIVE,
    ]

    results = []
    for lvl in levels:
        res = minimize(text, content_type=args.mode, level=lvl, model=args.model, skeleton=args.skeleton)
        results.append((lvl.value, res))

    try:
        from rich.console import Console
        from rich.table import Table

        console = Console()
        table = Table(title=f"🚀 Multi-Level Benchmark: {args.file}", title_style="bold yellow")
        table.add_column("Level", style="bold")
        table.add_column("Original Tokens", style="dim")
        table.add_column("Minimized Tokens", style="cyan")
        table.add_column("Reduction %", style="bold green")
        table.add_column("Time (ms)", style="magenta")

        for lvl_name, res in results:
            m = res.metrics
            table.add_row(
                lvl_name.capitalize(),
                str(m.original_tokens),
                str(m.minimized_tokens),
                f"{m.reduction_percentage}%",
                f"{m.elapsed_ms:.2f} ms"
            )
        console.print(table)
    except ImportError:
        print(f"Benchmark results for {args.file}:")
        for lvl_name, res in results:
            m = res.metrics
            print(f"[{lvl_name.upper()}] Tokens: {m.original_tokens} -> {m.minimized_tokens} ({m.reduction_percentage}% saved) in {m.elapsed_ms:.2f}ms")


def main() -> None:
    """Main CLI entrypoint."""
    parser = argparse.ArgumentParser(
        prog="tokenmin",
        description="⚡ TokenMin: High-performance token minimizer and context optimizer for AI prompts and code."
    )

    subparsers = parser.add_subparsers(dest="command", help="Available subcommands")

    # Subcommand: minimize
    p_min = subparsers.add_parser("minimize", aliases=["min"], help="Minimize tokens in a file or stdin")
    p_min.add_argument("file", help="Path to input file, or '-' for stdin")
    p_min.add_argument("-o", "--output", help="Output file path (default: stdout)")
    p_min.add_argument("-m", "--mode", default="auto", choices=["auto", "prompt", "text", "python", "code", "json", "log", "chat", "markdown"], help="Content type mode")
    p_min.add_argument("-l", "--level", default="moderate", choices=["conservative", "moderate", "aggressive"], help="Compression aggressiveness")
    p_min.add_argument("-b", "--budget", type=int, default=None, help="Target token budget cap")
    p_min.add_argument("--skeleton", action="store_true", help="Extract interface skeleton only (for code)")
    p_min.add_argument("--diff", action="store_true", help="Show unified diff")
    p_min.add_argument("--stats", action="store_true", help="Show token metrics report")
    p_min.add_argument("--model", default="gpt-4o", help="Target LLM tokenizer model")
    p_min.set_defaults(func=handle_minimize)

    # Subcommand: stats
    p_stats = subparsers.add_parser("stats", help="Analyze token counts across multiple LLM tokenizers")
    p_stats.add_argument("file", help="Path to input file, or '-' for stdin")
    p_stats.set_defaults(func=handle_stats)

    # Subcommand: bench
    p_bench = subparsers.add_parser("bench", aliases=["benchmark"], help="Benchmark all compression levels on a file")
    p_bench.add_argument("file", help="Path to input file, or '-' for stdin")
    p_bench.add_argument("-m", "--mode", default="auto", help="Content type mode")
    p_bench.add_argument("--skeleton", action="store_true", help="Extract interface skeleton only")
    p_bench.add_argument("--model", default="gpt-4o", help="Target LLM tokenizer model")
    p_bench.set_defaults(func=handle_benchmark)

    # Default to minimize if user calls `tokenmin <file>`
    if len(sys.argv) > 1 and sys.argv[1] not in ("minimize", "min", "stats", "bench", "benchmark", "-h", "--help"):
        sys.argv.insert(1, "minimize")

    args = parser.parse_args()
    if hasattr(args, "func"):
        args.func(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
