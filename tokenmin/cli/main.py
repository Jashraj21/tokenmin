"""
Command Line Interface for TokenMin with first-class support for latest AI models.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional

from tokenmin.metrics import display_rich_report, format_summary_table
from tokenmin.models import CompressionLevel, ContentType
from tokenmin.models_registry import LATEST_MODELS, resolve_model
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
        reasoning_mode=args.reasoning,
        cache_friendly=args.cache_friendly,
    )

    if args.output:
        Path(args.output).write_text(result.text, encoding="utf-8")
        sys.stderr.write(f"Wrote minimized output to {args.output}\n")
    elif not args.stats and not args.diff:
        sys.stdout.write(result.text)
        if not result.text.endswith("\n"):
            sys.stdout.write("\n")

    if args.diff:
        sys.stderr.write("\n=== Diff (Original vs Minimized) ===\n")
        sys.stderr.write(result.diff())
        sys.stderr.write("\n")

    if args.stats or (not args.output and sys.stdout.isatty() and not args.diff):
        display_rich_report(result)


def handle_stats(args: argparse.Namespace) -> None:
    """Handle `tokenmin stats` command with deep multi-model context window analysis."""
    text = _read_input(args.file)
    char_count = len(text)
    word_count = len(text.split())
    line_count = len(text.splitlines())

    try:
        from rich.console import Console
        from rich.table import Table

        console = Console()
        console.print(f"\n[bold cyan]📊 Content Analysis for: [white]{args.file}[/white][/bold cyan]")
        console.print(f"[dim]Lines: {line_count:,} | Words: {word_count:,} | Characters: {char_count:,}[/dim]\n")

        table = Table(title="⚡ Token Count & Context Utilization Across Latest Models", title_style="bold green")
        table.add_column("Model", style="cyan")
        table.add_column("Provider", style="dim")
        table.add_column("Calibrated Tokens", style="bold yellow")
        table.add_column("Context Window", style="white")
        table.add_column("Context Util %", style="green")
        table.add_column("Type", style="magenta")

        for key, info in LATEST_MODELS.items():
            t_count = Tokenizer.count(text, model=key)
            util_pct = (t_count / info.context_window) * 100.0
            m_type = "🧠 Reasoning" if info.is_reasoning_model else "Standard"

            table.add_row(
                info.display_name,
                info.provider,
                f"{t_count:,}",
                f"{info.context_window:,}",
                f"{util_pct:.3f}%",
                m_type
            )
        console.print(table)
    except ImportError:
        print(f"File: {args.file} (Lines: {line_count}, Words: {word_count}, Chars: {char_count})")
        for key, info in LATEST_MODELS.items():
            t_count = Tokenizer.count(text, model=key)
            print(f" • {info.display_name:<24}: {t_count} tokens ({info.context_window} context limit)")


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
        res = minimize(
            text,
            content_type=args.mode,
            level=lvl,
            model=args.model,
            skeleton=args.skeleton,
            reasoning_mode=args.reasoning,
        )
        results.append((lvl.value, res))

    model_info = resolve_model(args.model)
    try:
        from rich.console import Console
        from rich.table import Table

        console = Console()
        table = Table(title=f"🚀 Multi-Level Benchmark for [{model_info.display_name}]: {args.file}", title_style="bold yellow")
        table.add_column("Level", style="bold")
        table.add_column("Original Tokens", style="dim")
        table.add_column("Minimized Tokens", style="cyan")
        table.add_column("Reduction %", style="bold green")
        table.add_column("Time (ms)", style="magenta")

        for lvl_name, res in results:
            m = res.metrics
            table.add_row(
                lvl_name.capitalize(),
                f"{m.original_tokens:,}",
                f"{m.minimized_tokens:,}",
                f"{m.reduction_percentage}%",
                f"{m.elapsed_ms:.2f} ms"
            )
        console.print(table)
    except ImportError:
        print(f"Benchmark results for {args.file} [{model_info.display_name}]:")
        for lvl_name, res in results:
            m = res.metrics
            print(f"[{lvl_name.upper()}] Tokens: {m.original_tokens} -> {m.minimized_tokens} ({m.reduction_percentage}% saved) in {m.elapsed_ms:.2f}ms")


def handle_models(args: argparse.Namespace) -> None:
    """Handle `tokenmin models` command to list all supported latest frontier models."""
    try:
        from rich.console import Console
        from rich.table import Table

        console = Console()
        table = Table(title="🤖 Supported Latest AI Models & Specifications", title_style="bold cyan")
        table.add_column("Identifier", style="bold yellow")
        table.add_column("Display Name", style="white")
        table.add_column("Provider", style="dim")
        table.add_column("Context Window", style="green")
        table.add_column("Standard Rate (1M)", style="cyan")
        table.add_column("Cached Rate (1M)", style="dim")
        table.add_column("Reasoning", style="magenta")

        for key, info in LATEST_MODELS.items():
            table.add_row(
                info.name,
                info.display_name,
                info.provider,
                f"{info.context_window:,}",
                f"${info.input_price_per_m:.2f}",
                f"${info.cached_input_price_per_m:.2f}",
                "Yes" if info.is_reasoning_model else "No"
            )
        console.print(table)
    except ImportError:
        for key, info in LATEST_MODELS.items():
            print(f"{info.name:<20} | {info.display_name:<24} | {info.provider:<10} | {info.context_window:,} ctx")


def main() -> None:
    """Main CLI entrypoint."""
    parser = argparse.ArgumentParser(
        prog="tokenmin",
        description="⚡ TokenMin: High-performance token minimizer and context optimizer for latest AI models."
    )

    subparsers = parser.add_subparsers(dest="command", help="Available subcommands")

    # Subcommand: minimize
    p_min = subparsers.add_parser("minimize", aliases=["min"], help="Minimize tokens in a file or stdin")
    p_min.add_argument("file", help="Path to input file, or '-' for stdin")
    p_min.add_argument("-o", "--output", help="Output file path (default: stdout)")
    p_min.add_argument("-m", "--mode", default="auto", choices=["auto", "prompt", "reasoning", "cache", "text", "python", "code", "json", "log", "chat", "markdown"], help="Content type mode")
    p_min.add_argument("-l", "--level", default="moderate", choices=["conservative", "moderate", "aggressive"], help="Compression aggressiveness")
    p_min.add_argument("-b", "--budget", type=int, default=None, help="Target token budget cap")
    p_min.add_argument("--skeleton", action="store_true", help="Extract interface skeleton only (for code)")
    p_min.add_argument("--reasoning", action="store_true", help="Optimize prompt structure for reasoning models (o1, o3-mini, DeepSeek-R1)")
    p_min.add_argument("--cache-friendly", action="store_true", help="Optimize prompt prefix for prompt caching (Claude 3.5, Gemini, GPT-4o)")
    p_min.add_argument("--diff", action="store_true", help="Show unified diff")
    p_min.add_argument("--stats", action="store_true", help="Show token metrics report")
    p_min.add_argument("--model", default="gpt-4o", help="Target LLM model (e.g. o3-mini, claude-3-5-sonnet, gemini-2.0-flash, deepseek-v3)")
    p_min.set_defaults(func=handle_minimize)

    # Subcommand: stats
    p_stats = subparsers.add_parser("stats", help="Analyze token counts & context usage across latest LLMs")
    p_stats.add_argument("file", help="Path to input file, or '-' for stdin")
    p_stats.set_defaults(func=handle_stats)

    # Subcommand: bench
    p_bench = subparsers.add_parser("bench", aliases=["benchmark"], help="Benchmark all compression levels on a file")
    p_bench.add_argument("file", help="Path to input file, or '-' for stdin")
    p_bench.add_argument("-m", "--mode", default="auto", help="Content type mode")
    p_bench.add_argument("--skeleton", action="store_true", help="Extract interface skeleton only")
    p_bench.add_argument("--reasoning", action="store_true", help="Optimize prompt structure for reasoning models")
    p_bench.add_argument("--model", default="gpt-4o", help="Target LLM model")
    p_bench.set_defaults(func=handle_benchmark)

    # Subcommand: models
    p_models = subparsers.add_parser("models", help="List all supported latest frontier models & specs")
    p_models.set_defaults(func=handle_models)

    # Default to minimize if user calls `tokenmin <file>`
    if len(sys.argv) > 1 and sys.argv[1] not in ("minimize", "min", "stats", "bench", "benchmark", "models", "-h", "--help"):
        sys.argv.insert(1, "minimize")

    args = parser.parse_args()
    if hasattr(args, "func"):
        args.func(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
