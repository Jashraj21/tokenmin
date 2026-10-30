"""
Metrics reporting, terminal visualization, and cost analysis.
"""
from __future__ import annotations

from typing import Dict, Optional

from tokenmin.models import MODEL_INPUT_PRICES_PER_M, MinifyResult
from tokenmin.tokenizer import Tokenizer

try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table
    _RICH_AVAILABLE = True
except ImportError:
    _RICH_AVAILABLE = False


def format_summary_table(result: MinifyResult) -> str:
    """Generate a clean ASCII or Rich table comparing original and minimized stats."""
    m = result.metrics
    orig_t = m.original_tokens
    min_t = m.minimized_tokens
    saved_t = m.saved_tokens
    pct = m.reduction_percentage

    lines = [
        "╔══════════════════════════════════════════════════════════╗",
        "║                TOKENMIN COMPRESSION REPORT               ║",
        "╠══════════════════════════════════════════════════════════╣",
        f"║  Strategy:         {result.strategy_used:<38}║",
        f"║  Content Type:     {result.content_type:<38}║",
        f"║  Execution Time:   {m.elapsed_ms:.2f} ms{'':<31}║",
        "╠══════════════════════════════════════════════════════════╣",
        f"║  Original Tokens:  {orig_t:<38}║",
        f"║  Minimized Tokens: {min_t:<38}║",
        f"║  Tokens Saved:     {saved_t:<38}║",
        f"║  Reduction:        {pct:.1f}%{'':<34}║",
        f"║  Original Chars:   {m.original_chars:<38}║",
        f"║  Minimized Chars:  {m.minimized_chars:<38}║",
        "╠══════════════════════════════════════════════════════════╣",
        "║  Estimated Savings per 1M Requests:                      ║",
    ]

    savings = m.estimated_savings(1_000_000)
    for model_name, saved_usd in savings.items():
        line = f"║    • {model_name:<18} : ${saved_usd:,.2f} USD{'':<18}║"
        # truncate or pad line to 60 chars width
        pad = 60 - len(line)
        if pad > 0:
            line = line[:-1] + " " * pad + "║"
        lines.append(line)

    lines.append("╚══════════════════════════════════════════════════════════╝")
    return "\n".join(lines)


def display_rich_report(result: MinifyResult) -> None:
    """Print an interactive Rich table if rich is installed."""
    if not _RICH_AVAILABLE:
        print(format_summary_table(result))
        return

    console = Console()
    m = result.metrics

    table = Table(title="⚡ TokenMin Optimization Report", title_style="bold cyan")
    table.add_column("Metric", style="bold")
    table.add_column("Original", style="red")
    table.add_column("Minimized", style="green")
    table.add_column("Saved", style="bold yellow")

    table.add_row(
        "Tokens",
        str(m.original_tokens),
        str(m.minimized_tokens),
        f"-{m.saved_tokens} ({m.reduction_percentage}%)"
    )
    table.add_row(
        "Characters",
        str(m.original_chars),
        str(m.minimized_chars),
        f"-{m.original_chars - m.minimized_chars} ({m.char_reduction_percentage}%)"
    )

    steps_str = ", ".join(result.applied_steps) if result.applied_steps else "None"
    console.print(table)
    console.print(f"[dim]Strategy: [cyan]{result.strategy_used}[/cyan] | Content: [cyan]{result.content_type}[/cyan] | Time: [cyan]{m.elapsed_ms:.2f}ms[/cyan] | Steps: {steps_str}[/dim]")

    # Cost table
    cost_table = Table(title="💰 Estimated Cost Savings per 1M Requests", title_style="bold green")
    cost_table.add_column("Model", style="cyan")
    cost_table.add_column("Standard Rate (per 1M)", style="dim")
    cost_table.add_column("Dollars Saved", style="bold green")

    for model_name, price in MODEL_INPUT_PRICES_PER_M.items():
        saved = (m.saved_tokens / 1_000_000.0) * price * 1_000_000
        cost_table.add_row(model_name, f"${price:.3f}", f"${saved:,.2f}")

    console.print(cost_table)
