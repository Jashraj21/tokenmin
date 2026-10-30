"""
Metrics reporting, terminal visualization, and cost analysis for latest AI models.
"""
from __future__ import annotations

from typing import Dict, Optional

from tokenmin.models import MinifyResult
from tokenmin.models_registry import LATEST_MODELS, resolve_model
from tokenmin.tokenizer import Tokenizer

try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table
    _RICH_AVAILABLE = True
except ImportError:
    _RICH_AVAILABLE = False


def format_summary_table(result: MinifyResult) -> str:
    """Generate a clean ASCII table comparing original and minimized stats across latest models."""
    m = result.metrics
    target_info = resolve_model(m.model)

    lines = [
        "╔══════════════════════════════════════════════════════════════════════╗",
        "║                     TOKENMIN OPTIMIZATION REPORT                     ║",
        "╠══════════════════════════════════════════════════════════════════════╣",
        f"║  Target Model:     {target_info.display_name:<48}║",
        f"║  Strategy:         {result.strategy_used:<48}║",
        f"║  Execution Time:   {m.elapsed_ms:.2f} ms{'':<41}║",
        "╠══════════════════════════════════════════════════════════════════════╣",
        f"║  Original Tokens:  {m.original_tokens:<48}║",
        f"║  Minimized Tokens: {m.minimized_tokens:<48}║",
        f"║  Tokens Saved:     {m.saved_tokens:<48}║",
        f"║  Reduction:        {m.reduction_percentage:.1f}%{'':<44}║",
        f"║  Context Window:   {target_info.context_window:,} tokens (using {m.context_usage_percentage():.3f}%){'':<14}║",
        "╠══════════════════════════════════════════════════════════════════════╣",
        "║  Estimated Savings per 1M Requests:                                  ║",
    ]

    savings = m.estimated_savings(1_000_000)
    for model_name, saved_usd in list(savings.items())[:8]:
        info = LATEST_MODELS[model_name]
        line = f"║    • {info.display_name:<24} : ${saved_usd:,.2f} USD{'':<20}║"
        pad = 70 - len(line)
        if pad > 0:
            line = line[:-1] + " " * pad + "║"
        lines.append(line)

    lines.append("╚══════════════════════════════════════════════════════════════════════╝")
    return "\n".join(lines)


def display_rich_report(result: MinifyResult) -> None:
    """Print an interactive Rich table displaying savings across latest models."""
    if not _RICH_AVAILABLE:
        print(format_summary_table(result))
        return

    console = Console()
    m = result.metrics
    target_info = resolve_model(m.model)

    # 1. Main Compression Overview
    table = Table(title=f"⚡ TokenMin Optimization Report [{target_info.display_name}]", title_style="bold cyan")
    table.add_column("Metric", style="bold")
    table.add_column("Original", style="red")
    table.add_column("Minimized", style="green")
    table.add_column("Saved", style="bold yellow")

    table.add_row(
        "Tokens",
        f"{m.original_tokens:,}",
        f"{m.minimized_tokens:,}",
        f"-{m.saved_tokens:,} ({m.reduction_percentage}%)"
    )
    table.add_row(
        "Characters",
        f"{m.original_chars:,}",
        f"{m.minimized_chars:,}",
        f"-{m.original_chars - m.minimized_chars:,} ({m.char_reduction_percentage}%)"
    )

    steps_str = ", ".join(result.applied_steps) if result.applied_steps else "None"
    console.print(table)
    console.print(
        f"[dim]Strategy: [cyan]{result.strategy_used}[/cyan] | "
        f"Context Window: [cyan]{target_info.context_window:,}[/cyan] (Used: [green]{m.context_usage_percentage():.3f}%[/green]) | "
        f"Latency: [cyan]{m.elapsed_ms:.2f}ms[/cyan] | Steps: {steps_str}[/dim]"
    )

    # 2. Multi-Model Savings Table for Latest Models
    cost_table = Table(title="💰 Savings Matrix for Latest Frontier Models (per 1M Calls)", title_style="bold green")
    cost_table.add_column("Model", style="cyan")
    cost_table.add_column("Provider", style="dim")
    cost_table.add_column("Standard Rate", style="white")
    cost_table.add_column("Cached Rate", style="dim")
    cost_table.add_column("Standard Savings", style="bold green")
    cost_table.add_column("Cached Savings", style="green")

    # Display key latest models
    selected_keys = [
        "gpt-4o", "o3-mini", "o1",
        "claude-3-5-sonnet", "claude-3-5-haiku",
        "gemini-2.0-flash", "gemini-1.5-pro",
        "deepseek-v3", "deepseek-r1",
        "llama-3.3-70b"
    ]

    for key in selected_keys:
        if key not in LATEST_MODELS:
            continue
        info = LATEST_MODELS[key]
        calibrated_saved = m.saved_tokens * info.tokenizer_calibration
        std_dollars = (calibrated_saved / 1_000_000.0) * info.input_price_per_m * 1_000_000
        cached_dollars = (calibrated_saved / 1_000_000.0) * info.cached_input_price_per_m * 1_000_000

        cost_table.add_row(
            info.display_name,
            info.provider,
            f"${info.input_price_per_m:.2f}",
            f"${info.cached_input_price_per_m:.2f}",
            f"${std_dollars:,.2f}",
            f"${cached_dollars:,.2f}",
        )

    console.print(cost_table)
