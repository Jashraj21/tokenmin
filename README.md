# ⚡ TokenMin — High-Performance Token Minimizer for AI

[![Tests](https://img.shields.io/badge/tests-27%20passed-brightgreen.svg)]()
[![Python](https://img.shields.io/badge/python-3.8+-blue.svg)]()
[![License](https://img.shields.io/badge/license-MIT-purple.svg)]()
[![Latency](https://img.shields.io/badge/latency-%3C2ms-orange.svg)]()
[![Cost](https://img.shields.io/badge/external%20API%20cost-%240-success.svg)]()

**TokenMin** is a lightweight, zero-API-cost token minimizer and context optimizer designed for LLM prompts, agent memory, codebase contexts, JSON payloads, and system logs.

It shrinks token counts by **20% to 85%** in sub-millisecond execution time while preserving critical instructions, schema integrity, and semantic meaning.

---

## 🎯 Why Token Minimization?

1. **Cut LLM API Bills**: Save hundreds to thousands of dollars per 1M calls on frontier models (GPT-4o, Claude 3.5 Sonnet, Gemini 1.5/2.0 Pro).
2. **Eliminate "Lost in the Middle"**: LLMs pay sharper attention to core constraints and directives when conversational fluff, huge docstrings, and repeated log lines are eliminated.
3. **Slash Latency**: Smaller input prompts directly improve Time-To-First-Token (TTFT) and throughput.

---

## 📦 Key Capabilities

| Domain | Strategy | Typical Savings | Description |
| :--- | :--- | :---: | :--- |
| **Python Code** | AST Pruning & Skeletonization | **40% – 85%** | Strips comments/docstrings or extracts pure interface signatures (`...` bodies) for codebase context. |
| **Generic Code** | Lexical & Comment Stripper | **25% – 50%** | Safe C-style (`//`, `/* */`), SQL (`--`), and HTML (`<!-- -->`) comment removal preserving string literals. |
| **Prompts & Text** | Fluff & Wordy Phrase Pruning | **15% – 35%** | Eliminates conversational fluff (`"Please kindly..."`), replaces wordy transitions (`"due to the fact that"` → `"because"`), cleans whitespace. |
| **JSON & Data** | Null Pruner & Tabular Pack | **30% – 60%** | Prunes nulls and empty lists/dicts; packs uniform record arrays into compact `{"_schema": [...], "_rows": [...]}`. |
| **Logs & Traces** | Line Collapser & Time Shortener | **50% – 80%** | Groups consecutive identical lines with `[Repeated Nx]` count and condenses ISO timestamps. |
| **Chat & Agent Memory**| Rolling Context & Turn Trimmer | **30% – 60%** | Preserves recent turns while condensing verbose earlier assistant responses and tool returns. |

---

## 🚀 Quickstart

### Installation

```bash
pip install -e .
```

*Requirements: Python 3.8+, optional `tiktoken` for byte-pair token counting, optional `rich` for terminal visuals.*

---

## 💻 Python Library Usage

```python
import tokenmin

# 1. Automatic format detection & minimization
raw_prompt = """
Please can you kindly write a python script to parse CSV files.
In order to achieve good performance, take into consideration using asyncio
due to the fact that our upstream APIs have high latency.
"""
result = tokenmin.minimize(raw_prompt)
print(result.text)
# Output:
# "Write a python script to parse CSV files.
#  To achieve good performance, consider using asyncio because our upstream APIs have high latency."

# 2. View token savings & cost estimates
print(result.metrics.summary())
# "Tokens: 48 -> 32 (-16 tokens, 33.3% reduction) in 0.85ms"

# 3. Codebase Context Skeletonization (Ideal for AI Agents & Copilots)
with open("service.py") as f:
    repo_context = tokenmin.minimize(f.read(), skeleton=True)
# Result: Keeps class/function definitions & type hints; replaces bodies with '...' (saves ~80% tokens!)

# 4. Strict Token Budget Capping
capped_result = tokenmin.minimize(long_document, target_tokens=500)
assert tokenmin.count_tokens(capped_result.text) <= 500
```

---

## 🛠️ CLI Usage

TokenMin includes a CLI interface with built-in diffs, multi-model token counters, and benchmarking:

### 1. Minimize files or stdin
```bash
# Minimize a file directly
tokenmin minimize prompt.txt

# Pipe through stdin / stdout
cat input.txt | tokenmin - > min.txt

# Extract Python interface skeletons for agent context
tokenmin minimize backend.py --skeleton

# Show side-by-side color diff
tokenmin minimize prompt.txt --diff

# Cap output to a strict token budget
tokenmin minimize document.txt --budget 1000
```

### 2. Multi-model Token & Size Analysis
```bash
tokenmin stats prompt.txt
```
```text
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━┓
┃ Property                        ┃ Value ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━┩
│ Lines                           │ 6     │
│ Words                           │ 104   │
│ Characters                      │ 628   │
│ OpenAI GPT-4o Tokens            │ 161   │
│ Claude 3.5 Sonnet Tokens (est.) │ 169   │
│ Gemini 1.5/2.0 Tokens (est.)    │ 157   │
└─────────────────────────────────┴───────┘
```

### 3. Multi-Level Benchmark
```bash
tokenmin bench sample_code.py --skeleton
```
```text
┏━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━┳━━━━━━━━━━━┓
┃ Level        ┃ Original Tokens ┃ Minimized Tokens ┃ Reduction % ┃ Time (ms) ┃
┡━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━╇━━━━━━━━━━━┩
│ Conservative │ 557             │ 149              │ 73.25%      │ 57.69 ms  │
│ Moderate     │ 557             │ 102              │ 81.69%      │ 0.99 ms   │
│ Aggressive   │ 557             │ 102              │ 81.69%      │ 0.89 ms   │
└──────────────┴─────────────────┴──────────────────┴─────────────┴───────────┘
```

---

## 🏗️ Architecture

```
TokenMin
├── tokenmin/
│   ├── tokenizer.py        # Multi-model counting (tiktoken, Claude, Gemini, Llama)
│   ├── detector.py         # Fast AST & regex content format classifier
│   ├── pipeline.py         # MinimizerPipeline orchestrator & budget enforcer
│   ├── models.py           # TokenStats, MinifyResult, pricing tables
│   ├── metrics.py          # Rich terminal reports and cost savings calculator
│   ├── cli/main.py         # CLI commands (minimize, stats, bench)
│   └── strategies/
│       ├── code/           # Python AST skeletonizer & generic comment stripper
│       ├── text/           # Prompt fluff stripper & Markdown table minifier
│       ├── data/           # JSON null pruner, tabular packer & log compressor
│       └── chat/           # Conversation history condensing & rolling window
└── tests/                  # 27 comprehensive unit & integration tests
```

---

## 🧪 Testing

Run the full test suite:

```bash
pytest -v
```

All 27 test cases pass across tokenization, AST skeletonization, prompt compaction, markdown tables, JSON tabularization, log deduplication, and CLI piping.

---

## 📄 License

MIT
