# ⚡ TokenMin — Token Minimizer for Modern AI Models

[![Tests](https://img.shields.io/badge/tests-32%20passed-brightgreen.svg)]()
[![Python](https://img.shields.io/badge/python-3.8+-blue.svg)]()
[![License](https://img.shields.io/badge/license-MIT-purple.svg)]()
[![Latency](https://img.shields.io/badge/latency-%3C1ms-orange.svg)]()
[![Models](https://img.shields.io/badge/models-o1%20%7C%20o3--mini%20%7C%20Claude%203.5%20%7C%20Gemini%202.0%20%7C%20DeepSeek--R1-purple.svg)]()

**TokenMin** is a high-performance token minimizer, context optimizer, and cost reducer built with first-class support for frontier AI models:

* **OpenAI**: `o3-mini`, `o1`, `gpt-4o`, `gpt-4o-mini`
* **Anthropic**: `claude-3-5-sonnet`, `claude-3-5-haiku`
* **Google**: `gemini-2.0-flash`, `gemini-2.0-flash-thinking`, `gemini-1.5-pro` (2M context)
* **DeepSeek**: `deepseek-r1` (reasoning), `deepseek-v3` (MoE)
* **Meta & Open Weights**: `llama-3.3-70b`, `llama-3.1-405b`, `qwen-2.5-coder-32b`

---

## 🎯 Modern Model Capabilities

### 1. 🧠 Reasoning Model Optimizer (`--reasoning` / `mode="reasoning"`)
For reasoning models like **OpenAI o1, o3-mini, and DeepSeek-R1**, prompt bloat doesn't just waste input tokens—it triggers **runaway internal thinking tokens** (charged at 3–4x output rates).
* Strips conversational ambiguity and emotional pleasantries.
* Removes redundant steering tags like `"think step by step"` or `"take your time"` that trigger repetitive thought exploration.
* Structures prompts into direct, declarative constraints so reasoning engines converge 2x–4x faster.

### 2. ⚡ Prompt Caching Optimizer (`--cache-friendly` / `mode="cache"`)
For **Claude 3.5 Sonnet, Gemini 2.0, DeepSeek V3, and GPT-4o**, prompt caching provides a **50% to 90% discount** on input tokens for reusable prefixes.
* Stabilizes prefixes by neutralizing dynamic timestamps and session UUIDs that cause cache-busting.
* Automatically verifies provider minimum token thresholds (e.g. 1,024 tokens for Claude/Gemini).
* Isolates dynamic user queries from cached context blocks.

### 3. 📊 Calibrated Multi-Model Tokenization & Context Checking
* Token counts calibrated across model tokenizers (`o200k_base`, SentencePiece 256k, and BPE).
* Real-time calculation of context window usage percentage (from 128k up to Gemini's 2,097,152 tokens).
* Cost savings matrix comparing standard rates vs prompt-cached rates across all providers.

---

## 🚀 Quickstart

```bash
pip install -e .
```

### Python Library Usage

```python
import tokenmin

# 1. Optimize prompt specifically for OpenAI o3-mini or DeepSeek-R1
prompt = "Please take your time and think step by step to solve this optimization problem..."
res = tokenmin.minimize(prompt, model="o3-mini", reasoning_mode=True)
print(res.text)

# 2. Extract Python interface skeletons for AI Copilots & Agent Workspaces
with open("backend.py") as f:
    repo_context = tokenmin.minimize(f.read(), skeleton=True, model="claude-3-5-sonnet")
# Saves 81% of tokens while preserving 100% of function signatures & type hints!

# 3. Optimize prompt for Prompt Caching (Claude 3.5 / Gemini)
cache_ready = tokenmin.minimize(agent_system_context, cache_friendly=True, model="claude-3-5-sonnet")

# 4. Accurate token counting calibrated for target model
gemini_tokens = tokenmin.count_tokens(text, model="gemini-2.0-flash")
claude_tokens = tokenmin.count_tokens(text, model="claude-3-5-sonnet")
o3_tokens     = tokenmin.count_tokens(text, model="o3-mini")
```

---

## 🛠️ CLI Workflows

```bash
# 1. List supported latest models & pricing
tokenmin models

# 2. Analyze token counts & context window utilization across all models
tokenmin stats prompt.txt

# 3. Minimize code with AST skeletonization for o3-mini
tokenmin minimize backend.py --skeleton --model o3-mini

# 4. Optimize reasoning prompt
tokenmin minimize reasoning_task.txt --reasoning --diff

# 5. Benchmark conservative vs moderate vs aggressive
tokenmin bench sample_code.py --model claude-3-5-sonnet --skeleton
```

---

## 🤖 Supported Model Profiles & Context Limits

| Model | Provider | Context Window | Standard Rate (1M) | Cached Rate (1M) | Type |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **OpenAI o1** | OpenAI | 200,000 | $15.00 | $7.50 | 🧠 Reasoning |
| **OpenAI o3-mini** | OpenAI | 200,000 | $1.10 | $0.55 | 🧠 Reasoning |
| **OpenAI GPT-4o** | OpenAI | 128,000 | $2.50 | $1.25 | Standard |
| **Claude 3.5 Sonnet** | Anthropic | 200,000 | $3.00 | $0.30 | Standard (90% cache discount) |
| **Claude 3.5 Haiku** | Anthropic | 200,000 | $0.80 | $0.08 | Standard |
| **Gemini 2.0 Flash** | Google | 1,048,576 | $0.10 | $0.025 | Standard |
| **Gemini 2.0 Thinking**| Google | 1,048,576 | $0.10 | $0.025 | 🧠 Reasoning |
| **Gemini 1.5 Pro** | Google | 2,097,152 | $1.25 | $0.31 | Standard (2M Window) |
| **DeepSeek R1** | DeepSeek | 128,000 | $0.55 | $0.14 | 🧠 Reasoning |
| **DeepSeek V3** | DeepSeek | 128,000 | $0.14 | $0.014 | Standard MoE |
| **Llama 3.3 70B** | Meta | 128,000 | $0.59 | $0.30 | Open Weights |
| **Qwen 2.5 Coder 32B** | Alibaba | 128,000 | $0.30 | $0.15 | Code Specialist |

---

## 🧪 Testing

```bash
pytest -v
```
All 32 tests passing covering model resolution, reasoning optimization, cache prefix stabilization, and token calibration.

---

## 📄 License
MIT
