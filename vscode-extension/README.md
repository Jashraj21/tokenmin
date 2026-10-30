# ⚡ TokenMin for VS Code

**TokenMin for VS Code** slashes token consumption by **70% to 90%** when feeding codebase context, prompts, debug logs, and data structures into **GitHub Copilot Chat**, **Continue.dev**, **Claude Dev / Cline**, and **ChatGPT**.

---

## 🎯 How It Cuts Tokens in VS Code

### 1. AST Skeletonization for `@workspace` Context (Saves ~80% Tokens)
When you add files to Copilot Chat or Claude:
* **Without TokenMin**: The AI loads 500 lines of implementation details, internal algorithms, and docstrings (e.g. 3,500 tokens).
* **With TokenMin (`Skeletonize`)**: Extracts function signatures, class definitions, and type hints with `...` bodies (only ~350 tokens). The AI understands your entire architecture without filling your context window!

### 2. Right-Click "Copy Optimized for AI"
Highlight any snippet of code, JSON, SQL, or prompt text:
* Right-click → **TokenMin: Copy Minimized to Clipboard for Copilot/ChatGPT**.
* Automatically removes comments, dead whitespace, polite filler words, and null fields before you paste.

### 3. Terminal & Debug Console Log Minimization
When terminal output or crash logs contain 100 repeated lines of errors or timestamps:
* TokenMin collapses them down into `[Repeated 98x]` and shortens timestamps, saving hundreds of tokens.

---

## ⚙️ Extension Settings

| Setting | Options | Default | Description |
| :--- | :--- | :--- | :--- |
| `tokenmin.model` | `gpt-4o`, `o3-mini`, `o1`, `claude-3-5-sonnet`, `gemini-2.0-flash`, `gemini-4`, `gpt-6` | `gpt-4o` | Target model for token counting & optimization |
| `tokenmin.level` | `conservative`, `moderate`, `aggressive` | `moderate` | Compression aggressiveness |
