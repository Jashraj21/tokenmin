"""
TokenMin Interactive Showcase / Demo.
Demonstrates optimizing prompts, code, JSON, logs, and conversational contexts.
"""
from pathlib import Path
import tokenmin
from tokenmin.metrics import display_rich_report

def main():
    examples_dir = Path(__file__).parent

    print("\n" + "=" * 60)
    print(" 🚀 DEMO 1: NATURAL LANGUAGE PROMPT MINIMIZATION")
    print("=" * 60)
    prompt_text = (examples_dir / "sample_prompt.txt").read_text()
    res_prompt = tokenmin.minimize(prompt_text, level="moderate")
    display_rich_report(res_prompt)
    print("\n[Optimized Prompt]:\n", res_prompt.text)

    print("\n" + "=" * 60)
    print(" 🚀 DEMO 2: PYTHON CODE SKELETONIZATION (FOR AGENTS / REPO CONTEXT)")
    print("=" * 60)
    code_text = (examples_dir / "sample_code.py").read_text()
    res_code = tokenmin.minimize(code_text, skeleton=True)
    display_rich_report(res_code)
    print("\n[Skeletonized API Interface]:\n", res_code.text)

    print("\n" + "=" * 60)
    print(" 🚀 DEMO 3: STRUCTURED JSON TABULARIZATION & NULL PRUNING")
    print("=" * 60)
    json_text = (examples_dir / "sample_data.json").read_text()
    res_json = tokenmin.minimize(json_text, level="aggressive")
    display_rich_report(res_json)
    print("\n[Minified Compact JSON]:\n", res_json.text)

    print("\n" + "=" * 60)
    print(" 🚀 DEMO 4: REPEATED LOG COMPACTING")
    print("=" * 60)
    log_text = (examples_dir / "sample_logs.log").read_text()
    res_log = tokenmin.minimize(log_text)
    display_rich_report(res_log)
    print("\n[Compressed Logs]:\n", res_log.text)

if __name__ == "__main__":
    main()

