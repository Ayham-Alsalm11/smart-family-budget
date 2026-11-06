"""The risk of §3.1: does GP1's parser get cut at max_tokens=200 on gpt-oss-120b? Spends free-tier Groq quota.

    backend\\.venv\\Scripts\\python -m eval.check_truncation

Ten sentences spread evenly over baseline/evaluation/test_dataset.csv go through GP1's parser at the default reasoning
level and at low. Writes results/gp1_truncation_<timestamp>/ (results.json, summary.md). The API key goes to the GP1
process through its environment and is never printed.
"""
import csv
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from statistics import mean
from typing import Any

from eval import EVAL_DIR, GENERATED, GP1_PYTHON, RESULTS, ROOT

from app.settings import get_settings  # noqa: E402

DATASET = ROOT / "baseline" / "evaluation" / "test_dataset.csv"
SENTENCES = 10
LEVELS = {"default": None, "low": "low"}
DAILY_QUOTA_EXIT = 2


def choose_sentences(path: Path = DATASET, count: int = SENTENCES) -> list[dict[str, str]]:
    """Evenly spaced, so that the set is reproducible and reaches the questions at the end of the file."""
    with open(path, encoding="utf-8") as file:
        rows = list(csv.DictReader(file))
    step = len(rows) / count
    return [rows[int(index * step + step / 2)] for index in range(count)]


def summarise(items: list[dict[str, Any]]) -> dict[str, Any]:
    finished = [item for item in items if item["finish_reason"]]
    return {
        "sentences": len(items),
        "valid_json": sum(1 for item in items if item["valid_json"]),
        "cut_at_max_tokens": sum(1 for item in items if item["finish_reason"] == "length"),
        "crashes": sum(1 for item in items if item["crash"]),
        "reasoning_tokens_mean": round(mean(item["reasoning_tokens"] for item in finished), 1) if finished else None,
        "reasoning_tokens_max": max((item["reasoning_tokens"] for item in finished), default=None),
        "completion_tokens_max": max((item["completion_tokens"] for item in finished), default=None),
        "total_tokens": sum(item["total_tokens"] for item in items),
    }


def markdown(report: dict[str, Any]) -> str:
    lines = [f"# GP1 truncation check, {report['started']}", "",
             f"Model `{report['model']}`, `max_tokens` 200, {SENTENCES} sentences through `parse_expense` (§3.1).", "",
             "| Level | Valid JSON | Cut at max_tokens | Crashes | Reasoning tokens (mean / max) | Largest completion | Tokens |",
             "|---|---|---|---|---|---|---|"]
    for name, level in report["levels"].items():
        s = level["summary"]
        lines.append(f"| {name} | {s['valid_json']}/{s['sentences']} | {s['cut_at_max_tokens']} | {s['crashes']} | "
                     f"{s['reasoning_tokens_mean']} / {s['reasoning_tokens_max']} | {s['completion_tokens_max']} | "
                     f"{s['total_tokens']} |")
    lines += ["", "| Sentence | Expected | " + " | ".join(f"{name}: type, finish, reasoning" for name in report["levels"]) + " |",
              "|---|---|" + "---|" * len(report["levels"])]
    for index, chosen in enumerate(report["sentences"]):
        cells = []
        for level in report["levels"].values():
            item = level["items"][index] if index < len(level["items"]) else None
            cells.append(f"{item['type']}, {item['finish_reason']}, {item['reasoning_tokens']}" if item else "not run")
        lines.append(f"| {chosen['sentence']} | {chosen['expected_type']} | " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def main() -> int:
    for stream in (sys.stdout, sys.stderr):
        stream.reconfigure(encoding="utf-8")
    settings = get_settings()
    if not settings.groq_key_present:
        print("Stopped: GROQ_API_KEY is not set.", file=sys.stderr)
        return 1
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    folder = RESULTS / f"gp1_truncation_{stamp}"
    folder.mkdir(parents=True)
    work = GENERATED / "truncation" / stamp
    work.mkdir(parents=True)
    chosen = choose_sentences()
    sentences = work / "sentences.json"
    sentences.write_text(json.dumps([row["sentence"] for row in chosen], ensure_ascii=False), encoding="utf-8")

    report: dict[str, Any] = {"started": stamp, "model": settings.model_main, "sentences": chosen, "levels": {}}
    code = 0
    for name, effort in LEVELS.items():
        out = work / f"{name}.jsonl"
        env = {k: v for k, v in os.environ.items() if not k.startswith("BASELINE_")}
        # The parser opens neither path; they are set so that nothing can reach baseline/data/.
        env.update(PYTHONIOENCODING="utf-8", GROQ_API_KEY=settings.groq_api_key.get_secret_value(),
                   BASELINE_LLM_MODEL=settings.model_main, BASELINE_DB_PATH=str(work / "gp1.db"),
                   BASELINE_CHROMA_DIR=str(work / "gp1_chroma"))
        if effort:
            env["BASELINE_REASONING_EFFORT"] = effort
        done = subprocess.run([str(GP1_PYTHON), str(EVAL_DIR / "gp1_truncation.py"), "--sentences", str(sentences),
                               "--out", str(out)], cwd=ROOT / "baseline", env=env, capture_output=True, text=True,
                              encoding="utf-8")
        items = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines()] if out.exists() else []
        report["levels"][name] = {"reasoning_effort": effort, "items": items, "summary": summarise(items)}
        (folder / "results.json").write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
        print(name, report["levels"][name]["summary"], flush=True)
        if done.returncode:
            print(f"Stopped at level {name}, exit code {done.returncode}: {done.stderr[-500:]}", file=sys.stderr)
            code = done.returncode
            break
    (folder / "summary.md").write_text(markdown(report), encoding="utf-8")
    print(f"saved {folder}", flush=True)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
