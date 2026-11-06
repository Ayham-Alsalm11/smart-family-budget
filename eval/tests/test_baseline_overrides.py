"""The GP1 overrides of §3.1: without BASELINE_LLM_MODEL and BASELINE_REASONING_EFFORT the request is the original one."""
import json
import os
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
GP1_PYTHON = ROOT / "venv" / "Scripts" / "python.exe"
PROBE = Path(__file__).with_name("baseline_payload_probe.py")
MESSAGES = [{"role": "system", "content": "system"}, {"role": "user", "content": "user"}]
OLD_MODEL = "llama-3.3-70b-versatile"
ORIGINAL = {
    "parser": {"model": OLD_MODEL, "messages": MESSAGES, "temperature": 0.1, "max_tokens": 200},
    "engine": {"model": OLD_MODEL, "messages": MESSAGES, "temperature": 0.7, "max_tokens": 500},
}

pytestmark = pytest.mark.skipif(not GP1_PYTHON.exists(), reason="GP1's root venv is not here")


def built_requests(tmp_path: Path, **variables: str) -> dict:
    env = {k: v for k, v in os.environ.items() if not k.startswith("BASELINE_")}
    env.update(PYTHONIOENCODING="utf-8", BASELINE_DB_PATH=str(tmp_path / "gp1.db"),
               BASELINE_CHROMA_DIR=str(tmp_path / "chroma"), **variables)
    done = subprocess.run([str(GP1_PYTHON), str(PROBE)], cwd=ROOT / "baseline", env=env, capture_output=True, text=True,
                          encoding="utf-8", timeout=120)
    assert done.returncode == 0, done.stderr
    return json.loads(done.stdout.strip().splitlines()[-1])


def test_without_the_variables_the_request_is_the_original(tmp_path):
    assert built_requests(tmp_path) == ORIGINAL


def test_an_empty_effort_changes_nothing(tmp_path):
    assert built_requests(tmp_path, BASELINE_REASONING_EFFORT="") == ORIGINAL


def test_the_model_override_changes_only_the_model(tmp_path):
    built = built_requests(tmp_path, BASELINE_LLM_MODEL="openai/gpt-oss-120b")
    for part in ORIGINAL:
        assert built[part] == {**ORIGINAL[part], "model": "openai/gpt-oss-120b"}


def test_the_effort_override_adds_only_the_effort(tmp_path):
    built = built_requests(tmp_path, BASELINE_LLM_MODEL="openai/gpt-oss-120b", BASELINE_REASONING_EFFORT="low")
    for part in ORIGINAL:
        assert built[part] == {**ORIGINAL[part], "model": "openai/gpt-oss-120b", "reasoning_effort": "low"}
