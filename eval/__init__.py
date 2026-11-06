"""Evaluation code (§2, §9). Runs in backend/.venv from the repository root: python -m eval.<module>.

The exceptions are gp1_build.py, gp1_answer.py and gp1_truncation.py: they run in GP1's root venv from inside baseline/,
and are started by the code here, never imported.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
EVAL_DIR = ROOT / "eval"
DATA_DIR = ROOT / "data" / "eval"
GENERATED = DATA_DIR / "generated"
RESULTS = ROOT / "results"
GP1_PYTHON = ROOT / "venv" / "Scripts" / "python.exe"

sys.path.insert(0, str(ROOT / "backend"))  # for `app`, which is not installed as a package
