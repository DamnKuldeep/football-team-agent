"""Runtime settings shared by every front end (CLI and Streamlit UI)."""
from __future__ import annotations

import os
from pathlib import Path

from .cost_tracker import CostTracker
from .llm_catalog import DEFAULT_MODEL

ROOT = Path(__file__).resolve().parents[2]
# Where teams/, matches/ and scouting/ live. Point FTA_DATA_DIR elsewhere to
# keep a separate sandbox (demos, experiments) without touching your data.
DATA_ROOT = Path(os.environ.get("FTA_DATA_DIR") or ROOT)


def load_dotenv(path: Path = ROOT / ".env") -> None:
    """Minimal KEY=VALUE .env reader (supports `#` comments). Variables already
    set in the real environment always win over the file."""
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.split(" #", 1)[0].strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip("\"'"))


SOURCES = ("synthetic", "kaggle", "both")
_LEGACY_TIERS = {"free": DEFAULT_MODEL, "cheap": DEFAULT_MODEL}  # old MODEL_TIER values


def llm_model() -> str:
    """OpenRouter model id: MODEL in .env, else the old MODEL_TIER, else the default."""
    return (os.environ.get("MODEL") or _LEGACY_TIERS.get(os.environ.get("MODEL_TIER", ""), "")
            or DEFAULT_MODEL)


def player_source() -> str:
    source = os.environ.get("PLAYER_SOURCE", "synthetic")
    return source if source in SOURCES else "synthetic"


def make_tracker() -> CostTracker:
    return CostTracker(budget_usd=float(os.environ.get("BUDGET_USD", "5.0")),
                       mode=os.environ.get("BUDGET_MODE", "warn"))


def llm_enabled() -> bool:
    return bool(os.environ.get("OPENROUTER_API_KEY"))
