from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

from .models import Player, PositionTemplate

ROOT = Path(__file__).resolve().parents[2]
PROCESSED = ROOT / "data" / "processed"
TEMPLATES = ROOT / "data" / "templates"

PlayerSource = Literal["synthetic", "kaggle", "both"]


SYNTHETIC_PATH = PROCESSED / "players_synthetic.json"
KAGGLE_PATH = PROCESSED / "players_kaggle.json"
_MISSING_HINT = {
    SYNTHETIC_PATH: "python scripts/generate_synthetic_players.py",
    KAGGLE_PATH: "python scripts/download_kaggle_players.py && fta load-kaggle --csv data/raw/kaggle_players.csv",
}


def load_pool(source: PlayerSource = "both") -> list[Player]:
    """`both` means every pool that exists (so a machine without the Kaggle
    download still works); `synthetic` / `kaggle` require that exact file."""
    files = {
        "synthetic": [SYNTHETIC_PATH],
        "kaggle": [KAGGLE_PATH],
        "both": [p for p in (SYNTHETIC_PATH, KAGGLE_PATH) if p.exists()] or [SYNTHETIC_PATH],
    }[source]

    pool: list[Player] = []
    seen_ids: set[str] = set()
    for path in files:
        if not path.exists():
            raise FileNotFoundError(f"{path} not found. Run: {_MISSING_HINT[path]}")
        raw = json.loads(path.read_text(encoding="utf-8"))
        for pdict in raw:
            if pdict["player_id"] in seen_ids:
                continue  # both-mode collision guard; synthetic/kaggle ids never collide by prefix anyway
            seen_ids.add(pdict["player_id"])
            pool.append(Player.model_validate(pdict))
    return pool


def load_position_templates() -> dict[str, PositionTemplate]:
    raw = json.loads((TEMPLATES / "positions.json").read_text(encoding="utf-8"))
    return {t["position"]: PositionTemplate.model_validate(t) for t in raw}


def load_formations() -> dict[str, list[dict]]:
    return json.loads((TEMPLATES / "formations.json").read_text(encoding="utf-8"))


def load_formation_weights() -> dict[str, dict]:
    raw = json.loads((TEMPLATES / "formation_weights.json").read_text(encoding="utf-8"))
    return {k: v for k, v in raw.items() if not k.startswith("_")}


def load_chemistry_rules() -> list[dict]:
    return json.loads((TEMPLATES / "chemistry_rules.json").read_text(encoding="utf-8"))


def pool_lookup(pool: list[Player]) -> dict[str, Player]:
    return {p.player_id: p for p in pool}
