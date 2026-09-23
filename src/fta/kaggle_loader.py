"""Loads a Kaggle FIFA/EA-FC-style ratings CSV, verifies it, and converts it
into our Player schema so the rest of the pipeline can't tell the difference
between a synthetic pool and a Kaggle-derived one.

NOTE ON LICENSING: the underlying ratings in these Kaggle datasets are
derived from EA Sports FC / sofifa.com data. Treat this as personal/research
use only -- don't redistribute the raw CSV, and don't ship real player names
in anything public-facing. Use `anonymize=True` (default) to swap real names
for generated ones while keeping the real stat distributions; that sidesteps
likeness concerns entirely at zero cost to data quality.

Get the CSV with `python scripts/download_kaggle_players.py` (public dataset,
no Kaggle account needed), which saves it to data/raw/kaggle_players.csv; or
point --csv at any CSV with the same headers. Column mapping and derived
attributes live in data/adapters/kaggle_column_map.json.
"""
from __future__ import annotations

import csv
import json
import random
import string
from pathlib import Path

from .models import ATTRIBUTE_NAMES, GK_ONLY, Player

ROOT = Path(__file__).resolve().parents[2]
COLMAP_PATH = ROOT / "data" / "adapters" / "kaggle_column_map.json"

# Same name pools as scripts/generate_synthetic_players.py so anonymized players
# are indistinguishable in style from synthetic ones. With a middle initial this
# gives 32 * 26 * 28 = 23,296 unique names -- enough for a full FIFA season.
_FIRST = [
    "Arjun", "Kabir", "Rohan", "Dev", "Aarav", "Vikram", "Ishaan", "Rahul",
    "Marco", "Lucas", "Diego", "Mateus", "Thiago", "Bruno", "Kwame", "Amir",
    "Noah", "Liam", "Ethan", "Mason", "Leo", "Hugo", "Karim", "Youssef",
    "Sven", "Erik", "Jonas", "Felix", "Tomas", "Milos", "Kenji", "Haru",
]
_LAST = [
    "Verma", "Mehta", "Sharma", "Kapoor", "Nair", "Rossi", "Silva", "Costa",
    "Muller", "Novak", "Kovačić", "Diallo", "Traore", "Nakamura",
    "Sato", "Andersen", "Berg", "Fischer", "Garcia", "Martins", "Okafor",
    "Osei", "Petrov", "Lindqvist", "Haddad", "Al-Farsi", "Torres", "Reyes",
]


def _load_colmap() -> dict:
    return json.loads(COLMAP_PATH.read_text(encoding="utf-8"))


def _parse_positions(raw: str) -> list[str]:
    """Kaggle position strings look like 'CB, CDM' or 'ST'. We map their
    finer-grained codes down to our coarser position buckets."""
    code_map = {
        "GK": "GK",
        "CB": "CB", "RCB": "CB", "LCB": "CB",
        "RB": "FB", "LB": "FB", "RWB": "FB", "LWB": "FB",
        "CDM": "DM",
        "CM": "CM", "RCM": "CM", "LCM": "CM",
        "CAM": "AM",
        "RM": "WING", "LM": "WING", "RW": "WING", "LW": "WING",
        "ST": "ST", "CF": "ST",
    }
    out = []
    for tok in raw.split(","):
        tok = tok.strip().upper()
        mapped = code_map.get(tok)
        if mapped and mapped not in out:
            out.append(mapped)
    return out or ["CM"]


def _to_float_or_none(v: str) -> float | None:
    v = (v or "").strip()
    if v == "":
        return None
    try:
        return float(v)
    except ValueError:
        return None


def convert_row(row: dict, colmap: dict) -> tuple[dict | None, list[str]]:
    """Returns (player_dict_or_None, warnings). Never raises on a single bad
    row -- callers should skip and log rather than abort the whole load."""
    warnings: list[str] = []
    ident = colmap["identity_columns"]

    try:
        player_id = f"k_{row[ident['player_id']].strip()}"
        name = row[ident["name"]].strip()
        foot_raw = row[ident["preferred_foot"]].strip().lower()
        preferred_foot = "left" if foot_raw.startswith("l") else "right"
        weak_foot = int(float(row[ident["weak_foot_rating"]]))
        age = int(float(row[ident["age"]]))
        # full name, not a 2-letter prefix: "Austria"[:2] == "Australia"[:2] would
        # falsely trigger the same-nationality chemistry rule
        nationality = (row.get(ident["nationality"], "") or "").strip() or "XX"
        positions = _parse_positions(row[ident["player_positions"]])
    except (KeyError, ValueError) as e:
        return None, [f"skipped row, bad identity fields: {e}"]

    is_gk = "GK" in positions
    attrs: dict[str, float | None] = {name: None for name in ATTRIBUTE_NAMES}
    for csv_col, attr_name in colmap["column_to_attribute"].items():
        if csv_col not in row:
            continue
        val = _to_float_or_none(row[csv_col])
        if val is None:
            continue
        if attr_name in GK_ONLY and not is_gk:
            continue  # GK stats stay null for outfield players
        attrs[attr_name] = round(min(100.0, max(0.0, val)), 1)

    for attr_name, rule in colmap.get("derived_attributes", {}).items():
        if attr_name.startswith("_") or (attr_name in GK_ONLY and not is_gk):
            continue
        (op, sources), = rule.items()
        vals = [_to_float_or_none(row.get(c, "")) for c in sources]
        if sources and all(v is not None for v in vals):
            combined = sum(vals) / len(vals) if op == "mean" else max(vals)
            attrs[attr_name] = round(min(100.0, max(0.0, combined)), 1)

    for missing in colmap.get("no_direct_mapping", []):
        if attrs.get(missing) is None and not (missing in GK_ONLY and not is_gk):
            warnings.append(f"{player_id}: no source column for '{missing}', left null")

    stamina = attrs.get("stamina")
    player = {
        "player_id": player_id,
        "name": name,
        "preferred_foot": preferred_foot,
        "weak_foot_rating": max(1, min(5, weak_foot)),
        "age": max(15, min(45, age)),
        "nationality": nationality,
        "stamina_base": int(stamina) if stamina is not None else 65,
        "attributes": attrs,
        "natural_positions": positions,
        "tags": ["kaggle_derived"],
        "source": "kaggle",
    }
    return player, warnings


def anonymize(players: list[dict], seed: int = 7) -> list[dict]:
    """Deterministic, collision-free fake names. Walks a shuffled list of every
    name combination (the old retry-until-unique loop never terminated once a
    pool outgrew the 224 possible names); a numeric suffix only kicks in past
    23,296 players."""
    combos = [f"{f} {i}. {last}" for f in _FIRST for i in string.ascii_uppercase for last in _LAST]
    random.Random(seed).shuffle(combos)
    for n, p in enumerate(players):
        lap, idx = divmod(n, len(combos))
        p["name"] = combos[idx] + (f" {lap + 1}" if lap else "")
        p["tags"] = [t for t in p["tags"] if t != "kaggle_derived"] + ["kaggle_derived", "anonymized"]
    return players


def load_and_convert(
    csv_path: Path,
    anonymize_names: bool = True,
    limit: int | None = None,
) -> tuple[list[Player], list[str]]:
    """Full pipeline: read CSV -> convert rows -> validate against Player
    schema -> optionally anonymize. Returns (players, all_warnings)."""
    colmap = _load_colmap()
    all_warnings: list[str] = []
    raw_players: list[dict] = []

    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            if limit and i >= limit:
                break
            pdict, warns = convert_row(row, colmap)
            all_warnings.extend(warns)
            if pdict:
                raw_players.append(pdict)

    if anonymize_names:
        raw_players = anonymize(raw_players)

    validated: list[Player] = []
    seen_ids = set()
    for pdict in raw_players:
        if pdict["player_id"] in seen_ids:
            all_warnings.append(f"duplicate player_id {pdict['player_id']}, skipped")
            continue
        seen_ids.add(pdict["player_id"])
        try:
            validated.append(Player.model_validate(pdict))
        except Exception as e:  # noqa: BLE001 -- want to keep loading on bad rows
            all_warnings.append(f"validation failed for {pdict.get('player_id')}: {e}")

    return validated, all_warnings


def write_processed(players: list[Player], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps([p.model_dump() for p in players], indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
