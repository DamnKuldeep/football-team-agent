"""Team persistence. Every change writes a NEW version rather than mutating
the existing file, so you can always go back or compare before/after.

teams/
├── index.json                # {team_id: {name, latest_version, max_version, created_at}}
└── <team_id>/
    ├── v1.json
    ├── v2.json
    └── ...

Individual versions can be deleted, but version numbers are never reused
(`max_version` remembers the highest ever issued), so "v3" in an old match
always means the same line-up.
"""
from __future__ import annotations

import json
import re
import shutil
import time
from pathlib import Path

from .config import DATA_ROOT
from .models import Team

TEAMS_DIR = DATA_ROOT / "teams"
INDEX_PATH = TEAMS_DIR / "index.json"


def _load_index() -> dict:
    if not INDEX_PATH.exists():
        return {}
    return json.loads(INDEX_PATH.read_text(encoding="utf-8"))


def _save_index(index: dict) -> None:
    TEAMS_DIR.mkdir(parents=True, exist_ok=True)
    INDEX_PATH.write_text(json.dumps(index, indent=2), encoding="utf-8")


def save_version(team: Team) -> Path:
    team_dir = TEAMS_DIR / team.team_id
    team_dir.mkdir(parents=True, exist_ok=True)
    path = team_dir / f"v{team.version}.json"
    if path.exists():
        raise FileExistsError(
            f"{path} already exists -- versions are append-only, bump team.version instead of overwriting"
        )
    path.write_text(team.model_dump_json(indent=2), encoding="utf-8")

    index = _load_index()
    entry = index.get(team.team_id, {"name": team.name, "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())})
    entry["name"] = team.name
    entry["latest_version"] = max(entry.get("latest_version", 0), team.version)
    entry["max_version"] = max(entry.get("max_version", 0), team.version)
    index[team.team_id] = entry
    _save_index(index)
    return path


def load_team(team_id: str, version: int | None = None) -> Team:
    index = _load_index()
    if team_id not in index:
        raise KeyError(f"no team '{team_id}' found. Known teams: {list(index.keys())}")
    v = version or index[team_id]["latest_version"]
    path = TEAMS_DIR / team_id / f"v{v}.json"
    if not path.exists():
        raise FileNotFoundError(f"{path} missing (index says latest_version={index[team_id]['latest_version']})")
    return Team.model_validate(json.loads(path.read_text(encoding="utf-8")))


def list_teams() -> dict:
    return _load_index()


def list_versions(team_id: str) -> list[int]:
    """Saved version numbers, newest first (gaps where versions were deleted)."""
    team_dir = TEAMS_DIR / team_id
    numbers = [int(m.group(1)) for p in team_dir.glob("v*.json") if (m := re.fullmatch(r"v(\d+)\.json", p.name))]
    return sorted(numbers, reverse=True)


def next_version_number(team_id: str) -> int:
    entry = _load_index().get(team_id, {})
    return max(entry.get("max_version", 0), entry.get("latest_version", 0)) + 1


def delete_version(team_id: str, version: int) -> None:
    """Delete one version. The team's only remaining version can't be deleted
    (delete the team instead); its number is never issued again."""
    versions = list_versions(team_id)
    if version not in versions:
        raise KeyError(f"{team_id} has no v{version}")
    if len(versions) == 1:
        raise ValueError(f"v{version} is {team_id}'s only version — delete the team instead")
    (TEAMS_DIR / team_id / f"v{version}.json").unlink()
    index = _load_index()
    entry = index[team_id]
    entry["max_version"] = max(entry.get("max_version", 0), entry["latest_version"])
    entry["latest_version"] = max(v for v in versions if v != version)
    _save_index(index)


def delete_team(team_id: str) -> None:
    """Remove a team and every one of its versions."""
    index = _load_index()
    if team_id not in index:
        raise KeyError(f"no team '{team_id}' found. Known teams: {list(index.keys())}")
    shutil.rmtree(TEAMS_DIR / team_id, ignore_errors=True)
    del index[team_id]
    _save_index(index)


def next_version(team: Team, diff: dict) -> Team:
    """Clone `team` as the next version number with a diff attached, ready
    for save_version(). Does not mutate the input team."""
    new_team = team.model_copy(deep=True)
    new_team.parent_version = team.version
    new_team.version = max(team.version + 1, next_version_number(team.team_id))
    new_team.diff = diff
    return new_team
