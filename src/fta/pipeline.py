"""The pipeline as plain functions. The CLI and the Streamlit UI are
both thin shells over this module, so neither holds any selection, swap or
reporting logic of its own -- a decision made in one front end is identical
to the same decision made in the other.

On-disk layout (all plain JSON):
    teams/<team_id>/v<N>.json      append-only team versions (+ teams/index.json)
    matches/M0001/events.json      one folder per match: event log with line-ups ...
    matches/M0001/report.json      ... its per-player report
    matches/M0001/analysis.json    ... and the assistant's match analysis, if generated
    scouting/<player_id>.json      the assistant's scouting reports
    teams/<team_id>/review.json    the assistant's review of a team's form
"""
from __future__ import annotations

import bisect
import json
import random
import re
import shutil
from collections import Counter
from dataclasses import dataclass, field
from datetime import UTC, datetime
from itertools import combinations
from pathlib import Path

from . import data_loader, storage
from .config import DATA_ROOT, llm_model, player_source
from .cost_tracker import CostTracker
from .llm_client import parse_brief
from .models import (
    ATTRIBUTE_NAMES,
    GK_ONLY,
    ChemistryRule,
    EventLog,
    HardFilter,
    Player,
    PlayerReport,
    PositionTemplate,
    Shortlist,
    SwapRequest,
    Team,
)
from .report_generator import build_reports, debrief, derive_swap_request, report_key, team_stats
from .scoring import Together, chemistry_breakdown, fit_score, team_avg_fit, team_chemistry
from .simulator import simulate_match
from .swap_engine import suggest_swap
from .team_builder import build_team as _build_team

ROOT = Path(__file__).resolve().parents[2]
MATCHES_DIR = DATA_ROOT / "matches"
SCOUTING_DIR = DATA_ROOT / "scouting"
_MATCH_ID = re.compile(r"^M(\d{4,})$")


# ---------------------------------------------------------------------------
# Context: everything loaded once per session
# ---------------------------------------------------------------------------

@dataclass
class Context:
    pool: list[Player]
    templates: dict[str, PositionTemplate]          # base weights (positions.json)
    formations: dict[str, list[dict]]
    rules: list[ChemistryRule]
    formation_weights: dict[str, dict]              # formation_weights.json
    lookup: dict[str, Player] = field(init=False)
    _percentiles: dict = field(default_factory=dict, init=False, repr=False)

    def __post_init__(self):
        self.lookup = {p.player_id: p for p in self.pool}

    def percentile(self, position: str, attr: str, value: float) -> float:
        """Share (0-100) of players listed at `position` with a lower `attr`."""
        key = (position, attr)
        if key not in self._percentiles:
            self._percentiles[key] = sorted(
                p.attributes[attr] for p in self.pool
                if position in p.natural_positions and p.attributes.get(attr) is not None)
        values = self._percentiles[key]
        return round(100 * bisect.bisect_left(values, value) / len(values), 1) if values else 0.0


def load_context(source: str | None = None) -> Context:
    """Ingest the player pool and load templates. `source`
    defaults to PLAYER_SOURCE from .env (itself defaulting to the demo pool)."""
    return Context(
        pool=data_loader.load_pool(source or player_source()),
        templates=data_loader.load_position_templates(),
        formations=data_loader.load_formations(),
        rules=[ChemistryRule.model_validate(r) for r in data_loader.load_chemistry_rules()],
        formation_weights=data_loader.load_formation_weights(),
    )


# ---------------------------------------------------------------------------
# Weights: base -> formation defaults -> team-tuned
# ---------------------------------------------------------------------------

def formation_positions(ctx: Context, formation: str) -> list[str]:
    return list(dict.fromkeys(s["position"] for s in ctx.formations[formation]))


def role_name(ctx: Context, formation: str, position: str) -> str:
    return ctx.formation_weights.get(formation, {}).get(position, {}).get("role", position)


def formation_defaults(ctx: Context, formation: str) -> dict[str, dict[str, float]]:
    """position -> default weights for this formation."""
    overrides = ctx.formation_weights.get(formation, {})
    return {pos: dict(overrides.get(pos, {}).get("weights") or ctx.templates[pos].attribute_weights)
            for pos in formation_positions(ctx, formation)}


def normalize_weights(weights: dict[str, dict[str, float]]) -> dict[str, dict[str, float]]:
    """Drop zero/negative weights and rescale each position to sum to 1."""
    out = {}
    for pos, w in weights.items():
        w = {a: float(v) for a, v in w.items() if a in ATTRIBUTE_NAMES and v and v > 0}
        total = sum(w.values())
        if not total:
            raise ValueError(f"{pos}: at least one attribute needs a weight above 0")
        out[pos] = {a: round(v / total, 4) for a, v in w.items()}
    return out


def templates_for(ctx: Context, weights: dict[str, dict[str, float]]) -> dict[str, PositionTemplate]:
    out = dict(ctx.templates)
    for pos, w in weights.items():
        out[pos] = ctx.templates[pos].model_copy(update={"attribute_weights": dict(w)})
    return out


def team_weights(ctx: Context, team: Team) -> dict[str, dict[str, float]]:
    return team.weights or formation_defaults(ctx, team.formation)


def team_templates(ctx: Context, team: Team) -> dict[str, PositionTemplate]:
    return templates_for(ctx, team_weights(ctx, team))


def _rescore(ctx: Context, team: Team) -> Team:
    positions = {s.slot_id: s.position for s in team.slots}
    team.avg_fit_score = team_avg_fit(team, ctx.lookup, team_templates(ctx, team), positions)
    team.chemistry_score = team_chemistry(team, ctx.lookup, ctx.rules, together_counts())
    return team


def missing_players(ctx: Context, team: Team) -> list[str]:
    """Players in the team that aren't in the loaded pool (built from another pool)."""
    return [pid for pid in team.player_ids() if pid not in ctx.lookup]


def rescored(ctx: Context, team: Team) -> Team:
    """Copy of `team` with fit/chemistry recomputed now. Saved scores are a
    snapshot; chemistry drifts as matches add "played together" history."""
    return _rescore(ctx, team.model_copy(deep=True))


def chemistry_details(ctx: Context, team: Team) -> tuple[float, list[dict]]:
    return chemistry_breakdown(team, ctx.lookup, ctx.rules, together_counts())


# ---------------------------------------------------------------------------
# Build / retune
# ---------------------------------------------------------------------------

def build_team(ctx: Context, team_id: str, name: str, formation: str = "4-3-3",
               exclude_team_id: str | None = None,
               weights: dict[str, dict[str, float]] | None = None) -> tuple[Team, int]:
    """Build an XI with `weights` (default: the formation's) and save it.
    Returns (team, n_players_excluded)."""
    weights = normalize_weights(weights or formation_defaults(ctx, formation))
    pool = ctx.pool
    excluded = 0
    if exclude_team_id and exclude_team_id in storage.list_teams():
        used = set(storage.load_team(exclude_team_id).player_ids())
        pool = [p for p in pool if p.player_id not in used]
        excluded = len(used)

    team = _build_team(team_id, name, formation, ctx.formations[formation], pool,
                       templates_for(ctx, weights), ctx.rules)
    team.weights = weights
    latest = storage.list_teams().get(team_id, {}).get("latest_version")
    if latest:
        # versions are append-only: a rebuild of an existing team becomes its next version
        team.parent_version = latest
        team.version = storage.next_version_number(team_id)
        team.diff = {"reason": "rebuilt via build-team"}
    storage.save_version(_rescore(ctx, team))
    return team, excluded


def retune_team(ctx: Context, team_id: str, weights: dict[str, dict[str, float]]) -> Team:
    """Save new position weights as the team's next version (same players, re-scored)."""
    team = storage.load_team(team_id)
    weights = normalize_weights(weights)
    new_team = storage.next_version(team, {"reason": "position weights tuned",
                                           "old_avg_fit": team.avg_fit_score})
    new_team.weights = weights
    _rescore(ctx, new_team)
    new_team.diff["new_avg_fit"] = new_team.avg_fit_score
    storage.save_version(new_team)
    return new_team


# ---------------------------------------------------------------------------
# Requests, shortlists, swaps
# ---------------------------------------------------------------------------

def slot_position(team: Team, slot: str) -> str:
    return next(s.position for s in team.slots if s.slot_id == slot)


def _person(ctx: Context, slot_id: str, pid: str | None) -> dict:
    p = ctx.lookup.get(pid) if pid else None
    if p is None:
        return {"slot": slot_id, "name": "<empty>", "age": 0, "foot": "", "nationality": ""}
    return {"slot": slot_id, "name": p.name, "age": p.age, "foot": p.preferred_foot, "nationality": p.nationality}


def request_context(ctx: Context, team: Team, slot: str) -> dict:
    """Everything the AI needs to read a request about `slot` without guessing:
    the line-up, the current player's full profile, the slot's neighbours on the
    pitch, and which nationalities and ages actually exist in the pool."""
    from .scoring import team_links
    position = slot_position(team, slot)
    current_id = next(s.player_id for s in team.slots if s.slot_id == slot)
    current = ctx.lookup.get(current_id) if current_id else None
    candidates = [p for p in ctx.pool if p.is_gk() == (position == "GK")]
    return {
        "slot": slot, "role": role_name(ctx, team.formation, position), "formation": team.formation,
        "current_player": ({**_person(ctx, slot, current_id), "weak_foot": current.weak_foot_rating,
                            "natural_positions": current.natural_positions,
                            "attributes": {a: v for a, v in current.attributes.items() if v is not None}}
                           if current else None),
        "lineup": [_person(ctx, s.slot_id, s.player_id) for s in team.slots],
        "neighbours": sorted({b if a == slot else a for a, b in team_links(team) if slot in (a, b)}),
        "pool": {"nationalities": sorted({p.nationality for p in ctx.pool}),
                 "age_range": [min(p.age for p in candidates), max(p.age for p in candidates)],
                 "players_available_for_this_position": len(candidates)},
    }


def swap_request_from_brief(ctx: Context, team: Team, slot: str, brief: str, model: str | None = None,
                            tracker: CostTracker | None = None) -> tuple[SwapRequest, dict]:
    """Natural-language request -> (structured request, interpretation details:
    source, reasons, rejected items, keyword cross-check, what the AI was sent)."""
    position = slot_position(team, slot)
    current = team_weights(ctx, team)[position]
    parsed = parse_brief(brief, model=model or llm_model(), tracker=tracker, position=position,
                         current_weights=current, context=request_context(ctx, team, slot))
    hard_filters = ([HardFilter(attribute="preferred_foot", value=parsed["hard_foot"])]
                    if parsed.get("hard_foot") else [])
    reasons = dict(parsed["reasons"])
    for f in parsed.get("filters", []):
        hard_filters.append(HardFilter(attribute=f["field"], value=f["value"], values=f.get("values"),
                                       min=f["min"], max=f["max"]))
        reasons[f"filter:{f['field']}"] = f["reason"]
    request = SwapRequest(target_slot=slot, reweight=parsed["reweight"], hard_filters=hard_filters,
                          rationale=brief, source="manual", adjustment_reasons=reasons,
                          interpreted_by=parsed["source"])
    return request, parsed


def weight_changes(ctx: Context, team: Team, request: SwapRequest) -> list[dict]:
    """Before/after weights for the request's slot, normalised the same way the
    swap engine does it: every attribute, its default share, its new share."""
    before = team_weights(ctx, team)[slot_position(team, request.target_slot)]
    merged = {**before, **request.reweight}
    total = sum(merged.values()) or 1
    rows = []
    for attr in sorted(set(before) | set(request.reweight), key=lambda a: -merged.get(a, 0)):
        rows.append({"attribute": attr, "before": before.get(attr, 0.0), "after": merged.get(attr, 0) / total,
                     "requested": attr in request.reweight,
                     "reason": request.adjustment_reasons.get(attr, "")})
    return rows


def shortlist(ctx: Context, team: Team, request: SwapRequest, top_n: int = 5) -> Shortlist:
    """Rank candidates for a slot (warn + rank lower, never block)."""
    return suggest_swap(request, team, ctx.pool, team_templates(ctx, team), ctx.rules,
                        top_n=top_n, together=together_counts())


def preview_swaps(ctx: Context, team: Team, changes: dict[str, str]) -> Team:
    """What-if copy of `team` with {slot_id: player_id} applied and fit/chemistry
    recomputed. Nothing is saved. Raises ValueError if a player would end up
    in two slots or a slot doesn't exist."""
    draft = team.model_copy(deep=True)
    unknown = set(changes) - {s.slot_id for s in draft.slots}
    if unknown:
        raise ValueError(f"unknown slot(s) {sorted(unknown)} for {team.team_id}")
    for s in draft.slots:
        if s.slot_id in changes:
            s.player_id = changes[s.slot_id]
    ids = draft.player_ids()
    if len(ids) != len(set(ids)):
        dupes = sorted({pid for pid in ids if ids.count(pid) > 1})
        raise ValueError(f"player(s) {dupes} would occupy more than one slot")
    return _rescore(ctx, draft)


def apply_swaps(ctx: Context, team_id: str, changes: dict[str, str],
                reason: str = "human-selected from shortlist") -> tuple[Team, dict]:
    """Write one or more slot changes as a single new team version."""
    team = storage.load_team(team_id)
    old_ids = {s.slot_id: s.player_id for s in team.slots}
    changes = {slot: pid for slot, pid in changes.items() if old_ids.get(slot) != pid}
    if not changes:
        raise ValueError("no changes to save -- every slot already holds that player")
    preview = preview_swaps(ctx, team, changes)
    diff = {
        "changes": [{"slot": slot, "old_player_id": old_ids[slot], "new_player_id": pid}
                    for slot, pid in changes.items()],
        "reason": reason,
        "old_avg_fit": team.avg_fit_score, "old_chemistry": team.chemistry_score,
        "new_avg_fit": preview.avg_fit_score, "new_chemistry": preview.chemistry_score,
    }
    new_team = storage.next_version(team, diff)
    new_team.slots = preview.slots
    new_team.avg_fit_score = preview.avg_fit_score
    new_team.chemistry_score = preview.chemistry_score
    storage.save_version(new_team)
    return new_team, diff


def apply_swap(ctx: Context, team_id: str, slot: str, player_id: str,
               reason: str = "human-selected from shortlist") -> tuple[Team, dict]:
    """Single-slot convenience wrapper around apply_swaps."""
    return apply_swaps(ctx, team_id, {slot: player_id}, reason)


def diff_changes(diff: dict | None) -> list[dict]:
    """Slot changes recorded in a version diff (handles the older single-slot format)."""
    if not diff:
        return []
    if "changes" in diff:
        return diff["changes"]
    if "slot" in diff:
        return [{k: diff[k] for k in ("slot", "old_player_id", "new_player_id")}]
    return []


def _match_side(ctx: Context, team_id: str, version: int | None, rename: bool) -> Team:
    """The team as it plays in a match. When a team meets another version of
    itself, each side gets its own id ("t_A@v1") so the two can't be confused."""
    team = rescored(ctx, storage.load_team(team_id, version))
    if rename:
        team.team_id = f"{team_id}@v{team.version}"
        team.name = f"{team.name} v{team.version}"
    return team


def base_team_id(match_team_id: str) -> str:
    """'t_A@v1' -> 't_A' (plain ids pass through)."""
    return match_team_id.split("@", 1)[0]


def evaluate_lineups(ctx: Context, variants: dict[str, Team], opponent: Team, n: int = 200) -> dict:
    """Play each variant against `opponent` n times. Every run draws fresh
    random seeds, but within a run all variants share the SAME seeds, so luck
    is identical and only the line-up differs (a paired comparison).
    Returns {"variants": {label: stats}, "difference": {...}} where the
    difference compares the last variant with the first."""
    seeds = random.SystemRandom().sample(range(1, 10_000_000), n)
    points: dict[str, list[int]] = {}
    out: dict[str, dict] = {}
    for label, team in variants.items():
        chem = {team.team_id: team.chemistry_score, opponent.team_id: opponent.chemistry_score}
        w = d = gf = ga = 0
        pts = []
        for seed in seeds:
            log = simulate_match("eval", team, opponent, ctx.lookup, seed=seed, chemistry=chem)
            f, a = log.final_score[log.team_a], log.final_score[log.team_b]
            w += f > a
            d += f == a
            gf += f
            ga += a
            pts.append(3 if f > a else 1 if f == a else 0)
        points[label] = pts
        out[label] = {"Win %": round(100 * w / n, 1), "Draw %": round(100 * d / n, 1),
                      "Loss %": round(100 * (n - w - d) / n, 1), "Points per match": round(sum(pts) / n, 2),
                      "Goals for": round(gf / n, 2), "Goals against": round(ga / n, 2)}
    labels = list(variants)
    diffs = [b - a for a, b in zip(points[labels[0]], points[labels[-1]])]
    mean = sum(diffs) / n
    sd = (sum((x - mean) ** 2 for x in diffs) / max(1, n - 1)) ** 0.5
    margin = 1.96 * sd / n ** 0.5
    verdict = ("better" if mean - margin > 0 else "worse" if mean + margin < 0 else "no clear difference")
    return {"variants": out, "difference": {"points": round(mean, 2), "margin": round(margin, 2),
                                            "verdict": verdict, "matches": n}}


# ---------------------------------------------------------------------------
# Matches (matches/M0001/...)
# ---------------------------------------------------------------------------

def match_dir(match_id: str) -> Path:
    return MATCHES_DIR / match_id


def list_matches() -> list[str]:
    """Saved match ids, newest first."""
    if not MATCHES_DIR.exists():
        return []
    ids = [d.name for d in MATCHES_DIR.iterdir() if _MATCH_ID.match(d.name) and (d / "events.json").exists()]
    return sorted(ids, key=lambda m: int(m[1:]), reverse=True)


def _next_match_id() -> str:
    used = [int(m[1:]) for m in list_matches()]
    return f"M{max(used, default=0) + 1:04d}"


def simulate(ctx: Context, team_a_id: str, team_b_id: str, seed: int | None = None,
             version_a: int | None = None, version_b: int | None = None) -> EventLog:
    """Play a 90-minute match between any saved versions (default: latest)
    and save its event log (with line-ups) and stats report. A fresh random
    seed is drawn unless one is given. Each side's chemistry feeds its duels."""
    if seed is None:
        seed = random.SystemRandom().randrange(1, 1_000_000)
    same = team_a_id == team_b_id
    if same and (version_a or storage.list_teams()[team_a_id]["latest_version"]) == \
            (version_b or storage.list_teams()[team_b_id]["latest_version"]):
        raise ValueError("pick two different teams, or two different versions of the same team")
    ta = _match_side(ctx, team_a_id, version_a, same)
    tb = _match_side(ctx, team_b_id, version_b, same)
    for t in (ta, tb):
        if missing_players(ctx, t):
            raise ValueError(f"{t.name} uses players from a different player pool — switch the pool back "
                             f"(sidebar) to play it")
    match_id = _next_match_id()
    log = simulate_match(match_id, ta, tb, ctx.lookup, seed=seed,
                         chemistry={ta.team_id: ta.chemistry_score, tb.team_id: tb.chemistry_score})
    log.played_at = datetime.now(UTC).isoformat(timespec="seconds")
    log.team_names = {t.team_id: t.name for t in (ta, tb)}
    log.team_versions = {t.team_id: t.version for t in (ta, tb)}
    log.lineups = {t.team_id: {s.slot_id: s.player_id for s in t.slots if s.player_id} for t in (ta, tb)}
    log.slot_positions = {t.team_id: {s.slot_id: s.position for s in t.slots} for t in (ta, tb)}
    log.formations = {t.team_id: t.formation for t in (ta, tb)}
    match_dir(match_id).mkdir(parents=True, exist_ok=True)
    (match_dir(match_id) / "events.json").write_text(log.model_dump_json(indent=2), encoding="utf-8")
    build_report(match_id)
    return log


def load_event_log(match_id: str) -> EventLog:
    path = match_dir(match_id) / "events.json"
    if not path.exists():
        raise KeyError(f"no match '{match_id}' found")
    return EventLog.model_validate(json.loads(path.read_text(encoding="utf-8")))


def matches_for_team(team_id: str) -> list[str]:
    out = []
    for m in list_matches():
        log = load_event_log(m)
        if team_id in (base_team_id(log.team_a), base_team_id(log.team_b)):
            out.append(m)
    return out


def delete_match(match_id: str) -> None:
    """Remove a match folder (event log, report, debrief)."""
    if not (match_dir(match_id) / "events.json").exists():
        raise KeyError(f"no match '{match_id}' found")
    shutil.rmtree(match_dir(match_id))


def delete_team(team_id: str) -> list[str]:
    """Remove a team, all its versions, and every match it played (those
    matches can't be reported on without it). Returns the deleted match ids."""
    matches = matches_for_team(team_id)
    storage.delete_team(team_id)
    for m in matches:
        delete_match(m)
    return matches


def delete_version(team_id: str, version: int) -> None:
    """Delete one saved version of a team. Matches it played keep their recorded
    line-ups, so they stay readable; its version number is never reused."""
    storage.delete_version(team_id, version)


def clear_all() -> None:
    """Delete every team, match and scouting report (the player pool stays)."""
    for tid in list(storage.list_teams()):
        delete_team(tid)
    for m in list_matches():
        delete_match(m)
    shutil.rmtree(SCOUTING_DIR, ignore_errors=True)


_together_cache: dict[tuple, Together] = {}


def together_counts() -> Together:
    """(player, player) -> matches they started together for the same team.
    Backs the "Familiarity" chemistry rule. Cached until a match is added or removed."""
    key = tuple((m, (match_dir(m) / "events.json").stat().st_mtime_ns) for m in list_matches())
    if key not in _together_cache:
        counts: Counter = Counter()
        for m, _ in key:
            for slots in load_event_log(m).lineups.values():
                for pair in combinations(sorted(set(slots.values())), 2):
                    counts[frozenset(pair)] += 1
        _together_cache.clear()
        _together_cache[key] = dict(counts)
    return _together_cache[key]


# ---------------------------------------------------------------------------
# Reports, debrief and feedback
# ---------------------------------------------------------------------------

def build_report(match_id: str) -> dict[str, PlayerReport]:
    """Stats, role-based ratings and role-relevant strengths/weaknesses, keyed by
    report_key(team, player) so a player who appears for both sides keeps two reports."""
    log = load_event_log(match_id)
    starters = {(tid, pid) for tid, slots in log.lineups.items() for pid in slots.values()}
    reports = build_reports(log, starters)
    (match_dir(match_id) / "report.json").write_text(
        json.dumps({k: r.model_dump() for k, r in reports.items()}, indent=2, ensure_ascii=False),
        encoding="utf-8")
    return reports


def load_report(match_id: str) -> dict[str, PlayerReport] | None:
    path = match_dir(match_id) / "report.json"
    if not path.exists():
        return None
    raw = json.loads(path.read_text(encoding="utf-8"))
    if any("|" not in k for k in raw):
        return build_report(match_id)  # older one-report-per-player format: rebuild per side
    return {k: PlayerReport.model_validate(r) for k, r in raw.items()}


def team_reports(reports: dict[str, PlayerReport], match_team_id: str) -> dict[str, PlayerReport]:
    """One side's reports, keyed by player id."""
    return {r.player_id: r for r in reports.values() if r.team_id == match_team_id}


def side_for(log: EventLog, team_id: str) -> str | None:
    """The match-side id ('t_A' or 't_A@v2') a saved team played as, if it played."""
    return next((t for t in (log.team_a, log.team_b) if base_team_id(t) == team_id), None)


def match_debrief(ctx: Context, match_id: str) -> dict:
    """The computed coach's debrief: {team_id: {summary, worked, didnt, changes}}."""
    names = {pid: p.name for pid, p in ctx.lookup.items()}
    return debrief(load_event_log(match_id), load_report(match_id) or {}, names)


def weakest_outfield_slot(team: Team, reports: dict[str, PlayerReport]) -> str | None:
    """`reports` = one side's reports keyed by player id (see team_reports)."""
    slots = [s for s in team.slots if s.position != "GK" and s.player_id in reports]
    return min(slots, key=lambda s: reports[s.player_id].rating).slot_id if slots else None


def feedback(ctx: Context, team_id: str, match_id: str, slot: str, side: str | None = None
             ) -> tuple[SwapRequest | None, Shortlist | None, str]:
    """Derive a swap request from how `slot`'s player performed for `team_id`
    in the match, and shortlist it. `side` picks the match side ('t_A@v2') when
    the team played itself. Returns (request, shortlist, message)."""
    reports = load_report(match_id)
    if reports is None:
        return None, None, f"No report for {match_id} yet."
    log = load_event_log(match_id)
    team = storage.load_team(team_id)
    occupant = next(s.player_id for s in team.slots if s.slot_id == slot)
    sides = [side] if side else [t for t in (log.team_a, log.team_b) if base_team_id(t) == team_id]
    report = next((reports.get(report_key(t, occupant)) for t in sides if report_key(t, occupant) in reports), None)
    if report is None:
        return None, None, f"{slot}'s current player didn't play for {team.name} in {match_id}."
    request = derive_swap_request(slot, occupant, report.stats, slot_position(team, slot))
    if request is None:
        return None, None, f"No clear weakness for {slot} in {match_id} — no swap suggested."
    return request, shortlist(ctx, team, request), request.rationale


# ---------------------------------------------------------------------------
# Players and teams over time
# ---------------------------------------------------------------------------

def player_history(player_id: str) -> list[dict]:
    """Every match side the player started for, newest first (two rows when he
    played for both sides of a team-vs-own-version match)."""
    rows = []
    for m in list_matches():
        log = load_event_log(m)
        reports = None
        for team_id in log.sides_of(player_id):
            reports = reports if reports is not None else (load_report(m) or {})
            opp = log.team_b if team_id == log.team_a else log.team_a
            rows.append({"match_id": m, "label": log.label, "team": log.name_v(team_id),
                         "opponent": log.name_v(opp),
                         "result": f"{log.final_score[team_id]}–{log.final_score[opp]}",
                         "slot": log.slot_of(player_id, team_id)[1],
                         "report": reports.get(report_key(team_id, player_id))})
    return rows


def team_history(ctx: Context, team_id: str, limit: int | None = None, pick: str = "last",
                 seed: int | None = None) -> dict:
    """The games a team played (any version; a match against its own older
    version counts once per side): results and, per player, appearances,
    average rating, last rating, trend and recurring weaknesses.

    `limit` keeps a long history cheap to review: the last N games
    (pick="last") or N at random (pick="random", reproducible with `seed`)."""
    games = []  # oldest first, so "last" is the most recent
    for m in reversed(list_matches()):
        log = load_event_log(m)
        sides = [s for s in (log.team_a, log.team_b) if base_team_id(s) == team_id]
        if sides:
            reports = load_report(m) or {}
            games += [(m, log, side, reports) for side in sides]
    total = len(games)
    if limit and total > limit:
        if pick == "random":
            games = [games[i] for i in sorted(random.Random(seed).sample(range(total), limit))]
        else:
            games = games[-limit:]
    matches, players = [], {}
    for m, log, side, reports in games:
        opp = log.team_b if side == log.team_a else log.team_a
        gf, ga = log.final_score[side], log.final_score[opp]
        stats = team_stats(log, reports)
        matches.append({"match_id": m, "version": log.team_versions.get(side), "opponent": log.name_v(opp),
                        "result": "W" if gf > ga else "L" if gf < ga else "D", "score": f"{gf}-{ga}",
                        "goals_for": gf, "goals_against": ga, "clean_sheet": ga == 0,
                        "xg_for": stats[side]["Expected goals (xG)"],
                        "xg_against": stats[opp]["Expected goals (xG)"]})
        for r in reports.values():
            if r.team_id != side:
                continue
            entry = players.setdefault(r.player_id, {"player_id": r.player_id,
                                                     "name": ctx.lookup[r.player_id].name
                                                     if r.player_id in ctx.lookup else r.player_id,
                                                     "ratings": [], "weaknesses": Counter()})
            entry["ratings"].append(r.rating)
            entry["weaknesses"].update(w for w in r.weaknesses if not w.startswith("No clear"))
    rows = []
    for e in players.values():
        ratings = e["ratings"]
        rows.append({"player_id": e["player_id"], "name": e["name"], "apps": len(ratings),
                     "avg_rating": round(sum(ratings) / len(ratings), 2), "last_rating": ratings[-1],
                     "trend": round(ratings[-1] - sum(ratings) / len(ratings), 2),
                     "recurring_weaknesses": [w for w, n in e["weaknesses"].most_common(2) if n >= 2]})
    record = Counter(m["result"] for m in matches)
    selected = bool(limit and total > limit)
    return {"team_id": team_id, "matches": matches, "record": {k: record.get(k, 0) for k in "WDL"},
            "totals": {"goals_for": sum(m["goals_for"] for m in matches),
                       "goals_against": sum(m["goals_against"] for m in matches),
                       "clean_sheets": sum(m["clean_sheet"] for m in matches)},
            "players": sorted(rows, key=lambda r: -r["avg_rating"]),
            "selection": {"pick": pick if selected else "all", "used": len(matches), "total": total}}


def position_fits(ctx: Context, player: Player) -> list[dict]:
    """Fit at every position using the base weights, best first."""
    positions = ["GK"] if player.is_gk() else [p for p in ctx.templates if p != "GK"]
    fits = [{"position": pos, "fit": fit_score(player, ctx.templates[pos]),
             "natural": pos in player.natural_positions} for pos in positions]
    return sorted(fits, key=lambda f: -f["fit"])


def player_profile(ctx: Context, player_id: str) -> dict:
    """Everything the UI/LLM needs about one player: attributes with
    percentiles vs players listed at the same position, fits per position,
    strengths/weaknesses (attributes that matter for the position) and form."""
    p = ctx.lookup[player_id]
    fits = position_fits(ctx, p)
    main = next((pos for pos in p.natural_positions if pos in ctx.templates), fits[0]["position"])
    relevant = set(ctx.templates[main].attribute_weights)
    attrs = []
    for a in ATTRIBUTE_NAMES:
        v = p.attributes.get(a)
        if v is None or (a in GK_ONLY and not p.is_gk()):
            continue
        attrs.append({"attribute": a, "value": v, "percentile": ctx.percentile(main, a, v),
                      "relevant": a in relevant})
    ranked = sorted((a for a in attrs if a["relevant"]), key=lambda a: -a["percentile"])
    # only call something a weakness if it's genuinely below the positional median
    weaknesses = [a for a in ranked[::-1] if a["percentile"] < 50][:3]
    rival = max((q for q in ctx.pool if main in q.natural_positions and q.player_id != player_id),
                key=lambda q: fit_score(q, ctx.templates[main]), default=None)
    history = player_history(player_id)
    reps = [h["report"] for h in history if h["report"]]
    form = {"matches": len(reps),
            "avg_rating": round(sum(r.rating for r in reps) / len(reps), 2) if reps else None,
            "goals": sum(r.stats.goals for r in reps), "key_passes": sum(r.stats.key_passes for r in reps),
            "recent_strengths": [s for r in reps[:3] for s in r.strengths][:4],
            "recent_weaknesses": [w for r in reps[:3] for w in r.weaknesses][:4]}
    return {
        "player_id": p.player_id, "name": p.name, "positions": p.natural_positions, "main_position": main,
        "foot": p.preferred_foot, "weak_foot": p.weak_foot_rating, "age": p.age, "nationality": p.nationality,
        "stamina_base": p.stamina_base, "source": p.source,
        "attributes": attrs, "best_positions": fits[:3], "position_fits": fits,
        "strengths": ranked[:3], "weaknesses": weaknesses, "lowest_relevant": ranked[-1] if ranked else None,
        "best_rival": rival.player_id if rival else None, "form": form,
    }


# Scouting criteria per position: (criterion, attributes, share of the overall score).
# A defender is never judged on finishing, a striker never on tackling.
SCOUTING_CRITERIA: dict[str, list[tuple[str, list[str], float]]] = {
    "GK": [("Shot-stopping", ["gk_reflexes"], 0.35), ("Handling", ["gk_handling"], 0.2),
           ("Positioning", ["gk_positioning"], 0.2), ("Distribution", ["gk_kicking", "passing_short",
                                                                        "passing_long"], 0.15),
           ("Composure", ["composure"], 0.1)],
    "CB": [("Defending", ["tackling_standing", "tackling_sliding", "marking"], 0.35),
           ("Reading the game", ["positioning"], 0.15), ("Aerial", ["heading_accuracy", "heading_power"], 0.2),
           ("Physique", ["pace", "acceleration", "aggression"], 0.15),
           ("Ball-playing", ["passing_short", "passing_long", "composure"], 0.15)],
    "FB": [("Defending", ["tackling_standing", "marking", "positioning"], 0.3),
           ("Athleticism", ["pace", "acceleration", "stamina"], 0.3),
           ("Attacking support", ["crossing", "dribbling", "passing_short"], 0.4)],
    "DM": [("Ball-winning", ["tackling_standing", "tackling_sliding", "marking", "aggression"], 0.35),
           ("Positioning", ["positioning"], 0.2), ("Distribution", ["passing_short", "passing_long", "vision"], 0.3),
           ("Engine", ["stamina"], 0.15)],
    "CM": [("Passing", ["passing_short", "passing_long"], 0.3), ("Creativity", ["vision", "through_ball"], 0.25),
           ("Ball-winning", ["tackling_standing"], 0.15), ("Engine", ["stamina"], 0.15),
           ("Composure", ["composure", "ball_control"], 0.15)],
    "AM": [("Creativity", ["vision", "through_ball"], 0.35), ("Technique", ["dribbling", "ball_control"], 0.25),
           ("Goal threat", ["finishing", "composure"], 0.25), ("Passing", ["passing_short"], 0.15)],
    "WING": [("Pace", ["pace", "acceleration"], 0.25), ("1v1", ["dribbling", "ball_control"], 0.3),
             ("Delivery", ["crossing"], 0.15), ("Goal threat", ["finishing", "composure"], 0.2),
             ("Engine", ["stamina"], 0.1)],
    "ST": [("Finishing", ["finishing", "composure"], 0.4), ("Movement", ["positioning"], 0.2),
           ("Aerial", ["heading_accuracy", "heading_power"], 0.15), ("Hold-up play", ["ball_control",
                                                                                     "passing_short"], 0.15),
           ("Pace", ["pace", "acceleration"], 0.1)],
}
FORM_MAX_SHARE = 0.3   # match form's share of the overall once it's fully established
FORM_FULL_AFTER = 5    # matches


def _age_profile(age: int) -> dict:
    if age <= 21:
        return {"stage": "prospect", "note": "Young enough to keep improving; expect some inconsistency."}
    if age <= 27:
        return {"stage": "prime years", "note": "At or entering his peak."}
    if age <= 31:
        return {"stage": "experienced", "note": "Proven, with a few peak years left."}
    return {"stage": "veteran", "note": "Short-term option; physical decline is a real risk."}


def _role_form(reports: list[PlayerReport], group: str) -> str:
    """The match numbers that matter for the role, summed over matches."""
    s = [r.stats for r in reports]
    if group == "GK":
        saves, conceded = sum(x.saves for x in s), sum(x.goals_conceded for x in s)
        return f"{saves} saves, {conceded} conceded" + (f" ({100 * saves / (saves + conceded):.0f}% saved)"
                                                       if saves + conceded else "")
    if group == "DEF":
        won, lost = sum(x.tackles_won for x in s), sum(x.tackles_lost for x in s)
        aw, al = sum(x.aerial_duels_won for x in s), sum(x.aerial_duels_lost for x in s)
        return f"duels {won}/{won + lost}, aerials {aw}/{aw + al}"
    if group == "MID":
        pa, pc = sum(x.passes_attempted for x in s), sum(x.passes_completed for x in s)
        return (f"{100 * pc / pa:.0f}% passing, " if pa else "") + f"{sum(x.key_passes for x in s)} chances created"
    return (f"{sum(x.goals for x in s)} goals from {sum(x.xg for x in s):.2f} xG, "
            f"{sum(x.key_passes for x in s)} chances created")


def scouting_assessment(ctx: Context, player_id: str) -> dict:
    """Deterministic, role-specific criteria scores (0-10) behind the scouting
    report. Score = average percentile of the criterion's attributes against
    players at the same position, /10. Overall = weighted by the role's
    criterion shares; match form joins once he has played, its share growing
    to 30% over his first 5 matches."""
    from .report_generator import role_group
    p = ctx.lookup[player_id]
    prof = player_profile(ctx, player_id)
    main = prof["main_position"]
    by_attr = {a["attribute"]: a for a in prof["attributes"]}
    criteria = []
    for name, attrs, share in SCOUTING_CRITERIA[main]:
        present = [by_attr[a] for a in attrs if a in by_attr]
        if not present:
            continue
        criteria.append({
            "name": name, "share": share,
            "score": round(sum(a["percentile"] for a in present) / len(present) / 10, 1),
            "detail": ", ".join(f"{a['attribute'].replace('_', ' ')} {a['value']:.0f} "
                                f"(beats {a['percentile']:.0f}% of {main}s)" for a in present),
        })
    form = prof["form"]
    history = [h["report"] for h in player_history(player_id) if h["report"]]
    if history:
        form_share = FORM_MAX_SHARE * min(len(history), FORM_FULL_AFTER) / FORM_FULL_AFTER
        for c in criteria:
            c["share"] = round(c["share"] * (1 - form_share), 3)
        criteria.append({"name": "Match form", "share": round(form_share, 3), "score": form["avg_rating"],
                         "detail": f"average rating {form['avg_rating']:.1f} over {len(history)} match(es) · "
                                   + _role_form(history, role_group(main))})
    overall = round(sum(c["score"] * c["share"] for c in criteria) / sum(c["share"] for c in criteria), 1)
    age = _age_profile(p.age)
    if overall >= 8 and p.age <= 31:
        verdict, rule = "Sign", "Overall 8+/10 for his position and not yet in decline."
    elif overall >= 8:
        verdict, rule = "Short-term signing", "Elite for his position, but aged 32+."
    elif overall >= 6:
        verdict, rule = "Monitor", "Above average for his position, not yet a clear upgrade."
    else:
        verdict, rule = "Pass", "Below the positional average on the criteria that matter for the role."
    best = prof["best_positions"][0]
    roles = [f"{f} {role_name(ctx, f, best['position'])}" for f in ctx.formation_weights
             if best["position"] in formation_positions(ctx, f)]
    return {
        "player_id": player_id, "name": p.name, "age": p.age, "foot": p.preferred_foot,
        "weak_foot": p.weak_foot_rating, "positions": p.natural_positions, "main_position": main,
        "best_role": best["position"] + (f" ({' / '.join(roles)})" if roles else ""),
        "position_fits": prof["best_positions"], "criteria": criteria, "overall": overall,
        "age_profile": age, "verdict": verdict, "verdict_rule": rule, "form": form,
    }


# ---------------------------------------------------------------------------
# The assistant's written work (see agent.py), saved so it isn't paid for twice
# ---------------------------------------------------------------------------

def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def _read_json(path: Path) -> dict | None:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def _write_json(path: Path, record: dict) -> dict:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record, indent=2, ensure_ascii=False), encoding="utf-8")
    return record


def _scouting_path(player_id: str, team_id: str | None) -> Path:
    return SCOUTING_DIR / (f"{player_id}@{team_id}.json" if team_id else f"{player_id}.json")


def load_scouting(player_id: str, team_id: str | None = None) -> dict | None:
    record = _read_json(_scouting_path(player_id, team_id))
    return record if record and "analysis" in record else None  # ignore older formats


def generate_scouting(ctx: Context, player_id: str, model: str | None = None,
                      tracker: CostTracker | None = None, team_id: str | None = None) -> dict:
    """The assistant scouts the player on his own merits (tools + verification),
    on top of the deterministic criteria assessment. With `team_id`, it also
    tests him in that team. Saved per player (and team)."""
    from . import agent
    assessment = scouting_assessment(ctx, player_id)
    analysis = agent.scout_player(ctx, player_id, model=model, tracker=tracker, team_id=team_id)
    return _write_json(_scouting_path(player_id, team_id), {
        "player_id": player_id, "team_id": team_id, "assessment": assessment, "analysis": analysis,
        "source": analysis["source"], "matches_seen": assessment["form"]["matches"], "generated_at": _now()})


def load_match_analysis(match_id: str) -> dict | None:
    record = _read_json(match_dir(match_id) / "analysis.json")
    return record if record and "headline" in record else None  # ignore the older debrief format


def generate_match_analysis(ctx: Context, match_id: str, model: str | None = None,
                            tracker: CostTracker | None = None) -> dict:
    from . import agent
    analysis = agent.analyse_match(ctx, match_id, model=model, tracker=tracker)
    return _write_json(match_dir(match_id) / "analysis.json", {**analysis, "generated_at": _now()})


def load_team_review(team_id: str) -> dict | None:
    return _read_json(storage.TEAMS_DIR / team_id / "review.json")


def generate_team_review(ctx: Context, team_id: str, model: str | None = None,
                         tracker: CostTracker | None = None, limit: int | None = None,
                         pick: str = "last") -> dict:
    """The assistant reviews the team's form over all its games, or only the
    last / a random `limit` of them (keeps a long history cheap)."""
    from . import agent
    seed = random.SystemRandom().randrange(1, 1_000_000) if pick == "random" else None
    review = agent.review_team(ctx, team_id, model=model, tracker=tracker, limit=limit, pick=pick, seed=seed)
    scope = team_history(ctx, team_id, limit, pick, seed)
    return _write_json(storage.TEAMS_DIR / team_id / "review.json", {
        **review, "matches_seen": scope["selection"]["total"],
        "scope": {**scope["selection"], "limit": limit, "seed": seed,
                  "match_ids": sorted({m["match_id"] for m in scope["matches"]})},
        "generated_at": _now()})


def best_fits(ctx: Context, weights: dict[str, float], goalkeepers: bool = False,
              filters: list[HardFilter] | None = None, top_n: int = 10,
              exclude: set[str] | None = None) -> list[dict]:
    """Rank the pool (keepers or outfielders) by fit to any `weights`
    (normalised here). Players breaking a filter are kept but listed after
    everyone who meets them. Returns [{player_id, fit, issues}]."""
    from .scoring import check_hard_filters
    w = normalize_weights({"x": weights})["x"]
    template = ctx.templates["GK" if goalkeepers else "CM"].model_copy(
        update={"attribute_weights": w, "soft_bonuses": []})
    rows = []
    for p in ctx.pool:
        if (exclude and p.player_id in exclude) or p.is_gk() != goalkeepers:
            continue
        rows.append({"player_id": p.player_id, "fit": fit_score(p, template),
                     "issues": check_hard_filters(p, filters or [])})
    rows.sort(key=lambda r: (bool(r["issues"]), -r["fit"]))
    return rows[:top_n]
