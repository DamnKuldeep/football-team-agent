"""The AI assistant: a tool-using model that investigates the real data,
proposes, and explains -- while plain code verifies everything it says.

Four jobs, each a short agent loop (model -> tool calls -> model -> ... -> JSON):
  recommend_changes  a goal from the coach ("we lose in the air") -> up to 3 changes
  analyse_match      a played match -> key factors, player notes, tested fixes
  scout_player       one player -> strengths, risks, rivals, fit for your teams
  review_team        a team's whole match history -> trends and tested fixes

Tools are read-only and deterministic (they call the same pipeline functions
the UI uses). After the model answers, `verify` checks every id, slot and
requirement, insists every proposed change was tested with evaluate_changes,
and fact-checks the text: each number must appear in the data the tools
returned, each player name must be real. Problems go back to the model to fix
(up to 2 rounds); anything still wrong is dropped or flagged. The impact shown
is always the code's own measurement (the same test the model saw), never a
number the model wrote. With no API key, or if the model fails, a
deterministic version of each job runs instead -- clearly labelled.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field

import requests

from . import pipeline, storage
from .config import llm_model, make_tracker
from .cost_tracker import CostTracker
from .llm_client import (
    GK_ATTRS,
    GLOSSARY,
    OUTFIELD_ATTRS,
    _api_key,
    _friendly,
    _grounded,
    chat_message,
    strict_limit,
)
from .models import HardFilter
from .report_generator import ROLE_LABEL, role_group, team_insights, team_stats
from .scoring import check_hard_filters, describe_filter, fit_score

MAX_STEPS = 10     # model turns per job; the last one must be the answer
MAX_FIXES = 2
EVAL_MATCHES = 300   # paired simulations per measured change set

# ---------------------------------------------------------------------------
# Tool definitions (what the model sees)
# ---------------------------------------------------------------------------

_REQ_SCHEMA = {"type": "array", "description": "Hard requirements. field: age | nationality | natural_position | "
               "weak_foot_rating | preferred_foot | <attribute>. Numeric fields use min/max; nationality may list "
               "several values (any of them).",
               "items": {"type": "object", "properties": {
                   "field": {"type": "string"}, "min": {"type": "number"}, "max": {"type": "number"},
                   "value": {"type": "string"}, "values": {"type": "array", "items": {"type": "string"}}},
                   "required": ["field"]}}
_CHANGES_SCHEMA = {"type": "array", "items": {"type": "object", "properties": {
    "slot": {"type": "string"}, "player_id": {"type": "string"}}, "required": ["slot", "player_id"]}}


def _tool(name: str, description: str, properties: dict, required: list[str] | None = None) -> dict:
    return {"type": "function", "function": {"name": name, "description": description, "parameters": {
        "type": "object", "properties": properties, "required": required or []}}}


TOOLS = [
    _tool("get_team", "A saved team: formation, every slot's player (id, name, age, foot, nationality, fit, key "
          "attributes), chemistry rules with points, the team's position weights, and the other teams (possible "
          "opponents).", {"team_id": {"type": "string"}, "version": {"type": "integer"}}),
    _tool("search_players", "Rank the player pool for a position. `emphasis` raises attribute weights (0.05-0.5) on "
          "top of the team's weights for that position; `requirements` are hard filters (players who miss one are "
          "listed after those who meet all). Excludes players already in the team.",
          {"position": {"type": "string", "enum": ["GK", "CB", "FB", "DM", "CM", "AM", "WING", "ST"]},
           "emphasis": {"type": "object", "additionalProperties": {"type": "number"}},
           "requirements": _REQ_SCHEMA, "team_id": {"type": "string"},
           "limit": {"type": "integer", "minimum": 1, "maximum": 10}}, ["position"]),
    _tool("get_player", "One player: age, foot, nationality, natural positions, attributes with percentile vs his "
          "position, fit at each position, scouting scores and verdict, and match form.",
          {"player_id": {"type": "string"}}, ["player_id"]),
    _tool("evaluate_changes", "Test a set of changes to a team WITHOUT saving: new average fit and chemistry, fit "
          "per changed slot, and a paired simulation vs an opponent (same random seeds for both line-ups), "
          "reported as points per match with a 95% margin and a verdict.",
          {"changes": _CHANGES_SCHEMA, "team_id": {"type": "string"}, "opponent_team_id": {"type": "string"}},
          ["changes"]),
    _tool("team_history", "Every match a team has played (any version): results, xG, and per player appearances, "
          "average rating, last rating, trend and recurring weaknesses.", {"team_id": {"type": "string"}}),
    _tool("get_match", "A played match: score, goals, team stats, team strengths/weaknesses and every player's "
          "rating with evidence, per side.", {"match_id": {"type": "string"}}, ["match_id"]),
]

TOOL_NAMES = {t["function"]["name"] for t in TOOLS}

TOOL_GUIDE = """Working rules:
- Investigate with the tools before answering; never guess ids, slots, names or numbers.
- Call several tools in one turn when you can: you have a limited number of turns.
- Refer to players by the player_id the tools return; use names exactly as the tools give them.
- Every number you write must come from a tool result (you may round to what the tool shows).
- Before proposing any change, call evaluate_changes on your final set and base your judgement on it;
  compare players with its attributes_in_minus_out (positive = the new player is better).
- When you are done, reply with ONLY the JSON object described below (no tool call, no prose around it)."""


# ---------------------------------------------------------------------------
# Tool implementations (what the code does)
# ---------------------------------------------------------------------------

@dataclass
class Toolbox:
    ctx: pipeline.Context
    team_id: str | None = None
    trace: list[dict] = field(default_factory=list)
    numbers: list[float] = field(default_factory=list)
    evaluated: dict[tuple, dict] = field(default_factory=dict)   # (team, change set) -> its measurement
    match_options: bool = False   # get_match also tests replacements (only for the match being analysed)
    history_scope: dict = field(default_factory=dict)   # team_history(limit, pick, seed) for self.team_id
    allowed_matches: set[str] | None = None             # get_match only for these (a limited review)
    _matches: dict[str, dict] = field(default_factory=dict, repr=False)

    def tested(self, team_id: str, changes: dict[str, str]) -> dict | None:
        """The measurement the model saw for exactly this change set, if it ran one."""
        return self.evaluated.get((team_id, frozenset(changes.items())))

    def tested_within(self, team_id: str, changes: dict[str, str]) -> bool:
        """Was this change tested, alone or as part of a larger set?"""
        wanted = set(changes.items())
        return any(t == team_id and wanted <= key for t, key in self.evaluated)

    def call(self, name: str, args: dict) -> dict:
        fn = getattr(self, f"_{name}") if name in TOOL_NAMES else None
        try:
            result = fn(**args) if fn else {"error": f"unknown tool {name!r}"}
        except (KeyError, ValueError, TypeError, FileNotFoundError, StopIteration) as e:
            result = {"error": str(e) or type(e).__name__}
        self._remember(result)
        self.trace.append({"tool": name, "args": args, "result": _summarise(result)})
        return result

    def _remember(self, obj) -> None:
        """Collect every number a tool returned (and list sizes, so "5 matches" checks out)."""
        if isinstance(obj, bool):
            return
        if isinstance(obj, (int, float)):
            self.numbers.append(float(obj))
        elif isinstance(obj, dict):
            for v in obj.values():
                self._remember(v)
        elif isinstance(obj, list):
            self.numbers.append(float(len(obj)))
            for v in obj:
                self._remember(v)
        elif isinstance(obj, str):
            for n in re.findall(r"-?\d+(?:\.\d+)?", obj):
                self.numbers.append(float(n))

    # -- tools ---------------------------------------------------------------

    def _team(self, team_id: str | None, version: int | None = None):
        tid = team_id or self.team_id
        if not tid:
            raise ValueError("no team_id given")
        return pipeline.rescored(self.ctx, storage.load_team(tid, version))

    def _get_team(self, team_id: str | None = None, version: int | None = None) -> dict:
        ctx, team = self.ctx, self._team(team_id, version)
        templates = pipeline.team_templates(ctx, team)
        _, rules = pipeline.chemistry_details(ctx, team)
        lineup = []
        for s in team.slots:
            p = ctx.lookup.get(s.player_id)
            if not p:
                continue
            weights = templates[s.position].attribute_weights
            lineup.append({"slot": s.slot_id, "position": s.position,
                           "role": pipeline.role_name(ctx, team.formation, s.position), "player_id": p.player_id,
                           "name": p.name, "age": p.age, "foot": p.preferred_foot, "nationality": p.nationality,
                           "fit": round(fit_score(p, templates[s.position]), 1),
                           "key_attributes": {a: p.attr(a) for a in sorted(weights, key=lambda a: -weights[a])[:4]}})
        others = {t: v["name"] for t, v in storage.list_teams().items() if t != team.team_id}
        return {"team_id": team.team_id, "name": team.name, "version": team.version, "formation": team.formation,
                "avg_fit": team.avg_fit_score, "chemistry": team.chemistry_score,
                "chemistry_rules": [{"rule": r["label"], "points": r["points"], "detail": r["detail"]} for r in rules],
                "lineup": lineup, "weights": pipeline.team_weights(ctx, team), "other_teams": others}

    def _search_players(self, position: str, emphasis: dict | None = None, requirements: list | None = None,
                        team_id: str | None = None, limit: int = 8) -> dict:
        ctx = self.ctx
        tid = team_id or self.team_id
        team = self._team(tid) if tid else None
        base = (pipeline.team_weights(ctx, team).get(position) if team else None) or \
            ctx.templates[position].attribute_weights
        allowed = GK_ATTRS if position == "GK" else OUTFIELD_ATTRS
        emphasis = {a: min(0.5, max(0.05, float(w))) for a, w in (emphasis or {}).items() if a in allowed}
        weights = pipeline.normalize_weights({"x": {**base, **emphasis}})["x"]
        filters = _filters_from(requirements)
        exclude = set(team.player_ids()) if team else set()
        template = ctx.templates[position].model_copy(update={"attribute_weights": weights, "soft_bonuses": []})
        rows = []
        for p in ctx.pool:
            if p.player_id in exclude or p.is_gk() != (position == "GK"):
                continue
            issues = check_hard_filters(p, filters)
            rows.append((bool(issues), -fit_score(p, template), p, issues))
        rows.sort(key=lambda r: (r[0], r[1]))
        top = sorted(weights, key=lambda a: -weights[a])[:4]
        return {"position": position, "weights_used": weights,
                "requirements": [describe_filter(f) for f in filters],
                "candidates": [{"player_id": p.player_id, "name": p.name, "age": p.age, "foot": p.preferred_foot,
                                "nationality": p.nationality, "natural_positions": p.natural_positions,
                                "fit": round(-negfit, 1), "meets_requirements": not issues, "issues": issues,
                                "key_attributes": {a: p.attr(a) for a in top}}
                               for _, negfit, p, issues in rows[:max(1, min(int(limit), 10))]]}

    def _get_player(self, player_id: str) -> dict:
        ctx = self.ctx
        if player_id not in ctx.lookup:
            raise KeyError(f"no player {player_id!r} in the pool")
        prof = pipeline.player_profile(ctx, player_id)
        a = pipeline.scouting_assessment(ctx, player_id)
        teams = [f"{v['name']} ({t}) {s.slot_id}" for t, v in storage.list_teams().items()
                 for s in storage.load_team(t).slots if s.player_id == player_id]
        return {"player_id": player_id, "name": prof["name"], "age": prof["age"], "foot": prof["foot"],
                "weak_foot": prof["weak_foot"], "nationality": prof["nationality"],
                "natural_positions": prof["positions"], "main_position": prof["main_position"],
                "attributes": {x["attribute"]: {"value": x["value"], "beats_pct": x["percentile"]}
                               for x in prof["attributes"]},
                "fit_by_position": {f["position"]: f["fit"] for f in prof["position_fits"]},
                "roles_by_formation": {f: pipeline.role_name(ctx, f, prof["main_position"])
                                       for f in ctx.formation_weights
                                       if prof["main_position"] in pipeline.formation_positions(ctx, f)},
                "scouting": {"overall": a["overall"], "verdict": a["verdict"],
                             "criteria": {c["name"]: c["score"] for c in a["criteria"]}},
                "form": {k: prof["form"][k] for k in ("matches", "avg_rating", "goals", "key_passes",
                                                      "recent_strengths", "recent_weaknesses")},
                "in_your_teams": teams}

    def _evaluate_changes(self, changes: list, team_id: str | None = None,
                          opponent_team_id: str | None = None) -> dict:
        changes, tid = _changes_dict(changes), team_id or self.team_id
        result = measure(self.ctx, tid, changes, opponent_team_id)
        if "error" not in result:
            self.evaluated[(tid, frozenset(changes.items()))] = result
        return result

    def _team_history(self, team_id: str | None = None) -> dict:
        tid = team_id or self.team_id
        if not tid:
            raise ValueError("no team_id given")
        return pipeline.team_history(self.ctx, tid, **(self.history_scope if tid == self.team_id else {}))

    def _get_match(self, match_id: str) -> dict:
        if self.allowed_matches is not None and match_id not in self.allowed_matches:
            raise ValueError(f"{match_id} isn't one of the matches selected for this review "
                             f"({', '.join(sorted(self.allowed_matches))})")
        if match_id in self._matches:  # its replacement options are simulated: build once per job
            return self._matches[match_id]
        self._matches[match_id] = self._build_match(match_id)
        return self._matches[match_id]

    def _build_match(self, match_id: str) -> dict:
        ctx = self.ctx
        log = pipeline.load_event_log(match_id)
        reports = pipeline.load_report(match_id) or pipeline.build_report(match_id)
        stats, insights = team_stats(log, reports), team_insights(log, reports)
        names = {pid: p.name for pid, p in ctx.lookup.items()}
        goals = {log.team_a: [], log.team_b: []}
        for p in log.possessions:
            shot = next((e for e in p.chain if e.event == "shot" and e.outcome == "goal"), None)
            if shot:
                goals[p.attacking_team].append({"minute": p.minute or p.possession_id,
                                                "scorer": names.get(shot.player_id, shot.player_id),
                                                "xg": shot.xg})
        return {"match_id": match_id, "label": log.label, "sides": {tid: {
            "team": log.name_v(tid), "saved_team_id": pipeline.base_team_id(tid), "formation": log.formations.get(tid),
            "goals_for": log.final_score[tid], "goals": goals[tid], "stats": stats[tid], **insights[tid],
            "players": [{"player_id": r.player_id, "name": names.get(r.player_id, r.player_id), "slot": r.slot_id,
                         "role": ROLE_LABEL[role_group(r.position)], "rating": r.rating,
                         "strengths": r.strengths, "weaknesses": r.weaknesses}
                        for r in sorted(reports.values(), key=lambda r: -r.rating) if r.team_id == tid],
            **({"replacement_options": self._replacement_options(match_id, tid, reports)}
               if self.match_options else {})}
            for tid in (log.team_a, log.team_b)}}

    def _replacement_options(self, match_id: str, side: str, reports: dict, limit: int = 2) -> list[dict]:
        """For the side's players with a clear weakness (worst first): the top
        performance-derived candidate, already tested with evaluate_changes."""
        base = pipeline.base_team_id(side)
        if base not in storage.list_teams():
            return []
        options = []
        for r in sorted((r for r in reports.values() if r.team_id == side), key=lambda r: r.rating):
            if len(options) >= limit:
                break
            request, short, _ = pipeline.feedback(self.ctx, base, match_id, r.slot_id, side)
            if not (request and short and short.candidates):
                continue
            pick = short.candidates[0].player_id
            tested = self._evaluate_changes([{"slot": r.slot_id, "player_id": pick}], team_id=base)
            options.append({"team_id": base, "slot": r.slot_id, "current": self.ctx.lookup[r.player_id].name,
                            "why": request.rationale, "player_id": pick, "name": self.ctx.lookup[pick].name,
                            "tested": {k: tested.get(k) for k in ("avg_fit_change", "points_delta", "margin",
                                                                   "verdict", "opponent")}})
        return options


def _summarise(result: dict) -> str:
    """One line for the 'how it got there' trace."""
    if "error" in result:
        return f"error: {result['error']}"
    if "candidates" in result:
        return f"{len(result['candidates'])} candidates, best {result['candidates'][0]['name']} " \
               f"(fit {result['candidates'][0]['fit']})" if result["candidates"] else "no candidates"
    if "verdict" in result:
        return f"fit {result['avg_fit_before']} → {result['avg_fit_after']}, " + (
            f"{result['points_delta']:+.2f} pts/match ± {result['margin']:.2f} ({result['verdict']})"
            if result.get("points_delta") is not None else "no opponent to simulate")
    if "lineup" in result:
        return f"{result['name']} v{result['version']}: fit {result['avg_fit']}, chemistry {result['chemistry']}"
    if "scouting" in result:
        return f"{result['name']}: {result['scouting']['verdict']} ({result['scouting']['overall']}/10)"
    if "record" in result:
        r = result["record"]
        return f"{len(result['matches'])} matches: {r['W']}W {r['D']}D {r['L']}L"
    if "sides" in result:
        return result["label"]
    return "ok"


def _filters_from(requirements: list | None) -> list[HardFilter]:
    out = []
    for r in requirements or []:
        if not isinstance(r, dict) or not r.get("field"):
            continue
        field_name = r["field"]
        if field_name not in ("age", "nationality", "natural_position", "weak_foot_rating", "preferred_foot") \
                and field_name not in GLOSSARY:
            raise ValueError(f"unknown requirement field {field_name!r}")
        values = r.get("values") or (r["value"] if isinstance(r.get("value"), list) else None)
        out.append(HardFilter(attribute=field_name, min=r.get("min"), max=r.get("max"), values=values,
                              value=None if values else r.get("value")))
    return out


def _changes_dict(changes) -> dict[str, str]:
    if isinstance(changes, dict):
        return {str(k): str(v) for k, v in changes.items()}
    return {str(c["slot"]): str(c["player_id"]) for c in changes or [] if isinstance(c, dict)}


# ---------------------------------------------------------------------------
# Verification (plain code, never the model)
# ---------------------------------------------------------------------------

def check_changes(ctx: pipeline.Context, team, changes: dict[str, str],
                  requirements: list[HardFilter] | None = None) -> list[str]:
    """Every problem with a proposed change set: unknown slot or player,
    keeper/outfield mismatch, duplicates, player already in the team, unmet requirement."""
    problems = []
    slots = {s.slot_id: s for s in team.slots}
    current = set(team.player_ids())
    for slot, pid in changes.items():
        if slot not in slots:
            problems.append(f"{slot}: no such slot in {team.name} ({', '.join(slots)})")
            continue
        p = ctx.lookup.get(pid)
        if p is None:
            problems.append(f"{slot}: player {pid!r} doesn't exist")
            continue
        if (slots[slot].position == "GK") != p.is_gk():
            problems.append(f"{slot}: {p.name} is {'a keeper' if p.is_gk() else 'an outfield player'}")
        if slots[slot].player_id == pid:
            problems.append(f"{slot}: {p.name} already plays there — not a change")
        elif pid in current:
            problems.append(f"{slot}: {p.name} is already in the team")
        for issue in check_hard_filters(p, requirements or []):
            problems.append(f"{slot}: {p.name} — {issue}")
    if len(set(changes.values())) != len(changes):
        problems.append("the same player is proposed for two slots")
    return problems


def measure(ctx: pipeline.Context, team_id: str | None, changes: dict[str, str],
            opponent_team_id: str | None = None, n: int | None = None) -> dict:
    """Code-measured impact of a change set (the numbers the UI shows)."""
    n = n or EVAL_MATCHES
    if not team_id:
        raise ValueError("no team_id given")
    team = pipeline.rescored(ctx, storage.load_team(team_id))
    problems = check_changes(ctx, team, changes)
    if problems:
        return {"error": "invalid changes: " + "; ".join(problems)}
    after = pipeline.preview_swaps(ctx, team, changes)
    templates = pipeline.team_templates(ctx, team)
    slots = {s.slot_id: s for s in team.slots}
    per_change = []
    for slot, pid in changes.items():
        old, new, template = ctx.lookup.get(slots[slot].player_id), ctx.lookup[pid], templates[slots[slot].position]
        per_change.append({"slot": slot, "out": old.name if old else None, "in": new.name, "in_player_id": pid,
                           "fit_before": round(fit_score(old, template), 1) if old else None,
                           "fit_after": round(fit_score(new, template), 1),
                           "fit_change": round(fit_score(new, template) - fit_score(old, template), 1) if old else None,
                           "attributes_in_minus_out": _attribute_deltas(old, new, template) if old else {}})
    result = {"changes": per_change, "avg_fit_before": team.avg_fit_score, "avg_fit_after": after.avg_fit_score,
              "avg_fit_change": round(after.avg_fit_score - team.avg_fit_score, 2),
              "chemistry_before": team.chemistry_score, "chemistry_after": after.chemistry_score,
              "chemistry_change": round(after.chemistry_score - team.chemistry_score, 2),
              "opponent": None, "points_delta": None, "margin": None, "verdict": "not simulated"}
    opponents = [t for t in storage.list_teams() if t != team_id]
    opp_id = opponent_team_id if opponent_team_id in opponents else (opponents[0] if opponents else None)
    if opp_id and changes:
        opp = pipeline.rescored(ctx, storage.load_team(opp_id))
        if not pipeline.missing_players(ctx, opp):
            res = pipeline.evaluate_lineups(ctx, {"current": team, "with_changes": after}, opp, n=n)
            d = res["difference"]
            result.update(opponent=opp.name, matches=n, points_delta=d["points"], margin=d["margin"],
                          verdict=d["verdict"],
                          points_before=res["variants"]["current"]["Points per match"],
                          points_after=res["variants"]["with_changes"]["Points per match"])
    return result


def _attribute_deltas(old, new, template) -> dict[str, dict]:
    """The attributes that matter for the slot plus any big difference:
    {attr: {"out": 90, "in": 88, "change": -2}}, largest first."""
    attrs = [a for a in new.attributes if new.attributes[a] is not None and old.attributes.get(a) is not None]
    top = set(sorted(template.attribute_weights, key=lambda a: -template.attribute_weights[a])[:6])
    rows = {a: {"out": old.attr(a), "in": new.attr(a), "change": round(new.attr(a) - old.attr(a))}
            for a in attrs if a in top or abs(new.attr(a) - old.attr(a)) >= 8}
    return dict(sorted(rows.items(), key=lambda kv: -abs(kv[1]["change"]))[:10])


_ALWAYS_OK = {0.0, 1.0, 2.0, 3.0, 11.0, 60.0, 90.0, 100.0}
_NUMBER = re.compile(r"(?<![\w.])[-+−]?\d+(?:\.\d+)?")
_FULL_NAME = re.compile(r"\b([A-Z][a-zà-ž]+)\s+(?:[A-Z]\.\s+)?([A-Z][\w-]+)")


def fact_check(texts: list[str], toolbox: Toolbox, known_names: set[str], first_names: set[str]) -> list[str]:
    """Sentences with a number that no tool returned (rounding and % of a
    fraction allowed), or that name a player who doesn't exist."""
    facts = [abs(f) for f in toolbox.numbers]
    problems = []
    for text in texts:
        for sentence in re.split(r"(?<=[.!?])\s+", text or ""):
            for raw in _NUMBER.findall(sentence):
                x = abs(float(raw.lstrip("+−-")))
                tol = 0.051 if "." in raw else 0.5
                if x not in _ALWAYS_OK and not any(abs(x - f) <= tol or abs(x - 100 * f) <= tol for f in facts):
                    problems.append(f"“{sentence.strip()}” — {raw} isn't in the data the tools returned")
                    break
            for m in _FULL_NAME.finditer(sentence):
                if m.group(1) in first_names and m.group(0) not in known_names:
                    problems.append(f"“{sentence.strip()}” — no player called {m.group(0)}")
                    break
    return problems


def _names(ctx: pipeline.Context) -> tuple[set[str], set[str]]:
    """(every real player and team name, the players' first names)."""
    names = {p.name for p in ctx.pool}
    teams = {v["name"] for v in storage.list_teams().values()}
    return names | teams, {n.split()[0] for n in names if " " in n}


# ---------------------------------------------------------------------------
# The agent loop
# ---------------------------------------------------------------------------

def run_agent(system: str, brief: dict, toolbox: Toolbox, verify, model: str,
              tracker: CostTracker) -> dict:
    """Model <-> tools until a final JSON answer passes `verify` (or fixes run out).
    `verify(answer) -> (cleaned_answer, problems)`. Returns {"output", "problems",
    "error", "trace", "steps"}; output is None if the model never produced a usable answer."""
    messages = [{"role": "system", "content": system}, {"role": "user", "content": json.dumps(brief)}]
    fixes, last_output, last_problems = 0, None, []
    for step in range(MAX_STEPS):
        if tracker.blocked:
            return {"output": last_output, "problems": last_problems, "error": tracker.block_reason,
                    "trace": toolbox.trace, "steps": step}
        last = step == MAX_STEPS - 1
        if step == MAX_STEPS - 3:
            messages.append({"role": "user", "content": "Two turns left: finish your checks and answer."})
        elif last:
            messages.append({"role": "user", "content": "No more tool calls: reply now with ONLY the final JSON."})
        try:
            message, tin, tout = chat_message(model, messages, tools=TOOLS, max_tokens=1600,
                                              tool_choice="none" if last else "auto")
        except (requests.RequestException, KeyError, IndexError) as e:
            return {"output": last_output, "problems": last_problems, "error": _friendly(e),
                    "trace": toolbox.trace, "steps": step}
        tracker.log_call("assistant", model, tin, tout)
        calls = message.get("tool_calls") or []
        if calls:
            if (thought := (message.get("content") or "").strip()):
                toolbox.trace.append({"thought": thought})
            messages.append({"role": "assistant", "content": message.get("content") or "", "tool_calls": calls})
            for call in calls:
                name = call.get("function", {}).get("name", "")
                try:
                    args = json.loads(call["function"].get("arguments") or "{}")
                    result = toolbox.call(name, args if isinstance(args, dict) else {})
                except json.JSONDecodeError:
                    result = {"error": "the arguments weren't valid JSON"}
                messages.append({"role": "tool", "tool_call_id": call.get("id", ""), "name": name,
                                 "content": json.dumps(result)[:14000]})
            continue
        text = (message.get("content") or "").strip()
        cleaned = text.removeprefix("```json").removeprefix("```").removesuffix("```").strip()
        try:
            answer = json.loads(cleaned[cleaned.find("{"):cleaned.rfind("}") + 1])
        except ValueError:
            answer = None
        if not isinstance(answer, dict):
            messages += [{"role": "assistant", "content": text or "(empty)"},
                         {"role": "user", "content": "Continue: call the tools you need, or reply with ONLY the "
                                                     "final JSON object."}]
            continue
        last_output, last_problems = verify(answer)
        if not last_problems or fixes >= MAX_FIXES:
            return {"output": last_output, "problems": last_problems, "error": None, "trace": toolbox.trace,
                    "steps": step + 1}
        fixes += 1
        messages += [{"role": "assistant", "content": text},
                     {"role": "user", "content": "The code checked your answer and found problems:\n- "
                      + "\n- ".join(last_problems)
                      + "\nUse the tools to check the facts, fix every problem, and reply with the complete "
                        "corrected JSON."}]
    return {"output": last_output, "problems": last_problems, "error": "it ran out of steps", "trace": toolbox.trace,
            "steps": MAX_STEPS}


def _finish(result: dict, offline, model: str, toolbox: Toolbox) -> dict:
    """The model's verified answer, or -- if it never produced one -- the
    deterministic version (`offline` is only called then)."""
    if result.get("output") is None:
        return {**offline(), "source": "offline", "error": result.get("error"), "problems": [],
                "trace": toolbox.trace}
    return {**result["output"], "source": f"llm:{model}", "error": result.get("error"),
            "problems": result["problems"], "trace": toolbox.trace}


_TREND = re.compile(r"\b(recurring|repeated(ly)?|consistent(ly)?|keeps|again|habit(ual)?|trend|"
                    r"every match|in (every|each|\d+) (match|game)(es|s)?|lately|recently|form)\b", re.IGNORECASE)


def trend_check(texts: list[str], toolbox: Toolbox) -> list[str]:
    """A claim about a pattern over time needs the history behind it."""
    if any(t.get("tool") == "team_history" for t in toolbox.trace):
        return []
    for text in texts:
        if m := _TREND.search(text or ""):
            return [(f"“{text.strip()[:120]}” talks about a trend (“{m.group(0)}”) — check team_history first, "
                     "or describe this match only")]
    return []


_COUNT_WORDS = {"no": 0, "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7,
                "eight": 8, "nine": 9, "ten": 10}


def record_check(texts: list[str], history: dict) -> list[str]:
    """Claims about results ("clean sheets in both", "unbeaten", "3 clean sheets",
    "won every game") must match the record of the games reviewed -- the
    number check can't see these, as they often carry no number."""
    res = [m["result"] for m in history["matches"]]
    n, cs = len(res), sum(m["goals_against"] == 0 for m in history["matches"])
    problems = []
    for text in texts:
        t = (text or "").lower()
        wrong = []
        if re.search(r"clean sheets? in (all|both|every)|every (match|game)[^.]{0,30}clean sheet", t) and cs != n:
            wrong.append(f"only {cs} of the {n} games were clean sheets")
        for m in re.finditer(r"\b(\d+|no|zero|one|two|three|four|five|six|seven|eight|nine|ten) clean sheets?\b", t):
            k = int(m.group(1)) if m.group(1).isdigit() else _COUNT_WORDS[m.group(1)]
            if k != cs:
                wrong.append(f"there were {cs} clean sheet(s) in the games reviewed, not {k}")
        if re.search(r"\bunbeaten\b|without (a )?(defeat|loss)|\bnot lost\b|haven'?t lost", t) and "L" in res:
            wrong.append(f"they lost {res.count('L')} of the games reviewed")
        if re.search(r"\bwon (all|both|every)\b|\bperfect record\b|\b100% record\b", t) and set(res) != {"W"}:
            wrong.append(f"they won {res.count('W')} of {n}")
        if re.search(r"\bwinless\b|without a win|yet to win", t) and "W" in res:
            wrong.append(f"they won {res.count('W')} of the games reviewed")
        problems += [f"“{text.strip()[:120]}” — {w}" for w in wrong]
    return problems


def _texts(answer: dict, *keys: str) -> list[str]:
    """Every string the answer will show: `keys` may name a string, a list of
    strings, or a list of dicts (all their string values)."""
    out = []
    for k in keys:
        v = answer.get(k)
        for item in v if isinstance(v, list) else [v]:
            if isinstance(item, dict):
                out += [str(x) for x in item.values() if isinstance(x, str)]
            elif item is not None:
                out.append(str(item))
    return out


# ---------------------------------------------------------------------------
# Job 1: recommend changes for a goal
# ---------------------------------------------------------------------------

RECOMMEND_PROMPT = """You are the assistant manager of a football team. The head coach gives you a goal.
Find the best way to meet it with at most 3 changes to the starting XI, using the tools:
the brief already has the team (get_team); team_history if it has played; search_players for the positions that matter (put the
goal's qualities in `emphasis` and its hard constraints in `requirements`); get_player for close calls; then
evaluate_changes on your final set (try an alternative if it looks worse). If nothing genuinely helps,
propose no changes and say why.

Requirements are only the hard constraints the goal itself states ("under 25", "left-footed", "same
nationality as one of our centre-backs" -> nationality with the centre-backs' nationalities as `values`;
"under 30" -> age max 29, "over 30" -> age min 31). Each needs a "quote": the words of the goal it comes
from. Never invent one.

If evaluate_changes says a set is worse, look for a better set or propose none; if you still recommend it
(e.g. the coach's goal is worth a small cost), say so plainly in caveats.

""" + TOOL_GUIDE + """

Final JSON:
{"summary": "2-3 sentences: what you found and why these changes serve the goal",
 "requirements": [{"field": "...", "min": null, "max": null, "value": null, "values": null, "quote": "..."}],
 "changes": [{"slot": "CB_L", "player_id": "p_0012", "reason": "one sentence with the numbers behind it"}],
 "confidence": "high | medium | low",
 "caveats": ["anything the coach should know"]}"""


def recommend_changes(ctx: pipeline.Context, team_id: str, goal: str, model: str | None = None,
                      tracker: CostTracker | None = None) -> dict:
    """A goal in plain English -> verified, measured change proposals for a human
    to approve. Nothing is saved here."""
    model, tracker = model or llm_model(), tracker or make_tracker()
    toolbox = Toolbox(ctx, team_id)
    team = pipeline.rescored(ctx, storage.load_team(team_id))
    names, firsts = _names(ctx)

    def verify(answer: dict):
        problems = []
        for r in answer.get("requirements") or []:
            if isinstance(r, dict) and not _grounded(str(r.get("quote") or ""), goal):
                problems.append(f"requirement {r.get('field')!r}: its quote “{r.get('quote')}” isn't in the goal "
                                "— drop it unless the goal asks for it")
        try:
            reqs = _filters_from(answer.get("requirements"))
        except ValueError as e:
            reqs, problems = [], [*problems, str(e)]
        changes = _changes_dict(answer.get("changes"))
        if len(changes) > 3:
            problems.append("propose at most 3 changes")
        problems += check_changes(ctx, team, changes, reqs)
        for r in answer.get("requirements") or []:
            if isinstance(r, dict):
                problems += _bound_problems(r)
        if changes and not problems and not toolbox.tested(team_id, changes):
            problems.append("you didn't test this exact set of changes — call evaluate_changes with it, "
                            "then answer again")
        problems += fact_check(_texts(answer, "summary", "changes", "caveats"), toolbox, names, firsts)
        return answer, problems

    if _api_key() and goal.strip():
        brief = {"goal": goal, "team_id": team_id, "matches_played": len(pipeline.matches_for_team(team_id)),
                 "team (from get_team)": toolbox.call("get_team", {"team_id": team_id})}
        result = run_agent(RECOMMEND_PROMPT, brief, toolbox, verify, model, tracker)
    else:
        result = {"output": None, "error": None if goal.strip() else "no goal given"}
    out = _finish(result, lambda: _offline_recommend(ctx, team, goal, toolbox), model, toolbox)
    # keep only valid changes, then measure them ourselves -- the numbers shown come from code
    reqs = _safe_filters([r for r in out.get("requirements") or []
                          if isinstance(r, dict) and (out["source"] == "offline" or _grounded(str(r.get("quote") or ""), goal))])
    valid, dropped = {}, []
    for c in out.get("changes") or []:
        if not isinstance(c, dict):
            continue
        one = {str(c.get("slot")): str(c.get("player_id"))}
        issues = check_changes(ctx, team, {**valid, **one}, reqs) if len(valid) < 3 else ["more than 3 changes"]
        dropped.extend(issues) if issues else valid.update(one)
    out["changes"] = [{**c, "slot": str(c["slot"]), "player_id": str(c["player_id"])}
                      for c in out.get("changes") or []
                      if isinstance(c, dict) and valid.get(str(c.get("slot"))) == str(c.get("player_id"))]
    out["requirements"] = [describe_filter(f) for f in reqs]
    out["dropped"] = dropped
    out["measured"] = (toolbox.tested(team_id, valid) or measure(ctx, team_id, valid)) if valid else None
    out["caveats"] = [str(c) for c in out.get("caveats") or []] + _measured_caveats(out["measured"])
    out["team_id"], out["goal"] = team_id, goal
    return out


def _bound_problems(r: dict) -> list[str]:
    """'under 30' is max 29, not 30 -- the classic off-by-one when reading a limit."""
    quote = str(r.get("quote") or "")
    limit = strict_limit(quote)
    if not limit or r.get(limit[0]) is None or float(r[limit[0]]) == limit[1]:
        return []
    return [f"requirement {r.get('field')}: “{quote}” means {limit[0]} {limit[1]}, not {float(r[limit[0]]):g}"]


def _measured_caveats(m: dict | None) -> list[str]:
    """Plain facts from the code's own test, added whatever the model said."""
    if not m or m.get("error") or m.get("points_delta") is None:
        return []
    if m["verdict"] == "worse":
        return [(f"Measured by code: these changes did worse — {m['points_delta']:+.2f} points per match "
                 f"(± {m['margin']:.2f}) against {m['opponent']} over {m['matches']} paired matches.")]
    if m["verdict"] == "no clear difference":
        return [(f"Measured by code: no proven change in results against {m['opponent']} "
                 f"({m['points_delta']:+.2f} ± {m['margin']:.2f} points per match).")]
    return []


def _safe_filters(requirements) -> list[HardFilter]:
    try:
        return _filters_from(requirements)
    except ValueError:
        return []


# most specific first: a matched phrase is removed before the broader ones are tried,
# so "holding midfielder" is DM only and "wing-backs" aren't wingers
_POSITION_PHRASES = [
    (r"holding mid\w*|defensive mid\w*|\bdms?\b|\banchor\b", {"DM"}),
    (r"attacking mid\w*|number 10|\bno\.? ?10\b|\bams?\b|playmakers?", {"AM"}),
    (r"central mid\w*|box[- ]to[- ]box|\bcms?\b", {"CM"}),
    (r"full[- ]?backs?|wing[- ]?backs?|\blb\b|\brb\b", {"FB"}),
    (r"centre[- ]?backs?|center[- ]?backs?|\bcbs?\b|central defen\w*", {"CB"}),
    (r"keepers?|goalies?|goalkeepers?|shot[- ]stop\w*", {"GK"}),
    (r"\bwings?\b|wingers?|\bwide\b|flanks?", {"WING"}),
    (r"strikers?|centre[- ]forwards?|center[- ]forwards?|\bst\b|up front", {"ST"}),
    (r"\bforwards?\b|\battackers?\b|\battack\b", {"ST", "WING"}),   # "attacking <x>" is an adjective
    (r"at the back|back ?line|defen[cs]e|defenders?", {"CB", "FB"}),
    (r"midfield\w*", {"DM", "CM", "AM"}),
]


def goal_positions(goal: str) -> set[str]:
    """Positions a goal is about. Team-mates named as a reference ("same
    nationality as one of our centre-backs") are not targets."""
    text = re.sub(r"same (?:nationality|country) as [^,.;:]*", " ", goal.lower())
    found: set[str] = set()
    for pattern, positions in _POSITION_PHRASES:
        if re.search(pattern, text):
            found |= positions
            text = re.sub(pattern, " ", text)
    return found


def _offline_recommend(ctx: pipeline.Context, team, goal: str, toolbox: Toolbox) -> dict:
    """Deterministic planner: read the goal with the keyword rules (qualities,
    requirements, positions mentioned), rank candidates for each relevant slot
    under that emphasis and those requirements, and keep the biggest real gains
    (+2.0 fit, or -- at a position the goal names -- meeting a requirement the
    current player misses without a drop in fit), each player used once."""
    from .llm_client import _keyword_adjustments, _keyword_filters, _keyword_foot
    positions = goal_positions(goal)
    lineup = [pipeline._person(ctx, s.slot_id, s.player_id) for s in team.slots]
    emphasis_all = {a: 0.35 for a in _keyword_adjustments(goal, list(GLOSSARY))}
    reqs = [{"field": f["field"], "min": f["min"], "max": f["max"], "value": f["value"], "values": f.get("values")}
            for f in _keyword_filters(goal, list(GLOSSARY), {"lineup": lineup})]
    if foot := _keyword_foot(goal):
        reqs.append({"field": "preferred_foot", "value": foot})
    filters = _safe_filters(reqs)
    templates = pipeline.team_templates(ctx, team)
    options = []
    for s in team.slots:
        cur = ctx.lookup.get(s.player_id)
        if not cur or (positions and s.position not in positions):
            continue
        allowed = set(GK_ATTRS if s.position == "GK" else OUTFIELD_ATTRS)
        emphasis = {a: w for a, w in emphasis_all.items() if a in allowed
                    and (positions or a in templates[s.position].attribute_weights)}
        if emphasis_all and not emphasis:
            continue  # no position named, and this one doesn't use the qualities asked for
        found = toolbox.call("search_players", {"position": s.position, "emphasis": emphasis,
                                                 "requirements": reqs, "team_id": team.team_id, "limit": 5})
        if not found.get("candidates"):
            continue
        template = templates[s.position].model_copy(update={"attribute_weights": found["weights_used"]})
        current_fit = fit_score(cur, template)
        misses = bool(check_hard_filters(cur, filters))
        for cand in found["candidates"]:
            gain = cand["fit"] - current_fit
            # a requirement only forces a change at the positions the goal is about, and never a downgrade
            fixes_req = bool(positions) and misses and gain > 0
            if cand["meets_requirements"] and (gain >= 2.0 or fixes_req):
                options.append((fixes_req, gain, s.slot_id, cand))
    options.sort(key=lambda o: (o[0], o[1]), reverse=True)
    picks, used_slots, used_players = [], set(), set()
    for fixes, gain, slot, cand in options:
        if slot in used_slots or cand["player_id"] in used_players or len(picks) == 3:
            continue
        used_slots.add(slot)
        used_players.add(cand["player_id"])
        picks.append({"slot": slot, "player_id": cand["player_id"],
                      "reason": f"{cand['name']} fits {slot} {gain:+.1f} better with this emphasis"
                                + (" and meets the requirements the current player misses" if fixes else "")})
    summary = (f"Read by keyword rules: emphasis on {', '.join(emphasis_all) or 'nothing specific'}"
               + (f", positions {', '.join(sorted(positions))}" if positions else "")
               + (f", requirements {', '.join(describe_filter(f) for f in filters)}" if filters else "")
               + (f". {len(picks)} change(s) that gain at least +2.0 fit, or meet a requirement the current "
                  "player misses." if picks else ". Nobody available clearly improves on the current players."))
    return {"summary": summary, "requirements": reqs, "changes": picks, "confidence": "low",
            "caveats": ["Offline planner: keyword rules, no reasoning. Add an API key for the full assistant."]}


# ---------------------------------------------------------------------------
# Job 2: analyse a match
# ---------------------------------------------------------------------------

MATCH_PROMPT = """You are the performance analyst. Analyse the given match for the head coach, using the tools:
the brief already has the match (get_match), including replacement_options the code has tested for
players who struggled; team_history for each side's saved team (to tell a one-off from a trend); get_player
for the standout and struggling players; search_players + evaluate_changes if you want another replacement
(the slot must exist in that saved team as it is today -- get_team shows it).
Judge players by what their role is for: a keeper by saves and goals conceded, a defender by duels,
aerials and goals conceded, a midfielder by passing, chances and ball-winning, an attacker by goals, xG
and chances created.

""" + TOOL_GUIDE + """

Final JSON:
{"headline": "one line",
 "summary": "3-4 sentences on why the match went the way it did",
 "key_factors": [{"team_id": "<match side id>", "point": "...", "evidence": "the numbers"}],
 "players": [{"team_id": "<match side id>", "player_id": "...", "note": "rating, evidence, and form if known"}],
 "recommendations": [{"team_id": "<saved team id>", "slot": "<slot or null>",
                      "player_id": "<tested replacement or null>", "reason": "..."}]}
3-6 key factors; players = the best and the worst performer of each side; at most 2 recommendations per team.
A recommendation for a slot must name a replacement you tested with evaluate_changes; advice that isn't
about one slot has slot null and player_id null."""


def _check_recommendation(ctx: pipeline.Context, r: dict, saved: dict, toolbox: Toolbox | None = None
                          ) -> list[str]:
    """Problems with one suggested fix. With `toolbox`, a slot-specific fix must
    name a replacement the model actually tested (general advice has slot null)."""
    team = saved.get(r.get("team_id"))
    if team is None:
        return [f"recommendation team {r.get('team_id')!r} isn't one of the saved teams {sorted(saved)}"]
    if r.get("slot") is None:
        if r.get("player_id"):
            return ["a replacement needs a slot"]
        named = [sid for sid in (s.slot_id for s in team.slots) if re.search(rf"\b{sid}\b", str(r.get("reason")))]
        if toolbox and named:
            return [(f"general advice for {team.team_id} names {named[0]}: give that slot and a tested player_id "
                     "(see replacement_options), or leave that slot out")]
        return []
    if r["slot"] not in {s.slot_id for s in team.slots}:
        return [f"{r['slot']!r} isn't a slot in {team.name} ({', '.join(s.slot_id for s in team.slots)})"]
    if not r.get("player_id"):
        if not toolbox:
            return []
        return [(f"fix for {team.team_id} {r['slot']}: find a replacement with search_players and test it with "
                 "evaluate_changes (or make it general advice with slot null)")]
    problems = check_changes(ctx, team, {str(r["slot"]): str(r["player_id"])})
    if not problems and toolbox and not toolbox.tested_within(team.team_id, {str(r["slot"]): str(r["player_id"])}):
        problems.append(f"fix for {team.team_id} {r['slot']}: call evaluate_changes on it before recommending it")
    return problems


def _measure_recommendations(ctx: pipeline.Context, out: dict, saved: dict, toolbox: Toolbox) -> None:
    """Keep valid recommendations only; tested replacements carry the code's measurement."""
    kept = []
    for r in out.get("recommendations") or []:
        if not isinstance(r, dict) or r.get("team_id") not in saved:
            continue
        if _check_recommendation(ctx, {**r, "player_id": None}, saved):
            continue  # bad slot
        if r.get("player_id") and _check_recommendation(ctx, r, saved):
            r["player_id"] = None
        if r.get("player_id"):
            change = {r["slot"]: r["player_id"]}
            r["measured"] = toolbox.tested(r["team_id"], change) or measure(ctx, r["team_id"], change)
        kept.append(r)
    out["recommendations"] = kept


def analyse_match(ctx: pipeline.Context, match_id: str, model: str | None = None,
                  tracker: CostTracker | None = None) -> dict:
    model, tracker = model or llm_model(), tracker or make_tracker()
    toolbox = Toolbox(ctx, match_options=True)
    log = pipeline.load_event_log(match_id)
    sides = {log.team_a, log.team_b}
    names, firsts = _names(ctx)
    saved = {t: storage.load_team(t) for t in storage.list_teams()}

    def verify(answer: dict):
        problems = []
        for f in answer.get("key_factors") or []:
            if isinstance(f, dict) and f.get("team_id") not in sides:
                problems.append(f"key factor team_id {f.get('team_id')!r} isn't one of {sorted(sides)}")
        for p in answer.get("players") or []:
            if isinstance(p, dict) and p.get("player_id") not in log.lineups.get(p.get("team_id"), {}).values():
                problems.append(f"player {p.get('player_id')!r} didn't play for {p.get('team_id')!r}")
        for r in answer.get("recommendations") or []:
            if isinstance(r, dict):
                problems += _check_recommendation(ctx, r, saved, toolbox)
        problems += trend_check(_texts(answer, "headline", "summary", "key_factors", "players"), toolbox)
        problems += fact_check(_texts(answer, "headline", "summary", "key_factors", "players", "recommendations"),
                               toolbox, names, firsts)
        return answer, problems

    if _api_key():
        brief = {"match_id": match_id, "saved_teams": {t: {"name": team.name, "matches_played":
                                                           len(pipeline.matches_for_team(t))}
                                                       for t, team in saved.items()},
                 "match (from get_match)": toolbox.call("get_match", {"match_id": match_id})}
        result = run_agent(MATCH_PROMPT, brief, toolbox, verify, model, tracker)
    else:
        result = {"output": None, "error": None}
    out = _finish(result, lambda: _offline_match(ctx, match_id, saved, toolbox), model, toolbox)
    out["key_factors"] = [f for f in out.get("key_factors") or [] if isinstance(f, dict) and f.get("team_id") in sides]
    out["players"] = [p for p in out.get("players") or [] if isinstance(p, dict)
                      and p.get("player_id") in log.lineups.get(p.get("team_id"), {}).values()]
    _measure_recommendations(ctx, out, saved, toolbox)
    out["match_id"] = match_id
    return out


def _offline_match(ctx: pipeline.Context, match_id: str, saved: dict, toolbox: Toolbox) -> dict:
    """The computed debrief in the analysis format; slot fixes are the
    performance-derived replacements get_match has already tested."""
    data = toolbox.call("get_match", {"match_id": match_id})
    a, b = data["sides"].values()
    factors, players, recs = [], [], []
    for tid, side in data["sides"].items():
        factors += [{"team_id": tid, "point": x, "evidence": ""} for x in side["strengths"][:2]
                    if x != "Nothing stood out"]
        factors += [{"team_id": tid, "point": x, "evidence": ""} for x in side["weaknesses"][:2]
                    if not x.startswith("No clear")]
        if side["players"]:
            best, worst = side["players"][0], side["players"][-1]
            players += [{"team_id": tid, "player_id": best["player_id"],
                         "note": f"Best on the day ({best['rating']}): {best['strengths'][0]}"},
                        {"team_id": tid, "player_id": worst["player_id"],
                         "note": f"Lowest rating ({worst['rating']}): {worst['weaknesses'][0]}"}]
    for side in data["sides"].values():  # slot fixes: the options get_match already tested
        recs += [{"team_id": o["team_id"], "slot": o["slot"], "player_id": o["player_id"],
                  "reason": f"{o['current']}: {o['why']} — {o['name']} is the best-ranked replacement."}
                 for o in side.get("replacement_options", [])]
    for tid, d in pipeline.match_debrief(ctx, match_id).items():  # team-wide advice
        base = pipeline.base_team_id(tid)
        if base in saved:
            slots = {s.slot_id for s in saved[base].slots}
            recs += [{"team_id": base, "slot": None, "player_id": None, "reason": line} for line in d["changes"]
                     if line.split(" ", 1)[0] not in slots and not line.startswith("No change")]
    return {"headline": f"{a['team']} {a['goals_for']}–{b['goals_for']} {b['team']}",
            "summary": "Computed from the match numbers; no AI reasoning.", "key_factors": factors,
            "players": players, "recommendations": recs}


# ---------------------------------------------------------------------------
# Job 3: scout a player
# ---------------------------------------------------------------------------

SCOUT_PROMPT = """You are the chief scout. Assess the given player on his own merits for the head coach:
what kind of player he is, the roles he suits (roles_by_formation), his strengths and risks for those roles,
his form if he has played, and how he compares with the best alternatives at his position (use
search_players). The brief already has the player (get_player). Judge him only on what his role needs (a
centre-back is never judged on finishing, a striker never on tackling).
Only if the brief has tested_in_team (the coach asked about one of his teams): explain that test in
fit_for_teams (you may test another slot with evaluate_changes); otherwise fit_for_teams is [].

""" + TOOL_GUIDE + """

Final JSON:
{"summary": "3-4 sentences: what kind of player, the roles he suits, the main risk, form if he has played",
 "strengths": ["..."], "risks": ["..."],
 "compared_with": [{"player_id": "...", "note": "..."}],
 "fit_for_teams": [{"team_id": "...", "slot": "...", "reason": "the evaluate_changes numbers"}]}"""


def fit_verdict(m: dict) -> str:
    """Code's verdict on one tested change (never the model's)."""
    if m.get("error"):
        return "invalid"
    if m["verdict"] == "better":
        return "upgrade"
    if m["verdict"] == "worse":
        return "not needed"
    c = m["changes"][0] if m.get("changes") else {}
    gain = (c.get("fit_after") or 0) - (c.get("fit_before") or 0)
    return "squad option" if gain >= 2 else "not needed"


def _test_in_teams(ctx: pipeline.Context, player_id: str, teams: dict, toolbox: Toolbox) -> list[dict]:
    """For each team he isn't in: the slot where he'd play (a natural position
    first, then the biggest fit gain), tested with evaluate_changes."""
    p = ctx.lookup[player_id]
    out = []
    for tid, team in teams.items():
        if player_id in team.player_ids() or pipeline.missing_players(ctx, team):
            continue
        templates = pipeline.team_templates(ctx, team)
        options = [((s.position in p.natural_positions,
                     fit_score(p, templates[s.position]) - fit_score(ctx.lookup[s.player_id], templates[s.position])),
                    s) for s in team.slots if s.player_id and (s.position == "GK") == p.is_gk()]
        if not options:
            continue
        slot = max(options, key=lambda o: o[0])[1]
        tested = toolbox.call("evaluate_changes", {"changes": [{"slot": slot.slot_id, "player_id": player_id}],
                                                   "team_id": tid})
        out.append({"team_id": tid, "team": team.name, "slot": slot.slot_id,
                    "replacing": ctx.lookup[slot.player_id].name, "tested": tested})
    return out


def scout_player(ctx: pipeline.Context, player_id: str, model: str | None = None,
                 tracker: CostTracker | None = None, team_id: str | None = None) -> dict:
    """A scouting report on the player himself; with `team_id`, also tested in that team."""
    model, tracker = model or llm_model(), tracker or make_tracker()
    if player_id not in ctx.lookup:
        raise KeyError(f"no player {player_id!r} in the pool")
    toolbox = Toolbox(ctx)
    names, firsts = _names(ctx)
    teams = {team_id: storage.load_team(team_id)} if team_id else {}

    def verify(answer: dict):
        problems = []
        for c in answer.get("compared_with") or []:
            if isinstance(c, dict) and c.get("player_id") not in ctx.lookup:
                problems.append(f"compared_with player {c.get('player_id')!r} doesn't exist")
        for f in answer.get("fit_for_teams") or []:
            if not isinstance(f, dict):
                continue
            team = teams.get(f.get("team_id"))
            if team is None:
                problems.append(f"team {f.get('team_id')!r} doesn't exist")
            elif player_id in team.player_ids():
                problems.append(f"he already plays for {team.name}")
            else:
                issues = check_changes(ctx, team, {str(f.get("slot")): player_id})
                if not issues and not toolbox.tested_within(team.team_id, {str(f.get("slot")): player_id}):
                    issues.append(f"{team.team_id} {f.get('slot')}: call evaluate_changes before judging this slot")
                problems += issues
        covered = {f.get("team_id") for f in answer.get("fit_for_teams") or [] if isinstance(f, dict)}
        problems += [f"fit_for_teams is missing {t['team_id']} (tested at {t['slot']} in tested_in_team)"
                     for t in in_teams if t["team_id"] not in covered]
        problems += fact_check(_texts(answer, "summary", "strengths", "risks", "compared_with", "fit_for_teams"),
                               toolbox, names, firsts)
        return answer, problems

    in_teams = _test_in_teams(ctx, player_id, teams, toolbox)
    if _api_key():
        brief = {"player_id": player_id, "player (from get_player)": toolbox.call("get_player", {"player_id": player_id}),
                 **({"tested_in_team": in_teams} if team_id else {})}
        result = run_agent(SCOUT_PROMPT, brief, toolbox, verify, model, tracker)
    else:
        result = {"output": None, "error": None}
    out = _finish(result, lambda: _offline_scout(ctx, player_id, in_teams, toolbox), model, toolbox)
    fits = []
    for f in out.get("fit_for_teams") or []:
        team = teams.get(f.get("team_id")) if isinstance(f, dict) else None
        if team is None or player_id in team.player_ids() or check_changes(ctx, team, {str(f.get("slot")): player_id}):
            continue
        change = {f["slot"]: player_id}
        f["measured"] = toolbox.tested(f["team_id"], change) or measure(ctx, f["team_id"], change)
        f["verdict"] = fit_verdict(f["measured"])
        fits.append(f)
    out["fit_for_teams"] = fits
    out["team_id"] = team_id
    out["compared_with"] = [c for c in out.get("compared_with") or [] if isinstance(c, dict)
                            and c.get("player_id") in ctx.lookup and c["player_id"] != player_id]
    out["player_id"] = player_id
    return out


def _offline_scout(ctx: pipeline.Context, player_id: str, in_teams: list[dict], toolbox: Toolbox) -> dict:
    info = toolbox.call("get_player", {"player_id": player_id})
    crit = info["scouting"]["criteria"]
    ranked = sorted(crit, key=lambda k: -crit[k])
    rivals = toolbox.call("search_players", {"position": info["main_position"], "limit": 4})
    form = info["form"]
    roles = sorted(set(info["roles_by_formation"].values()))
    return {"summary": f"{info['name']} ({info['age']}, {info['main_position']}): "
                       f"{info['scouting']['verdict']} at {info['scouting']['overall']}/10 on the criteria for "
                       "the role." + (f" Suits: {', '.join(roles)}." if roles else "")
                       + (f" Average rating {form['avg_rating']} over {form['matches']} match(es)."
                          if form["matches"] else " No matches played yet."),
            "strengths": [f"{k}: {crit[k]}/10" for k in ranked[:2]],
            "risks": [f"{k}: {crit[k]}/10" for k in ranked[-1:]],
            "compared_with": [{"player_id": c["player_id"], "note": f"fit {c['fit']} at {info['main_position']}"}
                              for c in rivals.get("candidates", []) if c["player_id"] != player_id][:3],
            "fit_for_teams": [{"team_id": t["team_id"], "slot": t["slot"],
                               "reason": f"In place of {t['replacing']} at {t['slot']}."} for t in in_teams]}


# ---------------------------------------------------------------------------
# Job 4: review a team's history
# ---------------------------------------------------------------------------

REVIEW_PROMPT = """You are the assistant manager reviewing a team's form (all versions). The brief already has
team_history -- all its games, or a selection of them (see its "selection"; get_match only works for the
selected matches); say which you reviewed. Use get_team (who plays where now), get_player for players with
falling form or recurring weaknesses, and search_players + evaluate_changes to test any change you
recommend (a recommendation for a slot must name a replacement you tested; otherwise slot null).
Separate trends (seen in 2+ matches) from one-offs.

""" + TOOL_GUIDE + """

Final JSON:
{"summary": "3-4 sentences on the team's form and what is driving it",
 "trends": ["..."],
 "players": [{"player_id": "...", "note": "..."}],
 "recommendations": [{"team_id": "<this team>", "slot": "<slot or null>",
                      "player_id": "<tested replacement or null>", "reason": "..."}]}"""


def review_team(ctx: pipeline.Context, team_id: str, model: str | None = None,
                tracker: CostTracker | None = None, limit: int | None = None, pick: str = "last",
                seed: int | None = None) -> dict:
    """Review a team's form over all its games, or only the last / a random
    `limit` of them -- the tools are held to that selection, so a long history
    doesn't make the review expensive."""
    model, tracker = model or llm_model(), tracker or make_tracker()
    scope = {"limit": limit, "pick": pick, "seed": seed}
    history = pipeline.team_history(ctx, team_id, **scope)
    toolbox = Toolbox(ctx, team_id, history_scope=scope,
                      allowed_matches={m["match_id"] for m in history["matches"]} if limit else None)
    names, firsts = _names(ctx)
    team = storage.load_team(team_id)
    saved = {team_id: team}
    appeared = {p["player_id"] for p in history["players"]}

    def verify(answer: dict):
        problems = []
        for p in answer.get("players") or []:
            if isinstance(p, dict) and p.get("player_id") not in appeared:
                problems.append(f"player {p.get('player_id')!r} hasn't played for this team")
        for r in answer.get("recommendations") or []:
            if isinstance(r, dict):
                problems += _check_recommendation(ctx, {**r, "team_id": team_id}, saved, toolbox)
        problems += record_check(_texts(answer, "summary", "trends", "players"), history)
        problems += fact_check(_texts(answer, "summary", "trends", "players", "recommendations"),
                               toolbox, names, firsts)
        return answer, problems

    if _api_key() and history["matches"]:
        brief = {"team_id": team_id, "history (from team_history)": toolbox.call("team_history", {"team_id": team_id})}
        result = run_agent(REVIEW_PROMPT, brief, toolbox, verify, model, tracker)
    else:
        result = {"output": None, "error": None if history["matches"] else "the team hasn't played yet"}
    out = _finish(result, lambda: _offline_review(ctx, team, history, toolbox), model, toolbox)
    for r in out.get("recommendations") or []:
        if isinstance(r, dict):
            r["team_id"] = team_id
    _measure_recommendations(ctx, out, saved, toolbox)
    out["players"] = [p for p in out.get("players") or [] if isinstance(p, dict) and p.get("player_id") in appeared]
    out["team_id"] = team_id
    return out


def scope_label(selection: dict) -> str:
    """'all 4 games' · 'the last 10 of 42 games' · '10 random games of 42'."""
    used, total = selection["used"], selection["total"]
    if selection["pick"] == "last":
        return f"the last {used} of {total} games"
    if selection["pick"] == "random":
        return f"{used} random games of {total}"
    return f"all {total} game{'s' * (total != 1)}"


def _offline_review(ctx: pipeline.Context, team, history: dict, toolbox: Toolbox) -> dict:
    """Numbers-only review: record, falling form, recurring weaknesses, and a
    tested replacement for the lowest-rated regular still in the team."""
    r, players = history["record"], history["players"]
    if not history["matches"]:
        return {"summary": "No matches played yet — play a match to build a history.", "trends": [], "players": [],
                "recommendations": []}
    falling = [p for p in players if p["apps"] >= 2 and p["trend"] <= -0.5]
    recurring = [p for p in players if p["recurring_weaknesses"]]
    trends = ([f"{p['name']}: last rating {p['last_rating']} vs average {p['avg_rating']}" for p in falling]
              + [f"{p['name']} keeps showing: {p['recurring_weaknesses'][0]}" for p in recurring])[:5]
    notes = players[:2] + [p for p in players[-2:] if p not in players[:2]]
    recs = []
    slot_of = {s.player_id: s for s in team.slots}
    regulars = [p for p in reversed(players) if p["player_id"] in slot_of and slot_of[p["player_id"]].position != "GK"]
    if regulars:
        weakest = regulars[0]
        slot = slot_of[weakest["player_id"]]
        best = toolbox.call("search_players", {"position": slot.position, "team_id": team.team_id, "limit": 1})
        if best.get("candidates"):
            recs.append({"slot": slot.slot_id, "player_id": best["candidates"][0]["player_id"],
                         "reason": f"{weakest['name']} has the lowest average rating ({weakest['avg_rating']}) "
                                   f"of the current XI; best available {slot.position} by fit."})
    xg_for = sum(m["xg_for"] for m in history["matches"])
    xg_against = sum(m["xg_against"] for m in history["matches"])
    return {"summary": f"{scope_label(history['selection']).capitalize()}: {r['W']} won, {r['D']} drawn, "
                       f"{r['L']} lost; xG {xg_for:.2f} for, {xg_against:.2f} against.",
            "trends": trends or ["No trend yet — no player's form or weakness repeats across matches."],
            "players": [{"player_id": p["player_id"], "note": f"average {p['avg_rating']} over {p['apps']} "
                                                              f"match(es), last {p['last_rating']}"}
                        for p in notes],
            "recommendations": recs}
