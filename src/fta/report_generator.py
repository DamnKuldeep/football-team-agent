"""Match analysis, all pure arithmetic (no LLM):

* per-side, per-player stats from the event log (a player who appears for
  both sides -- a team against its own older version -- keeps two reports);
* role-based ratings -- a centre-back earns points for duels, aerials and
  clean sheets, a striker for goals, chances and xG, a keeper for saves;
* role-relevant feedback -- only what matters for that role, each line backed
  by a number;
* team strengths/weaknesses measured against the opponent;
* a coach's debrief (what worked, what didn't, concrete changes);
* swap requests derived from a player's weaknesses (the feedback loop).

The assistant (agent.py) reads these numbers through its tools; it never
changes them.
"""
from __future__ import annotations

from .models import EventLog, PlayerReport, PlayerStats, SwapRequest

ROLE_GROUP = {"GK": "GK", "CB": "DEF", "FB": "DEF", "DM": "MID", "CM": "MID", "AM": "MID", "WING": "ATT",
              "ST": "ATT"}
ROLE_LABEL = {"GK": "Goalkeeper", "DEF": "Defender", "MID": "Midfielder", "ATT": "Attacker"}
RATING_BASE = 6.0
# Points per action, by role. Rating = 6.0 + sum, kept between 0 and 10.
RATING_POINTS = {
    "GK": {"saves": 0.35, "goals_conceded": -0.4, "clean_sheet": 0.5, "passes_completed": 0.01},
    "DEF": {"tackles_won": 0.25, "tackles_lost": -0.2, "aerial_duels_won": 0.2, "aerial_duels_lost": -0.15,
            "passes_completed": 0.03, "passes_failed": -0.05, "key_passes": 0.3, "goals": 1.0,
            "team_conceded": -0.15, "clean_sheet": 0.3},
    "MID": {"passes_completed": 0.04, "passes_failed": -0.05, "key_passes": 0.35, "tackles_won": 0.2,
            "tackles_lost": -0.1, "dribbles_completed": 0.15, "dribbles_failed": -0.05, "goals": 1.0,
            "shots": 0.05, "wasted_xg": -0.2},
    "ATT": {"goals": 1.2, "shots": 0.1, "wasted_xg": -0.3, "key_passes": 0.3, "dribbles_completed": 0.2,
            "dribbles_failed": -0.05, "aerial_duels_won": 0.1, "passes_completed": 0.02, "passes_failed": -0.03},
}
BIG_CHANCE_XG = 0.3


def role_group(position: str | None) -> str:
    return ROLE_GROUP.get(position or "", "MID")


# ---------------------------------------------------------------------------
# Stats
# ---------------------------------------------------------------------------

_DEFENDING_EVENTS = {"turnover", "tackle_missed", "save", "conceded"}


def report_key(team_id: str, player_id: str) -> str:
    """Reports are per team AND player: the same player can play for both sides
    (a team against an older version of itself), and his two games must never merge."""
    return f"{team_id}|{player_id}"


def _side(log: EventLog, poss, e) -> str:
    if e.team_id:
        return e.team_id
    sides = log.sides_of(e.player_id)  # older logs: infer the side
    if len(sides) == 1:
        return sides[0]
    defending = log.team_b if poss.attacking_team == log.team_a else log.team_a
    return defending if e.event in _DEFENDING_EVENTS else poss.attacking_team


def aggregate_stats(log: EventLog, starters: set[tuple[str, str]] | None = None) -> dict[str, PlayerStats]:
    """Per-side stats keyed by report_key(team, player). `starters` ((team,
    player) pairs) seeds a zero-stat entry for everyone who started, so players
    who never touched the ball still get a report."""
    stats: dict[str, PlayerStats] = {report_key(t, pid): PlayerStats(player_id=pid) for t, pid in (starters or set())}

    def get(team_id: str, pid: str) -> PlayerStats:
        return stats.setdefault(report_key(team_id, pid), PlayerStats(player_id=pid))

    for poss in log.possessions:
        for e in poss.chain:
            s = get(_side(log, poss, e), e.player_id)
            if e.event in ("build_up_pass", "through_ball", "cross"):
                s.passes_attempted += 1
                s.passes_completed += bool(e.success)
            elif e.event == "dribble":
                s.dribbles_attempted += 1
                s.dribbles_completed += bool(e.success)
            elif e.event == "key_pass":
                s.key_passes += 1
            elif e.event == "turnover":
                s.tackles_won += 1
            elif e.event == "tackle_missed":
                s.tackles_lost += 1
            elif e.event == "aerial_duel":
                if e.success:
                    s.aerial_duels_won += 1
                else:
                    s.aerial_duels_lost += 1
            elif e.event == "shot":
                s.shots += 1
                s.xg = round(s.xg + (e.xg or 0), 3)
                s.goals += e.outcome == "goal"
            elif e.event == "save":
                s.saves += 1
            elif e.event == "conceded":
                s.goals_conceded += 1
    return stats


# ---------------------------------------------------------------------------
# Ratings
# ---------------------------------------------------------------------------

def rating_breakdown(stats: PlayerStats, position: str | None = None,
                     team_conceded: int = 0) -> tuple[float, list[tuple[str, float, float]]]:
    """(rating, [(stat, count, points)]) -- every non-zero term of the rating."""
    points = RATING_POINTS[role_group(position)]
    counts = {
        "passes_completed": stats.passes_completed,
        "passes_failed": stats.passes_attempted - stats.passes_completed,
        "key_passes": stats.key_passes, "tackles_won": stats.tackles_won, "tackles_lost": stats.tackles_lost,
        "aerial_duels_won": stats.aerial_duels_won, "aerial_duels_lost": stats.aerial_duels_lost,
        "dribbles_completed": stats.dribbles_completed,
        "dribbles_failed": stats.dribbles_attempted - stats.dribbles_completed,
        "shots": stats.shots, "goals": stats.goals, "wasted_xg": round(max(0.0, stats.xg - stats.goals), 2),
        "saves": stats.saves, "goals_conceded": stats.goals_conceded,
        "team_conceded": team_conceded, "clean_sheet": int(team_conceded == 0),
    }
    terms = [(k, v, round(v * points[k], 2)) for k, v in counts.items() if v and k in points]
    rating = RATING_BASE + sum(t[2] for t in terms)
    return round(max(0.0, min(10.0, rating)), 1), terms


def compute_rating(stats: PlayerStats, position: str | None = None, team_conceded: int = 0) -> float:
    return rating_breakdown(stats, position, team_conceded)[0]


# ---------------------------------------------------------------------------
# Player feedback: only what matters for the role
# ---------------------------------------------------------------------------

def player_feedback(stats: PlayerStats, position: str | None, team_conceded: int = 0
                    ) -> tuple[list[str], list[str]]:
    """Evidence-backed (strengths, weaknesses) for one match, judged on the
    things the player's role is for. Always at least one line in each."""
    s, group, good, bad = stats, role_group(position), [], []
    acc = s.pass_accuracy_pct
    duels = s.tackles_won + s.tackles_lost
    aerials = s.aerial_duels_won + s.aerial_duels_lost
    missed = round(s.xg - s.goals, 2)

    if group == "GK":
        faced = s.saves + s.goals_conceded
        if s.saves >= 3:
            good.append(f"Made {s.saves} saves")
        if s.goals_conceded == 0 and faced:
            good.append(f"Clean sheet from {faced} shots on target")
        if s.goals_conceded >= 2 and s.saves / faced < 0.6:
            bad.append(f"Saved only {s.saves} of {faced} shots on target ({100 * s.saves / faced:.0f}%)")
        return good or ["A quiet game — little to do"], bad or ["No clear weakness this match"]

    if group in ("DEF", "MID") and duels:
        if s.tackles_won >= 3 and s.tackles_won >= 2 * s.tackles_lost:
            good.append(f"Won {s.tackles_won} of {duels} duels")
        elif s.tackles_lost >= 3 and s.tackles_lost > s.tackles_won:
            bad.append(f"Bypassed in {s.tackles_lost} of {duels} duels")
    if (group == "DEF" or position == "ST") and aerials:
        if s.aerial_duels_won >= 2 and s.aerial_duels_won >= 2 * s.aerial_duels_lost:
            good.append(f"Won {s.aerial_duels_won} of {aerials} aerial duels")
        elif s.aerial_duels_lost >= 2 and s.aerial_duels_lost > s.aerial_duels_won:
            bad.append(f"Lost {s.aerial_duels_lost} of {aerials} aerial duels")
    if group in ("DEF", "MID"):
        if s.passes_attempted >= 8 and acc >= 88:
            good.append(f"Secure in possession: {acc:.0f}% passing ({s.passes_completed}/{s.passes_attempted})")
        elif s.passes_attempted >= 5 and acc < 70:
            bad.append(f"Gave the ball away: {acc:.0f}% passing ({s.passes_completed}/{s.passes_attempted})")
    if group == "DEF":
        if team_conceded == 0:
            good.append("Part of a clean-sheet back line")
        elif team_conceded >= 3:
            bad.append(f"Part of a back line that conceded {team_conceded}")
    if (group in ("MID", "ATT") or position == "FB") and s.key_passes >= 2:
        good.append(f"Created {s.key_passes} chances for teammates")
    if group in ("MID", "ATT") and s.dribbles_attempted:
        if s.dribbles_completed >= 2:
            good.append(f"Beat his man {s.dribbles_completed} times")
        elif s.dribbles_attempted >= 3 and s.dribbles_completed / s.dribbles_attempted < 0.5:
            bad.append(f"Lost the ball on {s.dribbles_attempted - s.dribbles_completed} of "
                       f"{s.dribbles_attempted} dribbles")
    if group in ("MID", "ATT"):
        if s.goals:
            good.append(f"Scored {s.goals} from {s.xg:.2f} xG" + (" — clinical" if s.goals >= s.xg + 0.5 else ""))
        if missed >= 0.6:
            bad.append(f"Missed chances: {s.goals} goal(s) from {s.xg:.2f} xG")
        elif group == "ATT" and s.shots >= 3 and not s.goals:
            bad.append(f"{s.shots} shots, no goal")
    actions = s.passes_attempted + duels + s.shots + aerials + s.dribbles_attempted
    if actions <= 3:
        bad.append(f"Barely involved: {actions} actions all match")
    return good or ["No standout contribution"], bad or ["No clear weakness this match"]


def build_reports(log: EventLog, starters: set[tuple[str, str]] | None = None) -> dict[str, PlayerReport]:
    """Reports keyed by report_key(team, player)."""
    reports = {}
    for key, stats in aggregate_stats(log, starters).items():
        team_id, pid = key.split("|", 1)
        slot_id = (log.slot_of(pid, team_id) or (None, None))[1]
        position = log.slot_positions.get(team_id, {}).get(slot_id)
        opp = log.team_b if team_id == log.team_a else log.team_a
        conceded = log.final_score.get(opp, 0) if team_id else 0
        good, bad = player_feedback(stats, position, conceded)
        reports[key] = PlayerReport(
            player_id=pid, match_id=log.match_id, stats=stats, rating=compute_rating(stats, position, conceded),
            team_id=team_id, slot_id=slot_id, position=position, strengths=good, weaknesses=bad)
    return reports


# ---------------------------------------------------------------------------
# Team stats and team strengths/weaknesses
# ---------------------------------------------------------------------------

def team_stats(log: EventLog, reports: dict[str, PlayerReport]) -> dict[str, dict[str, float]]:
    """Broadcast-style team totals, keyed by team_id."""
    out = {}
    total = max(1, len(log.possessions))
    for tid in (log.team_a, log.team_b):
        mine = [r.stats for r in reports.values() if r.team_id == tid]
        shots = [e for p in log.possessions if p.attacking_team == tid for e in p.chain if e.event == "shot"]
        passes = sum(s.passes_attempted for s in mine)
        duels_won = sum(s.tackles_won for s in mine)
        duels = duels_won + sum(s.tackles_lost for s in mine)
        aerials_won = sum(s.aerial_duels_won for s in mine)
        aerials = aerials_won + sum(s.aerial_duels_lost for s in mine)
        out[tid] = {
            "Goals": log.final_score[tid],
            "Expected goals (xG)": round(sum(e.xg or 0 for e in shots), 2),
            "Possession %": round(100 * sum(p.attacking_team == tid for p in log.possessions) / total),
            "Shots": len(shots),
            "Shots on target": sum(e.outcome in ("goal", "saved") for e in shots),
            "Big chances": sum((e.xg or 0) >= BIG_CHANCE_XG for e in shots),
            "Pass accuracy %": round(100 * sum(s.passes_completed for s in mine) / passes) if passes else 0,
            "Duels won %": round(100 * duels_won / duels) if duels else 0,
            "Aerial duels won %": round(100 * aerials_won / aerials) if aerials else 0,
            "Saves": sum(s.saves for s in mine),
        }
    return out


def _xg_share_by_period(log: EventLog, tid: str) -> tuple[float, float]:
    """This team's share of all xG before and after the 60th minute."""
    def share(lo: int, hi: int) -> float:
        xg = {log.team_a: 0.0, log.team_b: 0.0}
        for p in log.possessions:
            if lo <= (p.minute or p.possession_id) <= hi:
                for e in p.chain:
                    if e.event == "shot":
                        xg[p.attacking_team] += e.xg or 0
        total = sum(xg.values())
        return xg[tid] / total if total else 0.5
    return share(1, 60), share(61, 90)


def team_insights(log: EventLog, reports: dict[str, PlayerReport]) -> dict[str, dict[str, list[str]]]:
    """Strengths and weaknesses per team, each measured against the opponent."""
    stats = team_stats(log, reports)
    out = {}
    for tid in (log.team_a, log.team_b):
        opp = log.team_b if tid == log.team_a else log.team_a
        me, them = stats[tid], stats[opp]
        good, bad = [], []
        xg, xga = me["Expected goals (xG)"], them["Expected goals (xG)"]
        if xg - xga >= 0.4:
            good.append(f"Created the better chances: {xg:.2f} xG to {xga:.2f}")
        elif xga - xg >= 0.4:
            bad.append(f"Out-created: {xg:.2f} xG to {xga:.2f}")
        if me["Big chances"] >= 2 and me["Big chances"] > them["Big chances"]:
            good.append(f"{me['Big chances']} big chances (xG ≥ {BIG_CHANCE_XG}) to {them['Big chances']}")
        finishing = me["Goals"] - xg
        if finishing >= 0.7:
            good.append(f"Clinical: {me['Goals']} goals from {xg:.2f} xG")
        elif finishing <= -0.7:
            bad.append(f"Wasteful: {me['Goals']} goals from {xg:.2f} xG")
        on_target = [e for p in log.possessions if p.attacking_team == opp for e in p.chain
                     if e.event == "shot" and e.outcome in ("goal", "saved")]
        prevented = sum(e.xg or 0 for e in on_target) - them["Goals"]
        if prevented >= 0.5:
            good.append(f"Goalkeeping kept them in it: {prevented:.2f} goals prevented")
        elif prevented <= -0.5:
            bad.append(f"Conceded {-prevented:.2f} more than the chances were worth")
        if me["Possession %"] >= 55:
            good.append(f"Controlled the ball: {me['Possession %']}% possession")
        elif me["Possession %"] <= 45:
            bad.append(f"Couldn't keep the ball: {me['Possession %']}% possession")
        if me["Duels won %"] >= 58:
            good.append(f"Won the physical battle: {me['Duels won %']}% of duels")
        elif me["Duels won %"] and me["Duels won %"] <= 42:
            bad.append(f"Lost the physical battle: {me['Duels won %']}% of duels")
        if me["Aerial duels won %"] >= 60 and me["Aerial duels won %"] + them["Aerial duels won %"]:
            good.append(f"Dominant in the air: {me['Aerial duels won %']}% of aerials")
        elif me["Aerial duels won %"] and me["Aerial duels won %"] <= 40:
            bad.append(f"Beaten in the air: {me['Aerial duels won %']}% of aerials")
        early, late = _xg_share_by_period(log, tid)
        if late - early >= 0.2:
            good.append(f"Finished strongly: {early:.0%} of the chances before the hour, {late:.0%} after")
        elif early - late >= 0.2:
            bad.append(f"Faded late: {early:.0%} of the chances before the hour, {late:.0%} after")
        out[tid] = {"strengths": good or ["Nothing stood out"], "weaknesses": bad or ["No clear team-wide weakness"]}
    return out


# ---------------------------------------------------------------------------
# Coach's debrief (deterministic)
# ---------------------------------------------------------------------------

# (roles it applies to, weakness trigger, attribute weights to look for, why, what to look for in words).
# Role-aware: a striker is never told to find "a better passer", a centre-back never "a sharper finisher".
_FEEDBACK_RULES = [
    ({"DEF", "MID"}, lambda s: s.passes_attempted >= 5 and s.pass_accuracy_pct < 70,
     {"passing_short": 0.3, "vision": 0.2}, lambda s: f"{s.pass_accuracy_pct:.0f}% passing", "a better passer"),
    ({"DEF", "MID"}, lambda s: s.tackles_lost >= 3 and s.tackles_lost > s.tackles_won,
     {"tackling_standing": 0.3, "marking": 0.25}, lambda s: f"bypassed in {s.tackles_lost} duels",
     "a stronger tackler"),
    ({"DEF", "ATT"}, lambda s: s.aerial_duels_lost >= 2 and s.aerial_duels_lost > s.aerial_duels_won,
     {"heading_accuracy": 0.3, "heading_power": 0.2}, lambda s: f"lost {s.aerial_duels_lost} aerial duels",
     "someone better in the air"),
    ({"MID", "ATT"}, lambda s: s.xg - s.goals >= 0.6 or (s.shots >= 3 and not s.goals),
     {"finishing": 0.35, "composure": 0.2}, lambda s: f"{s.goals} goal(s) from {s.xg:.2f} xG", "a sharper finisher"),
    ({"MID", "ATT"}, lambda s: s.dribbles_attempted >= 3 and s.dribbles_completed / s.dribbles_attempted < 0.5,
     {"dribbling": 0.3, "ball_control": 0.2},
     lambda s: f"{s.dribbles_completed}/{s.dribbles_attempted} dribbles completed", "a better dribbler"),
    ({"GK"}, lambda s: s.goals_conceded >= 2 and s.saves < 2 * s.goals_conceded,
     {"gk_reflexes": 0.35, "gk_positioning": 0.3}, lambda s: f"conceded {s.goals_conceded}, {s.saves} saves",
     "a better shot-stopper"),
]


def debrief(log: EventLog, reports: dict[str, PlayerReport], names: dict[str, str]
            ) -> dict[str, dict[str, list[str] | str]]:
    """Per team: summary, what worked, what didn't, and up to 3 concrete changes."""
    insights = team_insights(log, reports)
    out = {}
    for tid in (log.team_a, log.team_b):
        opp = log.team_b if tid == log.team_a else log.team_a
        gf, ga = log.final_score[tid], log.final_score[opp]
        result = "won" if gf > ga else "lost" if gf < ga else "drew"
        mine = sorted((r for r in reports.values() if r.team_id == tid), key=lambda r: r.rating)
        best = mine[-1] if mine else None
        changes = []
        for r in mine:
            for roles, triggered, _, why, wanted in _FEEDBACK_RULES:
                if role_group(r.position) in roles and triggered(r.stats) and len(changes) < 3:
                    changes.append(f"{r.slot_id} ({names.get(r.player_id, r.player_id)}, {r.rating:.1f}): "
                                   f"{why(r.stats)} — look for {wanted}.")
                    break
        if any("Faded late" in w for w in insights[tid]["weaknesses"]):
            changes.append("Midfield ran out of legs after the hour — pick higher-stamina midfielders "
                           "(also lifts the 'Midfield engine' chemistry rule).")
        out[tid] = {
            "summary": (f"{log.name(tid)} {result} {gf}–{ga}. "
                        + (f"Best player: {names.get(best.player_id, best.player_id)} ({best.rating:.1f})."
                           if best else "")),
            "worked": insights[tid]["strengths"][:3],
            "didnt": insights[tid]["weaknesses"][:3],
            "changes": changes or ["No change needed on this evidence — play another match to confirm."],
        }
    return out


def derive_swap_request(target_slot: str, player_id: str, stats: PlayerStats,
                        position: str | None = None) -> SwapRequest | None:
    """Turn a player's match weaknesses into the same SwapRequest a human would
    type, so the feedback loop reuses the swap engine as-is."""
    reweight: dict[str, float] = {}
    reasons: dict[str, str] = {}
    for roles, triggered, weights, why, _ in _FEEDBACK_RULES:
        if (position is None or role_group(position) in roles) and triggered(stats):
            for attr, w in weights.items():
                reweight[attr] = max(reweight.get(attr, 0), w)
                reasons[attr] = why(stats)
    if not reweight:
        return None
    return SwapRequest(
        target_slot=target_slot, reweight=reweight, requested_by="agent", source="performance_derived",
        rationale="; ".join(dict.fromkeys(reasons.values())), adjustment_reasons=reasons,
        interpreted_by="performance",
    )
