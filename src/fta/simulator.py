"""Simulate a 90-minute match between two locked teams.

Deliberately event-level, not physics/positions on a pitch. Each minute one
side attacks, and the move is a chain of duels, each decided by a logistic
curve on the attribute difference of the two players involved:

  1. build-up pass    passer short passing        vs  presser standing tackle
  2. chance creation  (creator's best option)
       through ball   through-ball rating         vs  defender positioning
       cross          crossing                    vs  defender positioning
       dribble        dribbling 60% + pace 40%    vs  defender tackling 60% + pace 40%
  3. crosses only     aerial duel: heading acc 60% + heading power 40%
                                                  vs  defender heading acc 50% + marking 50%
  4. shot             finishing 80% + composure 20% (headers: heading acc 70% + power 30%)
                                                  vs  keeper reflexes 60% + positioning 40%

Players are picked for each duel in proportion to the attribute it uses, so
the best passer sees the ball most. After the 60th minute every outfield
attribute fades by up to 15% x (1 - stamina/100) -- stamina matters late.
Team chemistry adds (chemistry - 100) x 0.2 to every duel a team takes part in.

REPRODUCIBLE, NOT REPETITIVE: this function is deterministic -- same
(team_a, team_b, seed) always produces a byte-identical EventLog, which is
what lets tests assert exact output and lets you replay any past match from
its saved seed. The front ends (pipeline.simulate) draw a fresh random seed
for every match unless you explicitly ask to replay one.
"""
from __future__ import annotations

import math
import random

from .models import EventLog, MatchEvent, Player, Possession, Team

K = 12.0  # logistic steepness: every 12 points of advantage multiplies the odds by e (~2.7)
# Shots are tuned to real football: ~0.1 xG per shot, ~2.5 goals a match. A
# flatter curve (K=20) keeps finishing quality relevant without inflating scores.
SHOT_BASE, SHOT_K = 0.14, 20.0
MATCH_MINUTES = 90
FATIGUE_FROM_MINUTE = 60
MAX_FATIGUE = 0.15
CHEMISTRY_EFFECT = 0.2  # duel points per chemistry point above (or below) 100


def _p(attacker_val: float, defender_val: float, base: float = 0.5, k: float = K) -> float:
    """Logistic curve centered on `base`: equal attributes -> base probability."""
    diff = attacker_val - defender_val
    logit_base = math.log(base / (1 - base))
    return 1 / (1 + math.exp(-(logit_base + diff / k)))


def _team_outfield(team: Team, lookup: dict[str, Player]) -> list[Player]:
    return [lookup[s.player_id] for s in team.slots if s.player_id and not lookup[s.player_id].is_gk()]


def _team_gk(team: Team, lookup: dict[str, Player]) -> Player | None:
    for s in team.slots:
        if s.player_id and lookup[s.player_id].is_gk():
            return lookup[s.player_id]
    return None


def _eff(fade: float, p: Player, attr: str) -> float:
    """Effective (fatigue-adjusted) attribute: fades late for low-stamina players."""
    return p.attr(attr) * (1 - fade * (1 - p.attr("stamina") / 100))


def _mix(fade: float, p: Player, *parts: tuple[str, float]) -> float:
    return sum(_eff(fade, p, attr) * w for attr, w in parts)


def _pick_weighted(rng: random.Random, players: list[Player], attr: str) -> Player:
    weights = [max(1.0, p.attr(attr, default=50)) for p in players]
    total = sum(weights)
    r = rng.uniform(0, total)
    upto = 0.0
    for player, w in zip(players, weights):
        upto += w
        if upto >= r:
            return player
    return players[-1]


def simulate_match(match_id: str, team_a: Team, team_b: Team, lookup: dict[str, Player],
                   seed: int, num_possessions: int = MATCH_MINUTES,
                   chemistry: dict[str, float] | None = None) -> EventLog:
    rng = random.Random(seed)
    edge = {t: ((chemistry or {}).get(t, 100.0) - 100.0) * CHEMISTRY_EFFECT
            for t in (team_a.team_id, team_b.team_id)}
    outfield = {team_a.team_id: _team_outfield(team_a, lookup), team_b.team_id: _team_outfield(team_b, lookup)}
    gk = {team_a.team_id: _team_gk(team_a, lookup), team_b.team_id: _team_gk(team_b, lookup)}

    possessions: list[Possession] = []
    score = {team_a.team_id: 0, team_b.team_id: 0}

    # Possession is awarded probabilistically by relative midfield passing strength.
    def midfield_strength(team_id: str) -> float:
        players = outfield[team_id]
        return sum(p.attr("passing_short") + p.attr("vision") for p in players) / max(1, len(players))

    p_a_possession = _p(midfield_strength(team_a.team_id), midfield_strength(team_b.team_id), base=0.5)

    for i in range(1, num_possessions + 1):
        minute = max(1, math.ceil(i * MATCH_MINUTES / num_possessions))
        fade = MAX_FATIGUE * max(0.0, (minute - FATIGUE_FROM_MINUTE) / (MATCH_MINUTES - FATIGUE_FROM_MINUTE))

        attacking_id = team_a.team_id if rng.random() < p_a_possession else team_b.team_id
        defending_id = team_b.team_id if attacking_id == team_a.team_id else team_a.team_id
        attackers, defenders = outfield[attacking_id], outfield[defending_id]
        att, dfn = edge[attacking_id], edge[defending_id]
        chain: list[MatchEvent] = []

        # 1. build-up
        passer = _pick_weighted(rng, attackers, "passing_short")
        presser = _pick_weighted(rng, defenders, "tackling_standing")
        build_success = rng.random() < _p(_eff(fade, passer, "passing_short") + att, _eff(fade, presser, "tackling_standing") + dfn)
        chain.append(MatchEvent(event="build_up_pass", player_id=passer.player_id, team_id=attacking_id,
                                success=build_success))
        if not build_success:
            chain.append(MatchEvent(event="turnover", player_id=presser.player_id, team_id=defending_id,
                                    success=True))
            possessions.append(Possession(possession_id=i, attacking_team=attacking_id, chain=chain, minute=minute))
            continue
        chain.append(MatchEvent(event="tackle_missed", player_id=presser.player_id, team_id=defending_id,
                                success=False))

        # 2. chance creation: the creator uses whichever of his three options he's best at
        creator = _pick_weighted(rng, attackers, "vision")
        options = {"through_ball": _eff(fade, creator, "through_ball"), "cross": _eff(fade, creator, "crossing"),
                   "dribble": _mix(fade, creator, ("dribbling", 0.6), ("pace", 0.4))}
        chance_event = max(options, key=options.get)
        defender = _pick_weighted(rng, defenders, "positioning")
        defence = (_mix(fade, defender, ("tackling_standing", 0.6), ("pace", 0.4)) if chance_event == "dribble"
                   else _eff(fade, defender, "positioning"))
        chance_success = rng.random() < _p(options[chance_event] + att, defence + dfn, base=0.45)
        chain.append(MatchEvent(event=chance_event, player_id=creator.player_id, team_id=attacking_id,
                                success=chance_success))
        if not chance_success:
            chain.append(MatchEvent(event="turnover", player_id=defender.player_id, team_id=defending_id,
                                    success=True))
            possessions.append(Possession(possession_id=i, attacking_team=attacking_id, chain=chain, minute=minute))
            continue
        if chance_event != "dribble" and creator.player_id != passer.player_id:
            chain.append(MatchEvent(event="key_pass", player_id=creator.player_id, team_id=attacking_id,
                                    success=True))

        # 3. who shoots, and with what
        if chance_event == "dribble":
            finisher = creator
            finish = _mix(fade, finisher, ("finishing", 0.8), ("composure", 0.2))
        elif chance_event == "cross":
            finisher = _pick_weighted(rng, [p for p in attackers if p is not creator] or attackers,
                                      "heading_accuracy")
            marker = _pick_weighted(rng, defenders, "heading_accuracy")
            won = rng.random() < _p(_mix(fade, finisher, ("heading_accuracy", 0.6), ("heading_power", 0.4)) + att,
                                    _mix(fade, marker, ("heading_accuracy", 0.5), ("marking", 0.5)) + dfn)
            (winner, w_team), (loser, l_team) = (((finisher, attacking_id), (marker, defending_id)) if won
                                                 else ((marker, defending_id), (finisher, attacking_id)))
            chain.append(MatchEvent(event="aerial_duel", player_id=winner.player_id, team_id=w_team, success=True))
            chain.append(MatchEvent(event="aerial_duel", player_id=loser.player_id, team_id=l_team, success=False))
            if not won:
                possessions.append(Possession(possession_id=i, attacking_team=attacking_id, chain=chain, minute=minute))
                continue
            finish = _mix(fade, finisher, ("heading_accuracy", 0.7), ("heading_power", 0.3))
        else:
            finisher = _pick_weighted(rng, [p for p in attackers if p is not creator] or attackers, "finishing")
            finish = _mix(fade, finisher, ("finishing", 0.8), ("composure", 0.2))

        # 4. shot vs keeper
        keeper = gk[defending_id]
        keeper_val = keeper.attr("gk_reflexes") * 0.6 + keeper.attr("gk_positioning") * 0.4 if keeper else 40.0
        goal_prob = _p(finish + att, keeper_val + dfn, base=SHOT_BASE, k=SHOT_K)
        is_goal = rng.random() < goal_prob
        outcome = "goal" if is_goal else ("saved" if rng.random() < 0.55 else "off_target")
        chain.append(MatchEvent(event="shot", player_id=finisher.player_id, team_id=attacking_id, outcome=outcome,
                                xg=round(goal_prob, 3)))
        if keeper and outcome == "saved":
            chain.append(MatchEvent(event="save", player_id=keeper.player_id, team_id=defending_id, success=True))
        if keeper and is_goal:
            chain.append(MatchEvent(event="conceded", player_id=keeper.player_id, team_id=defending_id,
                                    success=False))
        if is_goal:
            score[attacking_id] += 1
        possessions.append(Possession(possession_id=i, attacking_team=attacking_id, chain=chain, minute=minute))

    return EventLog(
        match_id=match_id, seed=seed, team_a=team_a.team_id, team_b=team_b.team_id,
        possessions=possessions, final_score=score,
    )
