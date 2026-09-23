"""Build an initial XI.

Uses global greedy assignment (repeatedly pick the single best remaining
(slot, player) pair, assign, remove both from contention) rather than
filling slots in a fixed order -- that avoids the first slot processed
hogging a player who'd have been a similarly good fit elsewhere while
being a much better fit for the earlier slot.
"""
from __future__ import annotations

import copy

from .models import ChemistryRule, FormationSlot, Player, PositionTemplate, SoftBonus, Team
from .scoring import check_hard_filters, fit_score, team_avg_fit, team_chemistry


def build_team(
    team_id: str,
    name: str,
    formation_name: str,
    formation_slots: list[dict],
    pool: list[Player],
    templates: dict[str, PositionTemplate],
    chemistry_rules: list[ChemistryRule],
) -> Team:
    lookup = {p.player_id: p for p in pool}
    slot_position_map = {s["slot_id"]: s["position"] for s in formation_slots}

    # Precompute every valid (slot, player) score, skipping hard-filter violations.
    candidates: list[tuple[float, str, str]] = []  # (score, slot_id, player_id)
    for slot in formation_slots:
        template = templates[slot["position"]]
        # apply formation-specific foot bonus (e.g. left-footed LB) as an
        # ad-hoc extra soft bonus without mutating the shared template
        local_template = _with_foot_bonus(template, slot.get("foot_bonus"))
        for player in pool:
            if slot["position"] == "GK" and not player.is_gk():
                continue
            if slot["position"] != "GK" and player.is_gk():
                continue
            violations = check_hard_filters(player, local_template.hard_filters)
            if violations:
                continue
            score = fit_score(player, local_template)
            candidates.append((score, slot["slot_id"], player.player_id))

    candidates.sort(key=lambda c: c[0], reverse=True)

    assigned_slots: dict[str, str] = {}
    used_players: set[str] = set()
    for score, slot_id, player_id in candidates:
        if slot_id in assigned_slots or player_id in used_players:
            continue
        assigned_slots[slot_id] = player_id
        used_players.add(player_id)
        if len(assigned_slots) == len(formation_slots):
            break

    slots = [
        FormationSlot(slot_id=s["slot_id"], position=s["position"],
                       player_id=assigned_slots.get(s["slot_id"]))
        for s in formation_slots
    ]
    team = Team(team_id=team_id, name=name, formation=formation_name, slots=slots, version=1)
    team.avg_fit_score = team_avg_fit(team, lookup, templates, slot_position_map)
    team.chemistry_score = team_chemistry(team, lookup, chemistry_rules)
    return team


def _with_foot_bonus(template: PositionTemplate, foot: str | None) -> PositionTemplate:
    if not foot:
        return template
    t = copy.deepcopy(template)
    t.soft_bonuses = t.soft_bonuses + [
        SoftBonus(attribute="preferred_foot", value=foot, bonus_pct=6)
    ]
    return t
