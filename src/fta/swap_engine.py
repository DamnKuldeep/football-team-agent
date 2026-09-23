"""Rank the pool for one slot, given a swap request (typed by a human or
derived from match performance).

Policy: WARN AND RANK LOWER, NEVER BLOCK. A missed requirement (e.g. wrong
foot) becomes a warning plus a rank penalty, never an exclusion -- the human
always has the final say. Candidates are sorted by

    rank_score = fit_score - violation_penalty x violations + chemistry_weight x chemistry_delta

so a compliant player outranks a non-compliant one unless the latter is
genuinely better by more than the penalty.
"""
from __future__ import annotations

import copy

from .models import (
    ChemistryRule,
    Player,
    PositionTemplate,
    Shortlist,
    ShortlistCandidate,
    SwapRequest,
    Team,
)
from .scoring import Together, check_hard_filters, fit_score, team_chemistry


def _reweighted_template(base: PositionTemplate, request: SwapRequest) -> PositionTemplate:
    t = copy.deepcopy(base)
    if request.reweight:
        t.attribute_weights.update(request.reweight)
        total = sum(t.attribute_weights.values())
        if total > 0:
            t.attribute_weights = {k: v / total for k, v in t.attribute_weights.items()}
    t.hard_filters = t.hard_filters + request.hard_filters
    return t


def suggest_swap(
    request: SwapRequest,
    team: Team,
    pool: list[Player],
    templates: dict[str, PositionTemplate],
    chemistry_rules: list[ChemistryRule],
    top_n: int = 5,
    together: Together | None = None,
) -> Shortlist:
    lookup = {p.player_id: p for p in pool}
    target_slot = next(s for s in team.slots if s.slot_id == request.target_slot)
    base_template = templates[target_slot.position]
    local_template = _reweighted_template(base_template, request)

    used_ids = set(team.player_ids())
    current_player = lookup.get(target_slot.player_id) if target_slot.player_id else None
    current_fit = fit_score(current_player, base_template) if current_player else None
    # delta_fit compares like with like: incumbent and candidate both scored
    # against the request's (reweighted) template
    current_fit_for_request = fit_score(current_player, local_template) if current_player else 0.0
    current_chem = team_chemistry(team, lookup, chemistry_rules, together)

    # pass 1 (cheap, whole pool): fit and filter penalty
    scored: list[tuple[float, float, list[str], Player]] = []
    for player in pool:
        if player.player_id in used_ids or player.player_id in request.exclude_player_ids:
            continue
        if (target_slot.position == "GK") != player.is_gk():
            continue
        warnings = check_hard_filters(player, local_template.hard_filters)
        score = fit_score(player, local_template)
        scored.append((score - request.violation_penalty * len(warnings), score, warnings, player))

    # pass 2 (what-if chemistry, only where it could change the order): a single
    # swap moves chemistry by at most the sum of every rule's largest effect
    max_bonus = request.chemistry_weight * sum(max(abs(r.min_points), abs(r.max_points)) for r in chemistry_rules)
    scored.sort(key=lambda t: t[0], reverse=True)
    cutoff = scored[0][0] - 2 * max_bonus if scored else 0
    contenders = [t for t in scored if t[0] >= cutoff][:400]

    hypothetical = copy.deepcopy(team)
    target = next(s for s in hypothetical.slots if s.slot_id == request.target_slot)
    candidates: list[ShortlistCandidate] = []
    for pre_rank, score, warnings, player in contenders:
        target.player_id = player.player_id
        chem_delta = round(team_chemistry(hypothetical, lookup, chemistry_rules, together) - current_chem, 2)
        bonus = round(request.chemistry_weight * chem_delta, 2)
        candidates.append(ShortlistCandidate(
            player_id=player.player_id,
            fit_score=score,
            delta_fit=round(score - current_fit_for_request, 2),
            chemistry_delta=chem_delta,
            warnings=warnings,
            notes=_explain(player, current_player, local_template),
            penalty=round(score - pre_rank, 2),
            chemistry_bonus=bonus,
            rank_score=round(pre_rank + bonus, 2),
        ))

    candidates.sort(key=lambda c: (c.rank_score, c.fit_score), reverse=True)

    return Shortlist(
        target_slot=request.target_slot,
        current_player_id=current_player.player_id if current_player else None,
        current_fit_score=current_fit,
        candidates=candidates[:top_n],
    )


def _explain(candidate: Player, current: Player | None, template: PositionTemplate) -> str:
    """Cheap, deterministic 'expert note' -- arithmetic diffs turned into a
    sentence, never LLM-generated."""
    if current is None:
        return "No incumbent to compare against."
    diffs = []
    for attr, weight in sorted(template.attribute_weights.items(), key=lambda kv: -kv[1])[:3]:
        c_val = candidate.attr(attr)
        cur_val = current.attr(attr)
        delta = c_val - cur_val
        if abs(delta) >= 5:
            sign = "+" if delta > 0 else ""
            diffs.append(f"{sign}{delta:.0f} {attr}")
    if not diffs:
        return "Similar profile to the incumbent across the key attributes for this slot."
    return "Vs incumbent: " + ", ".join(diffs) + "."
