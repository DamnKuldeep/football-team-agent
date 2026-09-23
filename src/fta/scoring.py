"""Pure scoring functions. No LLM, no randomness -- every number here must be
traceable back to arithmetic on player attributes, so a swap suggestion can
always be explained rather than trusted on faith. The *_breakdown functions
return the same numbers itemised, for display.
"""
from __future__ import annotations

from .models import ChemistryRule, HardFilter, Player, PositionTemplate, SoftBonus, Team

# (player_a, player_b) -> number of matches they started together for the same team
Together = dict[frozenset, int]
FAMILIARITY_CAP = 3  # a pair counts as fully familiar after this many matches together


_FIELD_LABEL = {"age": "age", "weak_foot_rating": "weak foot"}


def describe_filter(f: HardFilter) -> str:
    """Human-readable requirement, e.g. 'age 20–24', 'left-footed', 'pace ≥ 80'."""
    label = _FIELD_LABEL.get(f.attribute, f.attribute.replace("_", " "))
    if f.attribute == "preferred_foot":
        return f"{f.value}-footed"
    if f.attribute == "nationality":
        return "nationality " + " or ".join(f.values or [f.value])
    if f.attribute == "natural_position":
        return f"natural {f.value}"
    if f.min is not None and f.max is not None:
        return f"{label} {f.min:g}–{f.max:g}"
    return f"{label} ≥ {f.min:g}" if f.min is not None else f"{label} ≤ {f.max:g}"


def _field_value(player: Player, field: str) -> float:
    if field == "age":
        return float(player.age)
    if field == "weak_foot_rating":
        return float(player.weak_foot_rating)
    return player.attr(field, default=float("nan"))


def check_hard_filters(player: Player, filters: list[HardFilter]) -> list[str]:
    """Returns a list of violation descriptions (empty = passes all filters).
    Callers decide whether to exclude or warn-and-allow on a violation."""
    violations = []
    for f in filters:
        if f.attribute == "preferred_foot":
            if f.value and player.preferred_foot != f.value:
                violations.append(f"requires {f.value}-footed, player is {player.preferred_foot}-footed")
        elif f.attribute == "nationality":
            wanted = [v for v in (f.values or [f.value]) if v]
            if wanted and player.nationality.lower() not in {v.lower() for v in wanted}:
                violations.append(f"requires nationality {' or '.join(wanted)}, player is {player.nationality}")
        elif f.attribute == "natural_position":
            if f.value and f.value not in player.natural_positions:
                violations.append(f"requires a natural {f.value}, player is {'/'.join(player.natural_positions)}")
        else:
            val = _field_value(player, f.attribute)
            label = _FIELD_LABEL.get(f.attribute, f.attribute.replace("_", " "))
            if f.min is not None and val < f.min:
                violations.append(f"{label} {val:.0f}, wanted ≥ {f.min:g}")
            if f.max is not None and val > f.max:
                violations.append(f"{label} {val:.0f}, wanted ≤ {f.max:g}")
    return violations


def _soft_bonus_pct(player: Player, bonuses: list[SoftBonus]) -> float:
    total_pct = 0.0
    for b in bonuses:
        if b.attribute == "preferred_foot":
            if b.value and player.preferred_foot == b.value:
                total_pct += b.bonus_pct
        else:
            # attribute-threshold style bonus: apply if attribute is above 70
            if player.attr(b.attribute, default=0) >= 70:
                total_pct += b.bonus_pct
    return total_pct


def fit_breakdown(player: Player, template: PositionTemplate) -> tuple[list[dict], float, float]:
    """(rows, bonus_pct, fit). Each row: attribute, weight, value, contribution
    (= weight x value); fit = sum(contributions) x (1 + bonus_pct/100).
    Missing attributes count as 40."""
    rows = [{"attribute": a, "weight": w, "value": player.attr(a, default=40.0),
             "contribution": player.attr(a, default=40.0) * w}
            for a, w in sorted(template.attribute_weights.items(), key=lambda kv: -kv[1])]
    bonus_pct = _soft_bonus_pct(player, template.soft_bonuses)
    fit = round(sum(r["contribution"] for r in rows) * (1 + bonus_pct / 100), 2)
    return rows, bonus_pct, fit


def fit_score(player: Player, template: PositionTemplate) -> float:
    """Weighted sum of attributes (0-100 each) against the template's
    weights (summing to 1.0), then soft bonuses applied as a percentage
    multiplier. Result is roughly 0-100+ (bonuses can push slightly over)."""
    return fit_breakdown(player, template)[2]


# Neighbours on the pitch, by slot id. Only pairs where both slots exist in a
# formation count, so the same list serves 4-3-3 and 4-2-3-1.
LINKS = [
    ("GK", "CB_L"), ("GK", "CB_R"), ("CB_L", "CB_R"), ("LB", "CB_L"), ("RB", "CB_R"),
    ("LB", "LW"), ("RB", "RW"), ("CB_L", "DM"), ("CB_R", "DM"), ("CB_L", "DM_L"), ("CB_R", "DM_R"),
    ("DM_L", "DM_R"), ("DM", "CM_L"), ("DM", "CM_R"), ("CM_L", "CM_R"), ("LB", "CM_L"), ("RB", "CM_R"),
    ("LB", "DM_L"), ("RB", "DM_R"), ("DM_L", "AM"), ("DM_R", "AM"), ("CM_L", "LW"), ("CM_R", "RW"),
    ("CM_L", "ST"), ("CM_R", "ST"), ("AM", "LW"), ("AM", "RW"), ("AM", "ST"), ("LW", "ST"), ("RW", "ST"),
]
MIDFIELD = ("DM", "CM", "AM")
BACKLINE_FEET = {"LB": "left", "CB_L": "left", "CB_R": "right", "RB": "right"}


def team_links(team: Team) -> list[tuple[str, str]]:
    slots = {s.slot_id for s in team.slots if s.player_id}
    return [(a, b) for a, b in LINKS if a in slots and b in slots]


def chemistry_breakdown(
    team: Team,
    lookup: dict[str, Player],
    rules: list[ChemistryRule],
    together: Together | None = None,
) -> tuple[float, list[dict]]:
    """(score, items). Base 100 plus each rule's points (clamped to the rule's
    min/max). Each item: rule_id, label, description, points, max, detail."""
    by_slot = {s.slot_id: lookup.get(s.player_id) for s in team.slots if s.player_id}
    positions = {s.slot_id: s.position for s in team.slots}
    links = team_links(team)
    together = together or {}
    items = []
    for rule in rules:
        if rule.rule_id == "familiarity":
            done = [min(together.get(frozenset((by_slot[a].player_id, by_slot[b].player_id)), 0),
                        FAMILIARITY_CAP) / FAMILIARITY_CAP for a, b in links if by_slot[a] and by_slot[b]]
            share = sum(done) / len(done) if done else 0.0
            points = rule.max_points * share
            detail = (f"{sum(d > 0 for d in done)} of {len(done)} links have played together · "
                      f"{share:.0%} of the way to full familiarity")
        elif rule.rule_id == "backline_balance":
            natural = [slot for slot, foot in BACKLINE_FEET.items()
                       if by_slot.get(slot) and by_slot[slot].preferred_foot == foot]
            points = float(len(natural))
            detail = f"natural side: {', '.join(natural) or 'none'} (of LB, CB_L, CB_R, RB)"
        elif rule.rule_id == "midfield_engine":
            mids = [p for slot, p in by_slot.items() if p and positions.get(slot) in MIDFIELD]
            avg = sum(p.stamina_base for p in mids) / len(mids) if mids else 70
            points = 4.0 if avg >= 78 else -6.0 if avg < 65 else 0.0
            detail = f"average midfield stamina {avg:.0f} (78+ → +4, under 65 → −6)"
        elif rule.rule_id == "shared_language":
            pairs = [f"{a}–{b}" for a, b in links if by_slot[a] and by_slot[b]
                     and by_slot[a].nationality == by_slot[b].nationality]
            points = float(len(pairs))
            detail = f"same-country neighbours: {', '.join(pairs) or 'none'}"
        else:
            continue
        points = round(max(rule.min_points, min(rule.max_points, points)), 2)
        items.append({"rule_id": rule.rule_id, "label": rule.label, "description": rule.description,
                      "points": points, "max": rule.max_points, "detail": detail})
    return round(100.0 + sum(i["points"] for i in items), 2), items


def team_chemistry(team: Team, lookup: dict[str, Player], rules: list[ChemistryRule],
                   together: Together | None = None) -> float:
    return chemistry_breakdown(team, lookup, rules, together)[0]


def team_avg_fit(team: Team, lookup: dict[str, Player], templates: dict[str, PositionTemplate],
                 slot_position_map: dict[str, str]) -> float:
    scores = []
    for slot in team.slots:
        if not slot.player_id:
            continue
        player = lookup.get(slot.player_id)
        if player is None:  # player from a pool that isn't loaded
            continue
        template = templates[slot_position_map[slot.slot_id]]
        scores.append(fit_score(player, template))
    return round(sum(scores) / len(scores), 2) if scores else 0.0
