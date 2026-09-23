import pytest

from fta.models import (
    ChemistryRule,
    FormationSlot,
    HardFilter,
    Player,
    PositionTemplate,
    SoftBonus,
    Team,
)
from fta.scoring import check_hard_filters, chemistry_breakdown, fit_score, team_chemistry


def make_player(**overrides) -> Player:
    base = {
        "player_id": "p1", "name": "Test Player", "preferred_foot": "right", "weak_foot_rating": 3,
        "age": 25, "nationality": "XX", "stamina_base": 70,
        "attributes": {"tackling_standing": 80, "marking": 70, "heading_accuracy": 60,
                       "heading_power": 60, "positioning": 65, "passing_short": 50, "pace": 55},
        "natural_positions": ["CB"], "tags": [],
    }
    base.update(overrides)
    return Player.model_validate(base)


CB_TEMPLATE = PositionTemplate(
    position="CB", formation_slot="GENERIC",
    attribute_weights={"tackling_standing": 0.5, "marking": 0.5},
)


def test_fit_score_is_weighted_sum():
    p = make_player(attributes={"tackling_standing": 80, "marking": 60})
    score = fit_score(p, CB_TEMPLATE)
    assert score == pytest.approx(70.0, abs=0.01)


def test_fit_score_missing_attribute_uses_default_not_crash():
    template = PositionTemplate(
        position="X", formation_slot="GENERIC", attribute_weights={"vision": 1.0}
    )
    p = make_player(attributes={"tackling_standing": 80})  # no 'vision' key
    score = fit_score(p, template)
    assert score == pytest.approx(40.0)  # default fallback


def test_soft_bonus_applies_pct_multiplier():
    template = PositionTemplate(
        position="CB", formation_slot="GENERIC",
        attribute_weights={"tackling_standing": 1.0},
        soft_bonuses=[SoftBonus(attribute="preferred_foot", value="left", bonus_pct=10)],
    )
    left = make_player(preferred_foot="left", attributes={"tackling_standing": 50})
    right = make_player(preferred_foot="right", attributes={"tackling_standing": 50})
    assert fit_score(left, template) == pytest.approx(55.0)
    assert fit_score(right, template) == pytest.approx(50.0)


def test_hard_filter_foot_violation_reported():
    p = make_player(preferred_foot="right")
    filters = [HardFilter(attribute="preferred_foot", value="left")]
    violations = check_hard_filters(p, filters)
    assert len(violations) == 1
    assert "left" in violations[0]


def test_hard_filter_min_max_numeric():
    p = make_player(attributes={"pace": 55})
    filters = [HardFilter(attribute="pace", min=60)]
    assert check_hard_filters(p, filters) != []
    filters_ok = [HardFilter(attribute="pace", min=50)]
    assert check_hard_filters(p, filters_ok) == []


def test_weights_must_sum_to_one():
    with pytest.raises(ValueError):
        PositionTemplate(
            position="CB", formation_slot="GENERIC",
            attribute_weights={"tackling_standing": 0.3, "marking": 0.3},  # sums to 0.6
        )


def test_unknown_attribute_name_rejected():
    with pytest.raises(ValueError):
        PositionTemplate(
            position="CB", formation_slot="GENERIC",
            attribute_weights={"not_a_real_attribute": 1.0},
        )


RULES = [ChemistryRule(rule_id="familiarity", label="F", description="", max_points=8),
         ChemistryRule(rule_id="backline_balance", label="B", description="", max_points=4),
         ChemistryRule(rule_id="midfield_engine", label="M", description="", min_points=-6, max_points=4),
         ChemistryRule(rule_id="shared_language", label="S", description="", max_points=4)]


def _team(slots):
    return Team(team_id="t1", name="T", formation="4-3-3",
                slots=[FormationSlot(slot_id=s, position=pos, player_id=pid) for s, pos, pid in slots])


def test_midfield_engine_penalises_tired_midfields_and_rewards_fit_ones():
    tired = {"m1": make_player(player_id="m1", stamina_base=55, nationality="BR"),
             "m2": make_player(player_id="m2", stamina_base=60, nationality="IT")}
    fit = {"m1": make_player(player_id="m1", stamina_base=85, nationality="BR"),
           "m2": make_player(player_id="m2", stamina_base=80, nationality="IT")}
    team = _team([("DM", "DM", "m1"), ("CM_L", "CM", "m2")])
    assert team_chemistry(team, tired, RULES) == pytest.approx(94.0)
    assert team_chemistry(team, fit, RULES) == pytest.approx(104.0)


def test_backline_balance_counts_natural_sides():
    lookup = {"lb": make_player(player_id="lb", preferred_foot="left"),
              "cl": make_player(player_id="cl", preferred_foot="left"),
              "cr": make_player(player_id="cr", preferred_foot="left"),   # wrong side
              "rb": make_player(player_id="rb", preferred_foot="right")}
    team = _team([("LB", "FB", "lb"), ("CB_L", "CB", "cl"), ("CB_R", "CB", "cr"), ("RB", "FB", "rb")])
    _, items = chemistry_breakdown(team, lookup, RULES)
    assert next(i for i in items if i["rule_id"] == "backline_balance")["points"] == 3


def test_familiarity_grows_with_matches_played_together_and_caps():
    lookup = {"a": make_player(player_id="a"), "b": make_player(player_id="b")}
    team = _team([("CB_L", "CB", "a"), ("CB_R", "CB", "b")])       # one linked pair
    pair = frozenset(("a", "b"))
    points = [next(i for i in chemistry_breakdown(team, lookup, RULES, {pair: n})[1]
                   if i["rule_id"] == "familiarity")["points"] for n in (0, 1, 3, 10)]
    assert points[0] == 0 and 0 < points[1] < points[2] and points[2] == points[3] == 8


def test_shared_nationality_only_counts_neighbours():
    lookup = {"a": make_player(player_id="a", nationality="BR"), "b": make_player(player_id="b", nationality="BR"),
              "c": make_player(player_id="c", nationality="BR")}
    team = _team([("CB_L", "CB", "a"), ("CB_R", "CB", "b"), ("ST", "ST", "c")])   # ST isn't next to the CBs
    _, items = chemistry_breakdown(team, lookup, RULES)
    assert next(i for i in items if i["rule_id"] == "shared_language")["points"] == 1
