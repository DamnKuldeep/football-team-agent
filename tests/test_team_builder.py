from fta.models import Player, PositionTemplate
from fta.team_builder import build_team


def make_player(pid, is_gk=False, **attrs) -> Player:
    all_attrs = {"tackling_standing": 40, "marking": 40, "heading_accuracy": 40,
                 "heading_power": 40, "positioning": 40, "passing_short": 40, "pace": 40,
                 "stamina": 60, "gk_reflexes": None, "gk_handling": None,
                 "gk_positioning": None, "gk_kicking": None}
    if is_gk:
        all_attrs.update({"gk_reflexes": 70, "gk_handling": 70, "gk_positioning": 70, "gk_kicking": 60})
    all_attrs.update(attrs)
    return Player(
        player_id=pid, name=pid, preferred_foot="right", weak_foot_rating=3, age=25,
        stamina_base=70, attributes=all_attrs,
        natural_positions=["GK"] if is_gk else ["CB"], tags=[],
    )


TEMPLATES = {
    "GK": PositionTemplate(position="GK", formation_slot="GENERIC",
                            attribute_weights={"gk_reflexes": 1.0}),
    "CB": PositionTemplate(position="CB", formation_slot="GENERIC",
                            attribute_weights={"tackling_standing": 0.5, "marking": 0.5}),
}
SIMPLE_FORMATION = [
    {"slot_id": "GK", "position": "GK", "foot_bonus": None},
    {"slot_id": "CB_L", "position": "CB", "foot_bonus": None},
    {"slot_id": "CB_R", "position": "CB", "foot_bonus": None},
]


def test_no_duplicate_players_across_slots():
    pool = [
        make_player("gk1", is_gk=True),
        make_player("cb1", tackling_standing=90, marking=90),
        make_player("cb2", tackling_standing=85, marking=85),
        make_player("cb3", tackling_standing=80, marking=80),
    ]
    team = build_team("t1", "T", "test", SIMPLE_FORMATION, pool, TEMPLATES, [])
    assigned = [s.player_id for s in team.slots]
    assert len(assigned) == len(set(assigned))
    assert all(assigned)


def test_every_slot_filled_when_enough_candidates():
    pool = [make_player("gk1", is_gk=True), make_player("cb1"), make_player("cb2")]
    team = build_team("t1", "T", "test", SIMPLE_FORMATION, pool, TEMPLATES, [])
    assert all(s.player_id is not None for s in team.slots)


def test_obvious_best_xi_is_selected():
    """A hand-crafted pool where one CB is clearly best should end up assigned,
    not left on the bench in favor of a worse player."""
    pool = [
        make_player("gk1", is_gk=True),
        make_player("best_cb", tackling_standing=95, marking=95),
        make_player("mid_cb", tackling_standing=60, marking=60),
        make_player("worst_cb", tackling_standing=30, marking=30),
    ]
    team = build_team("t1", "T", "test", SIMPLE_FORMATION, pool, TEMPLATES, [])
    assigned = {s.player_id for s in team.slots}
    assert "best_cb" in assigned
    assert "worst_cb" not in assigned


def test_gk_never_assigned_to_outfield_slot():
    pool = [make_player("gk1", is_gk=True), make_player("gk2", is_gk=True), make_player("cb1")]
    team = build_team("t1", "T", "test", SIMPLE_FORMATION, pool, TEMPLATES, [])
    slot_map = {s.slot_id: s.player_id for s in team.slots}
    assert slot_map["GK"] in {"gk1", "gk2"}
    assert slot_map["CB_L"] != slot_map["GK"]
