from fta.models import FormationSlot, HardFilter, Player, PositionTemplate, SwapRequest, Team
from fta.swap_engine import suggest_swap


def make_player(pid, foot="right", **attrs) -> Player:
    all_attrs = {"tackling_standing": 50, "marking": 50, "heading_accuracy": 50}
    all_attrs.update(attrs)
    return Player(
        player_id=pid, name=pid, preferred_foot=foot, weak_foot_rating=3, age=25,
        stamina_base=70, attributes=all_attrs, natural_positions=["CB"], tags=[],
    )


TEMPLATE = PositionTemplate(
    position="CB", formation_slot="GENERIC",
    attribute_weights={"tackling_standing": 0.5, "marking": 0.5},
)
TEMPLATES = {"CB": TEMPLATE}


def make_team(current_id="incumbent"):
    return Team(
        team_id="t1", name="T", formation="test",
        slots=[FormationSlot(slot_id="CB_L", position="CB", player_id=current_id)],
    )


def test_hard_filter_violation_is_warning_not_exclusion():
    team = make_team()
    pool = [
        make_player("incumbent", foot="right", tackling_standing=60, marking=60),
        make_player("only_option", foot="right", tackling_standing=90, marking=90),
    ]
    request = SwapRequest(target_slot="CB_L", hard_filters=[HardFilter(attribute="preferred_foot", value="left")])
    shortlist = suggest_swap(request, team, pool, TEMPLATES, [])
    ids = [c.player_id for c in shortlist.candidates]
    assert "only_option" in ids  # NOT excluded despite violating the hard filter
    candidate = next(c for c in shortlist.candidates if c.player_id == "only_option")
    assert len(candidate.warnings) == 1
    assert "left" in candidate.warnings[0]


def test_shortlist_sorted_by_fit_descending():
    team = make_team()
    pool = [
        make_player("incumbent", tackling_standing=50, marking=50),
        make_player("weak", tackling_standing=30, marking=30),
        make_player("strong", tackling_standing=90, marking=90),
        make_player("mid", tackling_standing=60, marking=60),
    ]
    request = SwapRequest(target_slot="CB_L")
    shortlist = suggest_swap(request, team, pool, TEMPLATES, [])
    scores = [c.fit_score for c in shortlist.candidates]
    assert scores == sorted(scores, reverse=True)


def test_current_incumbent_excluded_from_own_shortlist():
    team = make_team(current_id="incumbent")
    pool = [make_player("incumbent", tackling_standing=99, marking=99), make_player("other")]
    request = SwapRequest(target_slot="CB_L")
    shortlist = suggest_swap(request, team, pool, TEMPLATES, [])
    ids = [c.player_id for c in shortlist.candidates]
    assert "incumbent" not in ids


def test_reweight_changes_ranking():
    team = make_team()
    pool = [
        make_player("incumbent", tackling_standing=50, marking=50, heading_accuracy=50),
        make_player("header_specialist", tackling_standing=30, marking=30, heading_accuracy=95),
        make_player("tackler", tackling_standing=95, marking=95, heading_accuracy=30),
    ]
    # reweight is merged additively into the base template's weights, then
    # renormalized -- so it needs to be large relative to the existing
    # weights to actually flip the ranking, not just present.
    request = SwapRequest(target_slot="CB_L", reweight={"heading_accuracy": 5.0})
    shortlist = suggest_swap(request, team, pool, TEMPLATES, [])
    assert shortlist.candidates[0].player_id == "header_specialist"


def test_excluded_player_ids_respected():
    team = make_team()
    pool = [make_player("incumbent"), make_player("excluded_one", tackling_standing=99, marking=99)]
    request = SwapRequest(target_slot="CB_L", exclude_player_ids=["excluded_one"])
    shortlist = suggest_swap(request, team, pool, TEMPLATES, [])
    ids = [c.player_id for c in shortlist.candidates]
    assert "excluded_one" not in ids
