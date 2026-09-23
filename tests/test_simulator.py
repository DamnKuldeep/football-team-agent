from fta.data_loader import (
    load_chemistry_rules,
    load_formations,
    load_pool,
    load_position_templates,
)
from fta.models import ChemistryRule
from fta.simulator import simulate_match
from fta.team_builder import build_team


def _two_teams():
    pool = load_pool("synthetic")
    templates = load_position_templates()
    formations = load_formations()
    rules = [ChemistryRule.model_validate(r) for r in load_chemistry_rules()]
    team_a = build_team("t_A", "Alpha", "4-3-3", formations["4-3-3"], pool, templates, rules)
    used = set(team_a.player_ids())
    pool_b = [p for p in pool if p.player_id not in used]
    team_b = build_team("t_B", "Beta", "4-3-3", formations["4-3-3"], pool_b, templates, rules)
    lookup = {p.player_id: p for p in pool}
    return team_a, team_b, lookup


def test_same_seed_produces_identical_output():
    team_a, team_b, lookup = _two_teams()
    log1 = simulate_match("m1", team_a, team_b, lookup, seed=7, num_possessions=40)
    log2 = simulate_match("m1", team_a, team_b, lookup, seed=7, num_possessions=40)
    assert log1.model_dump_json() == log2.model_dump_json()


def test_different_seed_can_diverge():
    team_a, team_b, lookup = _two_teams()
    log1 = simulate_match("m1", team_a, team_b, lookup, seed=1, num_possessions=40)
    log2 = simulate_match("m1", team_a, team_b, lookup, seed=2, num_possessions=40)
    assert log1.model_dump_json() != log2.model_dump_json()


def test_score_always_non_negative():
    team_a, team_b, lookup = _two_teams()
    log = simulate_match("m1", team_a, team_b, lookup, seed=42, num_possessions=60)
    assert all(v >= 0 for v in log.final_score.values())


def test_every_event_player_is_on_one_of_the_two_teams():
    team_a, team_b, lookup = _two_teams()
    log = simulate_match("m1", team_a, team_b, lookup, seed=42, num_possessions=60)
    valid_ids = set(team_a.player_ids()) | set(team_b.player_ids())
    for poss in log.possessions:
        for event in poss.chain:
            assert event.player_id in valid_ids


def test_possession_count_matches_request():
    team_a, team_b, lookup = _two_teams()
    log = simulate_match("m1", team_a, team_b, lookup, seed=5, num_possessions=25)
    assert len(log.possessions) == 25


def test_chemistry_shifts_results_over_many_matches():
    team_a, team_b, lookup = _two_teams()

    def goals(chem_a):
        return sum(simulate_match("m", team_a, team_b, lookup, seed=s,
                                  chemistry={"t_A": chem_a, "t_B": 100}).final_score["t_A"] for s in range(150))
    assert goals(120) > goals(80)
