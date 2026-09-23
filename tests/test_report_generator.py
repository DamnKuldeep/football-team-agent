from fta.models import EventLog, MatchEvent, PlayerStats, Possession
from fta.report_generator import (
    aggregate_stats,
    build_reports,
    compute_rating,
    derive_swap_request,
    rating_breakdown,
    report_key,
)


def make_log() -> EventLog:
    return EventLog(
        match_id="m1", seed=1, team_a="t_A", team_b="t_B",
        final_score={"t_A": 1, "t_B": 0},
        lineups={"t_A": {"CM_L": "passer1", "AM": "creator1", "ST": "finisher1"},
                 "t_B": {"CM_L": "passer2", "CB_L": "defender1"}},
        possessions=[
            Possession(possession_id=1, attacking_team="t_A", chain=[
                MatchEvent(event="build_up_pass", player_id="passer1", success=True),
                MatchEvent(event="through_ball", player_id="creator1", success=True),
                MatchEvent(event="key_pass", player_id="creator1", success=True),
                MatchEvent(event="shot", player_id="finisher1", outcome="goal", xg=0.3),
            ]),
            Possession(possession_id=2, attacking_team="t_B", chain=[
                MatchEvent(event="build_up_pass", player_id="passer2", success=False),
                MatchEvent(event="turnover", player_id="defender1", success=True),
            ]),
        ],
    )


def test_aggregate_stats_exact_counts_from_hand_built_log():
    stats = aggregate_stats(make_log())  # old-style events without team_id: the side comes from the line-ups

    def a(pid):
        return stats[report_key("t_A", pid)]

    def b(pid):
        return stats[report_key("t_B", pid)]

    assert a("passer1").passes_attempted == 1 and a("passer1").passes_completed == 1
    assert a("creator1").passes_attempted == 1  # through_ball counted as a pass attempt
    assert a("creator1").key_passes == 1
    assert a("finisher1").shots == 1 and a("finisher1").goals == 1
    assert b("passer2").passes_attempted == 1 and b("passer2").passes_completed == 0
    assert b("defender1").tackles_won == 1


def test_seeded_zero_stat_players_are_present_even_with_no_events():
    stats = aggregate_stats(make_log(), starters={("t_A", "passer1"), ("t_B", "never_touched_ball")})
    assert stats[report_key("t_B", "never_touched_ball")].passes_attempted == 0


def test_a_player_on_both_sides_keeps_two_separate_games():
    """A team playing an older version of itself fields the same keeper twice.
    Side A wins 2-0: A's keeper gets the clean sheet and no goals against; B's
    keeper (the same person) concedes both -- never one merged, contradictory report."""
    goal = [MatchEvent(event="shot", player_id="st", team_id="A", outcome="goal", xg=0.4),
            MatchEvent(event="conceded", player_id="gk", team_id="B", success=False)]
    save = [MatchEvent(event="shot", player_id="st", team_id="B", outcome="saved", xg=0.2),
            MatchEvent(event="save", player_id="gk", team_id="A", success=True)]
    lineup = {f"S{i}": f"p{i}" for i in range(9)} | {"GK": "gk", "ST": "st"}
    positions = {s: "GK" if s == "GK" else "ST" if s == "ST" else "CM" for s in lineup}
    log = EventLog(match_id="m", seed=1, team_a="A", team_b="B", final_score={"A": 2, "B": 0},
                   lineups={"A": lineup, "B": lineup}, slot_positions={"A": positions, "B": positions},
                   possessions=[Possession(possession_id=i, attacking_team="B" if c is save else "A", chain=c)
                                for i, c in enumerate([goal, save, goal], 1)])
    reports = build_reports(log, {(t, pid) for t in ("A", "B") for pid in lineup.values()})
    assert sum(r.team_id == "A" for r in reports.values()) == 11 == sum(r.team_id == "B" for r in reports.values())
    gk_a, gk_b = reports[report_key("A", "gk")], reports[report_key("B", "gk")]
    assert (gk_a.stats.saves, gk_a.stats.goals_conceded) == (1, 0)
    assert (gk_b.stats.saves, gk_b.stats.goals_conceded) == (0, 2)
    terms_a = {k for k, *_ in rating_breakdown(gk_a.stats, "GK", 0)[1]}
    terms_b = {k for k, *_ in rating_breakdown(gk_b.stats, "GK", 2)[1]}
    assert "clean_sheet" in terms_a and "goals_conceded" not in terms_a
    assert "goals_conceded" in terms_b and "clean_sheet" not in terms_b
    assert gk_a.rating > gk_b.rating
    assert reports[report_key("A", "st")].stats.goals == 2 and reports[report_key("B", "st")].stats.goals == 0


def test_pass_accuracy_pct_zero_division_safe():
    s = PlayerStats(player_id="x")
    assert s.pass_accuracy_pct == 0.0


def test_compute_rating_rewards_goals_and_key_passes():
    quiet = PlayerStats(player_id="a")
    scorer = PlayerStats(player_id="b", goals=2, key_passes=3, passes_attempted=10, passes_completed=9)
    assert compute_rating(scorer) > compute_rating(quiet)


def test_compute_rating_bounded_0_to_10():
    extreme = PlayerStats(player_id="a", goals=20, key_passes=20, tackles_won=20, passes_completed=100)
    assert compute_rating(extreme) <= 10.0
    terrible = PlayerStats(player_id="b", passes_attempted=50, passes_completed=0, tackles_lost=20)
    assert compute_rating(terrible) >= 0.0


def test_derive_swap_request_triggers_on_weak_tackling():
    stats = PlayerStats(player_id="p1", tackles_lost=3, tackles_won=0)
    req = derive_swap_request("CB_L", "p1", stats)
    assert req is not None
    assert "tackling_standing" in req.reweight
    assert req.source == "performance_derived"


def test_derive_swap_request_triggers_on_weak_passing():
    stats = PlayerStats(player_id="p1", passes_attempted=10, passes_completed=3)
    req = derive_swap_request("CM_L", "p1", stats)
    assert req is not None
    assert "passing_short" in req.reweight


def test_derive_swap_request_none_when_no_clear_weakness():
    stats = PlayerStats(player_id="p1", passes_attempted=10, passes_completed=9, tackles_won=3)
    req = derive_swap_request("CB_L", "p1", stats)
    assert req is None


def test_ratings_and_feedback_follow_the_role():
    from fta.report_generator import player_feedback
    wall = PlayerStats(player_id="d", tackles_won=6, tackles_lost=1, aerial_duels_won=4)
    assert compute_rating(wall, "CB", team_conceded=0) > compute_rating(wall, "ST", team_conceded=0)
    good, bad = player_feedback(PlayerStats(player_id="d", shots=4, tackles_won=5), "CB")
    assert not any("shot" in line for line in good + bad)             # defenders aren't judged on shooting
    good, bad = player_feedback(PlayerStats(player_id="s", shots=4, xg=1.3, tackles_lost=5), "ST")
    assert any("goal" in line for line in bad) and not any("duel" in line for line in good + bad)


def test_keeper_rating_uses_saves_and_clean_sheet():
    keeper = PlayerStats(player_id="k", saves=5)
    assert compute_rating(keeper, "GK", team_conceded=0) > compute_rating(keeper, "GK", team_conceded=2)
