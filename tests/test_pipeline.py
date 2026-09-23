"""Multi-slot swaps, random-seed matches, and deletion -- the pipeline
behaviours both front ends rely on."""
import pytest

from fta import pipeline, storage


@pytest.fixture(autouse=True)
def isolate_state(tmp_path, monkeypatch):
    monkeypatch.setattr(storage, "TEAMS_DIR", tmp_path / "teams")
    monkeypatch.setattr(storage, "INDEX_PATH", tmp_path / "teams" / "index.json")
    monkeypatch.setattr(pipeline, "MATCHES_DIR", tmp_path / "matches")
    monkeypatch.setattr(pipeline, "SCOUTING_DIR", tmp_path / "scouting")


@pytest.fixture
def ctx():
    return pipeline.load_context("synthetic")


@pytest.fixture
def two_teams(ctx):
    a, _ = pipeline.build_team(ctx, "t_A", "Alpha")
    b, _ = pipeline.build_team(ctx, "t_B", "Beta", exclude_team_id="t_A")
    return a, b


def _bench(ctx, team, gk=False):
    used = set(storage.load_team("t_A").player_ids()) | set(storage.load_team("t_B").player_ids())
    return [p.player_id for p in ctx.pool if p.player_id not in used and p.is_gk() == gk]


def test_several_changes_saved_as_one_version(ctx, two_teams):
    a, _ = two_teams
    bench = _bench(ctx, a)
    new_team, diff = pipeline.apply_swaps(ctx, "t_A", {"CB_L": bench[0], "ST": bench[1]})
    assert new_team.version == 2 and storage.list_teams()["t_A"]["latest_version"] == 2
    assert {c["slot"] for c in diff["changes"]} == {"CB_L", "ST"}
    slots = {s.slot_id: s.player_id for s in storage.load_team("t_A").slots}
    assert slots["CB_L"] == bench[0] and slots["ST"] == bench[1]


def test_preview_saves_nothing_and_rejects_duplicates(ctx, two_teams):
    a, _ = two_teams
    bench = _bench(ctx, a)
    preview = pipeline.preview_swaps(ctx, a, {"CB_L": bench[0]})
    assert preview.version == 1 and storage.list_teams()["t_A"]["latest_version"] == 1
    with pytest.raises(ValueError, match="more than one slot"):
        pipeline.preview_swaps(ctx, a, {"CB_L": bench[0], "CB_R": bench[0]})


def test_matches_get_fresh_random_seeds_but_a_seed_replays_exactly(ctx, two_teams):
    m1 = pipeline.simulate(ctx, "t_A", "t_B")
    m2 = pipeline.simulate(ctx, "t_A", "t_B")
    assert m1.seed != m2.seed
    replay = pipeline.simulate(ctx, "t_A", "t_B", seed=m1.seed)
    assert replay.possessions == m1.possessions and replay.final_score == m1.final_score
    assert [p.minute for p in m1.possessions] == list(range(1, 91))


def test_delete_match_and_team_cascade(ctx, two_teams):
    m1 = pipeline.simulate(ctx, "t_A", "t_B").match_id
    pipeline.build_report(m1)
    pipeline.delete_match(m1)
    assert pipeline.list_matches() == [] and pipeline.load_report(m1) is None

    m2 = pipeline.simulate(ctx, "t_A", "t_B").match_id
    assert pipeline.delete_team("t_A") == [m2]
    assert "t_A" not in storage.list_teams() and pipeline.list_matches() == []
    assert "t_B" in storage.list_teams()


def test_request_filters_rank_lower_but_never_hide(ctx, two_teams):
    a, _ = two_teams
    request, _ = pipeline.swap_request_from_brief(ctx, a, "ST", "a finisher under 21")
    assert any(f.attribute == "age" and f.max == 20 for f in request.hard_filters)
    shortlist = pipeline.shortlist(ctx, a, request, top_n=10)
    ok = [c for c in shortlist.candidates if not c.warnings]
    missed = [c for c in shortlist.candidates if c.warnings]
    assert missed and all("age" in w for c in missed for w in c.warnings)
    if ok and missed:  # a compliant player ranks above one that misses by more than his fit edge
        assert shortlist.candidates.index(ok[0]) < shortlist.candidates.index(missed[0]) or \
            missed[0].fit_score - missed[0].penalty > ok[0].fit_score


def test_scouting_assessment_is_deterministic_and_explains_verdict(ctx):
    pid = ctx.pool[0].player_id
    a1, a2 = pipeline.scouting_assessment(ctx, pid), pipeline.scouting_assessment(ctx, pid)
    assert a1 == a2
    assert a1["verdict"] in ("Sign", "Short-term signing", "Monitor", "Pass") and a1["verdict_rule"]
    assert all(0 <= c["score"] <= 10 and 0 < c["share"] <= 1 for c in a1["criteria"])
    record = pipeline.generate_scouting(ctx, pid)  # offline assistant (no key in tests)
    assert record["source"] == "offline" and record["assessment"] == a1
    assert record["analysis"]["summary"] and pipeline.load_scouting(pid)["analysis"] == record["analysis"]


def test_best_fits_respects_weights_and_filters(ctx):
    top = pipeline.best_fits(ctx, {"pace": 1.0}, top_n=5)
    paces = [ctx.lookup[r["player_id"]].attr("pace") for r in top]
    assert paces == sorted(paces, reverse=True)                         # pure pace ranking
    from fta.models import HardFilter
    young = pipeline.best_fits(ctx, {"pace": 1.0}, filters=[HardFilter(attribute="age", max=21)], top_n=5)
    assert not young[0]["issues"] and ctx.lookup[young[0]["player_id"]].age <= 21


def test_model_setting_and_legacy_tier(monkeypatch):
    from fta.config import llm_model
    monkeypatch.delenv("MODEL", raising=False)
    monkeypatch.setenv("MODEL_TIER", "free")                     # old setting still works
    assert llm_model() == "deepseek/deepseek-chat"
    monkeypatch.setenv("MODEL", "anthropic/claude-haiku-4.5")
    assert llm_model() == "anthropic/claude-haiku-4.5"


def test_a_team_can_play_an_older_version_of_itself(ctx, two_teams):
    a, _ = two_teams
    bench = _bench(ctx, a)
    pipeline.apply_swaps(ctx, "t_A", {"ST": bench[0]})
    log = pipeline.simulate(ctx, "t_A", "t_A", version_a=2, version_b=1)
    assert {log.team_a, log.team_b} == {"t_A@v2", "t_A@v1"}
    assert log.lineups["t_A@v2"]["ST"] == bench[0] != log.lineups["t_A@v1"]["ST"]
    assert "v1" in log.label and "v2" in log.label
    with pytest.raises(ValueError):
        pipeline.simulate(ctx, "t_A", "t_A")                      # same version twice
    assert pipeline.delete_team("t_A") == [log.match_id]          # versioned ids still belong to t_A


def test_draft_test_uses_fresh_seeds_and_reports_a_margin(ctx, two_teams):
    a, b = two_teams
    bench = _bench(ctx, a)
    draft = pipeline.preview_swaps(ctx, a, {"ST": bench[0]})
    r1 = pipeline.evaluate_lineups(ctx, {"saved": a, "draft": draft}, b, n=60)
    r2 = pipeline.evaluate_lineups(ctx, {"saved": a, "draft": draft}, b, n=60)
    assert r1["variants"] != r2["variants"]                       # new seeds every run
    d = r1["difference"]
    assert d["margin"] > 0 and d["verdict"] in ("better", "worse", "no clear difference")
    same = pipeline.evaluate_lineups(ctx, {"saved": a, "again": a}, b, n=30)["difference"]
    assert same["points"] == 0 and same["verdict"] == "no clear difference"   # paired: identical XIs tie


def test_scouting_only_judges_what_the_role_needs(ctx):
    by_pos = {pos: next(p for p in ctx.pool if p.natural_positions[0] == pos) for pos in ("CB", "ST", "GK")}
    names = {pos: {c["name"] for c in pipeline.scouting_assessment(ctx, p.player_id)["criteria"]}
             for pos, p in by_pos.items()}
    assert "Finishing" not in names["CB"] and "Defending" in names["CB"]
    assert "Defending" not in names["ST"] and "Finishing" in names["ST"]
    assert names["GK"] <= {"Shot-stopping", "Handling", "Positioning", "Distribution", "Composure"}


def test_match_form_joins_scouting_and_grows(ctx, two_teams):
    a, _ = two_teams
    pid = a.slots[5].player_id
    assert not any(c["name"] == "Match form" for c in pipeline.scouting_assessment(ctx, pid)["criteria"])
    shares = []
    for _ in range(2):
        pipeline.simulate(ctx, "t_A", "t_B")
        shares.append(next(c["share"] for c in pipeline.scouting_assessment(ctx, pid)["criteria"]
                           if c["name"] == "Match form"))
    assert 0 < shares[0] < shares[1] <= pipeline.FORM_MAX_SHARE


def test_debrief_is_grounded_and_complete(ctx, two_teams):
    log = pipeline.simulate(ctx, "t_A", "t_B")
    record = pipeline.match_debrief(ctx, log.match_id)
    for tid in (log.team_a, log.team_b):
        d = record[tid]
        assert d["summary"] and d["worked"] and d["didnt"] and d["changes"]


def test_every_starter_of_both_sides_gets_one_consistent_report(ctx, two_teams):
    """Including a team against its own older version, where the same players
    appear on both sides: 11 reports per side, and a keeper's goals conceded
    always equal the opponent's score (so never a clean sheet with goals against)."""
    a, _ = two_teams
    pipeline.apply_swaps(ctx, "t_A", {"ST": _bench(ctx, a)[0]})
    for _ in range(3):
        for log in (pipeline.simulate(ctx, "t_A", "t_B"),
                    pipeline.simulate(ctx, "t_A", "t_A", version_a=2, version_b=1)):
            reports = pipeline.load_report(log.match_id)
            for tid, opp in ((log.team_a, log.team_b), (log.team_b, log.team_a)):
                side = pipeline.team_reports(reports, tid)
                assert set(side) == set(log.lineups[tid].values()) and len(side) == 11
                gk = next(r for r in side.values() if r.position == "GK")
                assert gk.stats.goals_conceded == log.final_score[opp]
                assert sum(r.stats.goals for r in side.values()) == log.final_score[tid]


def test_old_single_side_report_files_are_rebuilt(ctx, two_teams):
    import json
    log = pipeline.simulate(ctx, "t_A", "t_B")
    path = pipeline.match_dir(log.match_id) / "report.json"
    old = {r.player_id: r.model_dump() for r in pipeline.load_report(log.match_id).values()}
    path.write_text(json.dumps(old), encoding="utf-8")
    assert len(pipeline.load_report(log.match_id)) == 22


def test_team_history_follows_every_version(ctx, two_teams):
    a, _ = two_teams
    pipeline.simulate(ctx, "t_A", "t_B")
    pipeline.apply_swaps(ctx, "t_A", {"ST": _bench(ctx, a)[0]})
    pipeline.simulate(ctx, "t_B", "t_A")
    h = pipeline.team_history(ctx, "t_A")
    assert [m["version"] for m in h["matches"]] == [1, 2] and sum(h["record"].values()) == 2
    keeper = next(p for p in h["players"] if p["player_id"] == a.slots[0].player_id)   # played both
    old_st = next(p for p in h["players"] if p["player_id"] == next(s.player_id for s in a.slots
                                                                     if s.slot_id == "ST"))
    assert keeper["apps"] == 2 and old_st["apps"] == 1


def test_deleting_a_version_never_reuses_its_number(ctx, two_teams):
    a, _ = two_teams
    bench = _bench(ctx, a)
    pipeline.apply_swaps(ctx, "t_A", {"ST": bench[0]})               # v2
    log = pipeline.simulate(ctx, "t_A", "t_B")                      # played by v2
    pipeline.delete_version("t_A", 2)
    assert storage.list_versions("t_A") == [1] and storage.list_teams()["t_A"]["latest_version"] == 1
    new, _ = pipeline.apply_swaps(ctx, "t_A", {"ST": bench[1]})
    assert new.version == 3                                          # not a second "v2"
    assert pipeline.load_event_log(log.match_id).lineups["t_A"]["ST"] == bench[0]   # match still readable
    with pytest.raises(ValueError):
        pipeline.delete_version("t_B", 1)                            # only version: delete the team instead


def test_request_context_describes_the_team(ctx, two_teams):
    a, _ = two_teams
    c = pipeline.request_context(ctx, a, "CB_L")
    assert c["current_player"]["attributes"] and len(c["lineup"]) == 11
    assert "CB_R" in c["neighbours"] and c["pool"]["nationalities"]
