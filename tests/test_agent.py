"""The assistant: tools, verification, fact-checking, self-correction and the
offline fallbacks. The model is stubbed -- these tests never call OpenRouter."""
import json

import pytest

from fta import agent, pipeline, storage
from fta.models import HardFilter


@pytest.fixture(autouse=True)
def isolate_state(tmp_path, monkeypatch):
    monkeypatch.setattr(storage, "TEAMS_DIR", tmp_path / "teams")
    monkeypatch.setattr(storage, "INDEX_PATH", tmp_path / "teams" / "index.json")
    monkeypatch.setattr(pipeline, "MATCHES_DIR", tmp_path / "matches")
    monkeypatch.setattr(pipeline, "SCOUTING_DIR", tmp_path / "scouting")
    monkeypatch.setattr(agent, "EVAL_MATCHES", 20)   # keep the measured simulations quick


@pytest.fixture
def ctx():
    return pipeline.load_context("synthetic")


@pytest.fixture
def teams(ctx):
    a, _ = pipeline.build_team(ctx, "t_A", "Alpha")
    b, _ = pipeline.build_team(ctx, "t_B", "Beta", exclude_team_id="t_A")
    return a, b


def _bench(ctx, gk=False):
    used = set(storage.load_team("t_A").player_ids()) | set(storage.load_team("t_B").player_ids())
    return [p for p in ctx.pool if p.player_id not in used and p.is_gk() == gk]


class FakeModel:
    """Replays scripted assistant messages and records what it was sent."""

    def __init__(self, replies):
        self.replies, self.sent = list(replies), []

    def __call__(self, model, messages, tools=None, max_tokens=800, json_mode=False, tool_choice="auto"):
        self.sent.append([dict(m) for m in messages])
        self.choices = [*getattr(self, "choices", []), tool_choice]
        return self.replies.pop(0), 100, 50


def tool_call(name, args, call_id="c1"):
    return {"role": "assistant", "content": "", "tool_calls": [
        {"id": call_id, "type": "function", "function": {"name": name, "arguments": json.dumps(args)}}]}


def final(obj):
    return {"role": "assistant", "content": json.dumps(obj)}


# --- tools ------------------------------------------------------------------

def test_search_players_applies_requirements_and_skips_the_team(ctx, teams):
    a, _ = teams
    box = agent.Toolbox(ctx, "t_A")
    nats = sorted({p.nationality for p in ctx.pool if "CB" in p.natural_positions})[:2]
    out = box.call("search_players", {"position": "CB", "emphasis": {"heading_accuracy": 0.4},
                                      "requirements": [{"field": "nationality", "values": nats}], "limit": 5})
    ids = [c["player_id"] for c in out["candidates"]]
    assert not set(ids) & set(a.player_ids())
    met = [c for c in out["candidates"] if c["meets_requirements"]]
    assert met and all(c["nationality"] in nats for c in met)
    assert out["requirements"] == ["nationality " + " or ".join(nats)]
    assert box.trace[-1]["tool"] == "search_players"


def test_only_declared_tools_can_be_called(ctx, teams):
    box = agent.Toolbox(ctx, "t_A")
    assert "error" in box.call("team", {"team_id": "t_A"})          # an internal helper, not a tool
    assert "error" in box.call("get_player", {"player_id": "nobody"})
    assert "error" in box.call("search_players", {"position": "CB", "requirements": [{"field": "shoe_size"}]})


def test_evaluate_changes_is_measured_by_code(ctx, teams):
    new_st = _bench(ctx)[0].player_id
    out = agent.Toolbox(ctx, "t_A").call("evaluate_changes", {"changes": [{"slot": "ST", "player_id": new_st}]})
    assert out["opponent"] == "Beta" and out["matches"] == 20
    assert out["verdict"] in ("better", "worse", "no clear difference") and out["margin"] >= 0
    assert out["changes"][0]["in_player_id"] == new_st


# --- verification -----------------------------------------------------------

def test_check_changes_catches_every_kind_of_mistake(ctx, teams):
    a, _ = teams
    outfield, keeper = _bench(ctx)[0], _bench(ctx, gk=True)[0]
    assert agent.check_changes(ctx, a, {"ST": outfield.player_id}) == []
    assert "no such slot" in agent.check_changes(ctx, a, {"XX": outfield.player_id})[0]
    assert "doesn't exist" in agent.check_changes(ctx, a, {"ST": "p_9999"})[0]
    assert "keeper" in agent.check_changes(ctx, a, {"ST": keeper.player_id})[0]
    in_team = next(s.player_id for s in a.slots if s.slot_id == "CM_L")
    assert "already in the team" in agent.check_changes(ctx, a, {"ST": in_team})[0]
    assert "not a change" in agent.check_changes(ctx, a, {"CM_L": in_team})[0]
    assert any("two slots" in p for p in agent.check_changes(ctx, a, {"ST": outfield.player_id,
                                                                      "LW": outfield.player_id}))
    too_old = HardFilter(attribute="age", max=outfield.age - 1)
    assert "age" in agent.check_changes(ctx, a, {"ST": outfield.player_id}, [too_old])[0]


def test_fact_check_flags_invented_numbers_and_players(ctx, teams):
    box = agent.Toolbox(ctx, "t_A")
    box.call("get_team", {})
    names, firsts = agent._names(ctx)
    real = storage.load_team("t_A")
    p = ctx.lookup[real.slots[1].player_id]
    fit = next(x["fit"] for x in box.call("get_team", {})["lineup"] if x["player_id"] == p.player_id)
    ok = [f"{p.name} fits at {fit:.1f}, rounded {round(fit)}.", "Alpha play 4-3-3."]
    assert agent.fact_check(ok, box, names, firsts) == []
    fake_first = p.name.split()[0]
    bad = agent.fact_check([f"{p.name} fits at 12345.6.", f"{fake_first} Zzyzx should start."], box, names, firsts)
    assert len(bad) == 2 and "12345.6" in bad[0] and "Zzyzx" in bad[1]


# --- the loop ---------------------------------------------------------------

def test_the_assistant_corrects_itself_when_the_checker_objects(ctx, teams, monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "test")
    good = _bench(ctx)[0]
    answer = {"summary": "Swap the striker.", "requirements": [],
              "changes": [{"slot": "CF", "player_id": good.player_id, "reason": "Better finisher."}],
              "confidence": "medium", "caveats": []}
    fixed = {**answer, "changes": [{**answer["changes"][0], "slot": "ST"}]}
    model = FakeModel([
        final(answer),                                                        # wrong slot name
        final(fixed),                                                         # right slot, but untested
        tool_call("evaluate_changes", {"changes": [{"slot": "ST", "player_id": good.player_id}]}),
        final(fixed),
    ])
    monkeypatch.setattr(agent, "chat_message", model)
    out = agent.recommend_changes(ctx, "t_A", "a sharper striker", model="stub")
    assert out["source"] == "llm:stub" and out["problems"] == []
    assert [(c["slot"], c["player_id"]) for c in out["changes"]] == [("ST", good.player_id)]
    assert "team (from get_team)" in model.sent[0][1]["content"]           # the team came with the brief
    corrections = [m[-1]["content"] for m in model.sent[1:3]]
    assert "CF: no such slot" in corrections[0]                             # the code told it what was wrong
    assert "evaluate_changes" in corrections[1]                             # ... and that it must test it
    assert out["measured"]["changes"][0]["in_player_id"] == good.player_id  # numbers come from code
    assert [t["tool"] for t in out["trace"]] == ["get_team", "evaluate_changes"]


def test_match_fixes_must_be_tested_and_cannot_hide_a_slot(ctx, teams, monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "test")
    log = pipeline.simulate(ctx, "t_A", "t_B")
    box = agent.Toolbox(ctx, match_options=True)
    options = [o for side in box.call("get_match", {"match_id": log.match_id})["sides"].values()
               for o in side["replacement_options"]]
    for o in options:                                           # offered options are already tested
        assert box.tested(o["team_id"], {o["slot"]: o["player_id"]})
    saved = {t: storage.load_team(t) for t in storage.list_teams()}
    hidden = {"team_id": "t_A", "slot": None, "player_id": None, "reason": "Our RB was poor; find a better RB."}
    assert "names RB" in agent._check_recommendation(ctx, hidden, saved, box)[0]
    untested = {"team_id": "t_A", "slot": "ST", "player_id": _bench(ctx)[0].player_id, "reason": "Sharper."}
    assert "evaluate_changes" in agent._check_recommendation(ctx, untested, saved, box)[0]
    general = {"team_id": "t_A", "slot": None, "player_id": None, "reason": "Press higher as a team."}
    assert agent._check_recommendation(ctx, general, saved, box) == []


def test_a_model_that_never_stops_is_made_to_answer(ctx, teams, monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "test")
    wander = [tool_call("get_player", {"player_id": ctx.pool[0].player_id}, f"c{i}")
              for i in range(agent.MAX_STEPS - 1)]
    done = final({"summary": "Nothing clearly helps.", "requirements": [], "changes": [], "confidence": "low",
                  "caveats": []})
    model = FakeModel([*wander, done])
    monkeypatch.setattr(agent, "chat_message", model)
    out = agent.recommend_changes(ctx, "t_A", "a better team", model="stub")
    assert out["source"] == "llm:stub" and out["changes"] == []
    assert model.choices[-1] == "none" and set(model.choices[:-1]) == {"auto"}   # last turn: no tools allowed
    assert any("Two turns left" in m["content"] for m in model.sent[-1] if m["role"] == "user")


def test_trend_claims_need_the_history(ctx, teams):
    box = agent.Toolbox(ctx, "t_A")
    claim = ["Bypassed in 3 duels, a recurring weakness."]
    assert agent.trend_check(claim, box)                          # not checked -> sent back
    assert agent.trend_check(["Bypassed in 3 duels today."], box) == []
    box.call("team_history", {"team_id": "t_A"})
    assert agent.trend_check(claim, box) == []                    # checked -> allowed


def test_claims_about_the_record_must_match_it():
    history = {"matches": [{"result": "W", "goals_against": 0}, {"result": "D", "goals_against": 1}]}
    assert agent.record_check(["Clean sheets in both matches."], history)
    assert agent.record_check(["They kept two clean sheets."], history)
    assert agent.record_check(["Won every game so far."], history)
    assert agent.record_check(["Winless in this spell."], history)
    assert agent.record_check(["Unbeaten, with one clean sheet from two games."], history) == []
    assert agent.record_check(["Unbeaten."], {"matches": [*history["matches"], {"result": "L", "goals_against": 2}]})


def test_goals_are_about_the_positions_they_name():
    g = agent.goal_positions
    assert g("a better keeper, same nationality as at least one centre-back") == {"GK"}   # a reference, not a target
    assert g("centre-backs, holding midfielder and striker under 30") == {"CB", "DM", "ST"}
    assert g("attacking wing-backs who cross") == {"FB"}
    assert g("our midfield fades after the hour") == {"DM", "CM", "AM"}
    assert g("we lose too many headers at the back") == {"CB", "FB"}


def test_offline_planner_never_picks_one_player_twice(ctx, teams):
    out = agent.recommend_changes(ctx, "t_A", "more pace everywhere in attack and midfield")
    ids = [c["player_id"] for c in out["changes"]]
    assert len(ids) == len(set(ids)) and not out["dropped"]


def test_a_spent_budget_stops_ai_calls_before_they_cost_anything(ctx, teams, monkeypatch, tmp_path):
    from fta import cost_tracker
    monkeypatch.setattr(cost_tracker, "LOG_PATH", tmp_path / "costs.jsonl")
    monkeypatch.setenv("OPENROUTER_API_KEY", "test")
    model = FakeModel([])                                             # any call would fail: none must happen
    monkeypatch.setattr(agent, "chat_message", model)
    tracker = cost_tracker.CostTracker(budget_usd=0.0, mode="block")
    out = agent.recommend_changes(ctx, "t_A", "stronger in the air at the back", model="stub", tracker=tracker)
    assert model.sent == [] and out["source"] == "offline" and "spending limit" in out["error"]


def test_limits_are_read_exactly():
    assert agent._bound_problems({"field": "age", "max": 30, "quote": "under 30"})
    assert not agent._bound_problems({"field": "age", "max": 29, "quote": "under 30"})
    assert agent._bound_problems({"field": "age", "min": 30, "quote": "older than 30"})
    assert not agent._bound_problems({"field": "age", "min": 30, "quote": "30 or older"})


def test_unfixable_claims_are_dropped_or_flagged_never_trusted(ctx, teams, monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "test")
    good = _bench(ctx)[0]
    answer = {"summary": "This lifts us by 42.42 points a match.",
              "requirements": [{"field": "age", "max": 25, "quote": "must be young and cheap"}],   # not in the goal
              "changes": [{"slot": "ST", "player_id": "p_9999", "reason": "Invented player."},
                          {"slot": "LW", "player_id": good.player_id, "reason": "Quick."}],
              "confidence": "high", "caveats": []}
    monkeypatch.setattr(agent, "chat_message", FakeModel([final(answer)] * 3))
    out = agent.recommend_changes(ctx, "t_A", "more pace on the left wing", model="stub")
    assert [c["player_id"] for c in out["changes"]] == [good.player_id]      # invented player removed
    assert any("p_9999" in d for d in out["dropped"])
    assert out["requirements"] == []                                         # ungrounded requirement ignored
    assert any("42.42" in p for p in out["problems"])                        # invented number flagged


def test_api_failure_falls_back_to_the_offline_planner(ctx, teams, monkeypatch):
    import requests
    monkeypatch.setenv("OPENROUTER_API_KEY", "test")

    def broken(*a, **k):
        raise requests.ConnectionError("down")

    monkeypatch.setattr(agent, "chat_message", broken)
    out = agent.recommend_changes(ctx, "t_A", "stronger in the air at the back", model="stub")
    assert out["source"] == "offline" and out["error"]
    assert all(pipeline.slot_position(storage.load_team("t_A"), c["slot"]) in ("CB", "FB") for c in out["changes"])


# --- offline jobs -----------------------------------------------------------

def test_offline_recommendation_reads_requirements_from_the_goal(ctx, teams):
    out = agent.recommend_changes(ctx, "t_A", "better in the air at the back, under 30")
    assert out["source"] == "offline" and "age ≤ 29" in out["requirements"]
    for c in out["changes"]:
        assert ctx.lookup[c["player_id"]].age <= 29
    if out["changes"]:
        assert out["measured"]["verdict"] in ("better", "worse", "no clear difference")


def test_match_analysis_and_team_review_are_saved_and_valid(ctx, teams):
    log = pipeline.simulate(ctx, "t_A", "t_B")
    a = pipeline.generate_match_analysis(ctx, log.match_id)
    assert pipeline.load_match_analysis(log.match_id)["headline"] == a["headline"]
    assert all(f["team_id"] in (log.team_a, log.team_b) for f in a["key_factors"])
    assert len(a["players"]) == 4                                   # best and worst of each side
    for r in a["recommendations"]:
        assert r["team_id"] in storage.list_teams()
        if r.get("player_id"):
            assert r["measured"]["verdict"] in ("better", "worse", "no clear difference")
    review = pipeline.generate_team_review(ctx, "t_A")
    assert review["matches_seen"] == 1 and pipeline.load_team_review("t_A")["summary"] == review["summary"]
    assert {p["player_id"] for p in review["players"]} <= set(log.lineups["t_A"].values())


def test_scouting_is_about_the_player_unless_a_team_is_chosen(ctx, teams):
    pid = _bench(ctx)[0].player_id
    alone = agent.scout_player(ctx, pid)
    assert alone["fit_for_teams"] == [] and alone["team_id"] is None      # no team assumed
    assert pid not in {c["player_id"] for c in alone["compared_with"]}
    for_a = agent.scout_player(ctx, pid, team_id="t_A")
    assert [f["team_id"] for f in for_a["fit_for_teams"]] == ["t_A"]     # only the team asked about
    for f in for_a["fit_for_teams"]:
        assert f["verdict"] == agent.fit_verdict(f["measured"])           # the verdict comes from the test
    pipeline.generate_scouting(ctx, pid)
    pipeline.generate_scouting(ctx, pid, team_id="t_A")
    assert pipeline.load_scouting(pid)["team_id"] is None and pipeline.load_scouting(pid, "t_A")["team_id"] == "t_A"


def test_a_review_can_be_limited_to_the_last_or_random_games(ctx, teams, monkeypatch):
    for _ in range(4):
        pipeline.simulate(ctx, "t_A", "t_B")
    everything = pipeline.team_history(ctx, "t_A")
    last2 = pipeline.team_history(ctx, "t_A", limit=2)
    assert everything["selection"] == {"pick": "all", "used": 4, "total": 4}
    assert [m["match_id"] for m in last2["matches"]] == [m["match_id"] for m in everything["matches"][-2:]]
    r1 = pipeline.team_history(ctx, "t_A", limit=2, pick="random", seed=7)
    assert r1["matches"] == pipeline.team_history(ctx, "t_A", limit=2, pick="random", seed=7)["matches"]
    assert r1["selection"] == {"pick": "random", "used": 2, "total": 4}
    # the assistant's tools are held to the selection
    monkeypatch.setenv("OPENROUTER_API_KEY", "test")
    outside = everything["matches"][0]["match_id"]
    model = FakeModel([tool_call("get_match", {"match_id": outside}),
                       final({"summary": "Two games reviewed.", "trends": [], "players": [], "recommendations": []})])
    monkeypatch.setattr(agent, "chat_message", model)
    review = pipeline.generate_team_review(ctx, "t_A", model="stub", limit=2)
    assert "isn't one of the matches selected" in json.loads(model.sent[-1][-1]["content"])["error"]
    assert review["scope"]["used"] == 2 and review["matches_seen"] == 4
    assert json.loads(model.sent[0][1]["content"])["history (from team_history)"]["selection"]["used"] == 2
