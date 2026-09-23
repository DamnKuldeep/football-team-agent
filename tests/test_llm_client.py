"""The LLM's output feeds a calculation only in parse_brief, so that's where
the safety net is tested: the keyword parser, and validation of whatever the
model sends back (no network -- _chat is stubbed)."""
import json

import pytest

from fta import llm_client


@pytest.mark.parametrize("brief, expected, foot", [
    ("more aerial ability, stay left-footed", {"heading_accuracy", "heading_power"}, "left"),
    ("a fast, clinical finisher", {"pace", "acceleration", "finishing", "composure"}, None),
    ("needs a playmaker who reads the game", {"vision", "through_ball", "passing_short", "positioning"}, None),
    ("right footed full-back with work rate", {"stamina"}, "right"),
    ("an impressive, remarkable player", set(), None),        # no false hits inside other words
])
def test_keyword_parser(monkeypatch, brief, expected, foot):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    out = llm_client.parse_brief(brief, position="CB")
    assert set(out["reweight"]) == expected
    assert out["hard_foot"] == foot and out["source"] == "keywords"
    assert all(out["reasons"][a] for a in expected)          # every change says why


def test_llm_reply_is_validated(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "test")
    reply = {"adjustments": [
        {"attribute": "heading_accuracy", "weight": 0.4, "reason": "good in the air"},
        {"attribute": "strength", "weight": 0.3, "reason": "fast"},                  # not an attribute
        {"attribute": "gk_reflexes", "weight": 0.3, "reason": "fast"},               # GK-only on a CB
        {"attribute": "pace", "weight": 0.9, "reason": "fast"},                      # out of range -> clamped
        {"attribute": "marking", "weight": 0.35, "reason": "a solid defender"},      # not in the request
        {"attribute": "tackling_standing", "weight": 0.2, "reason": "fast"},         # restated, unchanged
    ], "hard_foot": "both"}                                                          # invalid foot
    monkeypatch.setattr(llm_client, "_chat", lambda *a, **k: (json.dumps(reply), 10, 10))
    out = llm_client.parse_brief("fast and good in the air", position="CB",
                                 current_weights={"tackling_standing": 0.2, "heading_accuracy": 0.15})
    assert out["reweight"] == {"heading_accuracy": 0.4, "pace": llm_client.WEIGHT_MAX}
    assert out["hard_foot"] is None
    assert any("isn't in your request" in i for i in out["ignored"])               # grounding check
    assert len(out["ignored"]) == 5                                                 # every rejection reported
    assert out["source"].startswith("llm:")


def test_unparseable_reply_gets_one_corrective_retry(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "test")
    replies = iter(["Sure! Here you go: {oops", json.dumps({"adjustments": [
        {"attribute": "pace", "weight": 0.35, "reason": "quick"}], "filters": [], "hard_foot": None})])
    seen = []

    def fake_chat(model, messages, **kwargs):
        seen.append(messages)
        return next(replies), 5, 5
    monkeypatch.setattr(llm_client, "_chat", fake_chat)
    out = llm_client.parse_brief("someone quick", position="WING")
    assert out["reweight"] == {"pace": 0.35} and len(seen) == 2
    assert "can't be used" in seen[1][-1]["content"]                                # told what was wrong


def test_model_gets_the_team_context_and_references_resolve(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "test")
    context = {"slot": "CB_L", "current_player": {"slot": "CB_L", "name": "A", "age": 30, "nationality": "IT"},
               "lineup": [{"slot": "CB_R", "name": "Bruno Silva", "age": 26, "foot": "right", "nationality": "BR"}],
               "pool": {"nationalities": ["BR", "IT"], "age_range": [18, 37]}}
    sent = {}

    def fake_chat(model, messages, **kwargs):
        sent.update(json.loads(messages[1]["content"]))
        return json.dumps({"adjustments": [], "filters": [
            {"field": "nationality", "value": "br", "reason": "same nationality as our CB_R"},
            {"field": "nationality", "value": "Atlantis", "reason": "same nationality as our CB_R"},
            {"field": "age", "max": 29, "reason": "younger than him"}], "hard_foot": None}), 5, 5
    monkeypatch.setattr(llm_client, "_chat", fake_chat)
    out = llm_client.parse_brief("same nationality as our CB_R, younger than him", position="CB", context=context)
    assert sent["lineup"] and sent["current_player"] and sent["pool"]["nationalities"]
    assert [(f["field"], f["value"], f["max"]) for f in out["filters"]] == [("nationality", "BR", None),
                                                                            ("age", None, 29)]
    assert any("Atlantis" in i for i in out["ignored"])                           # must exist in the pool


def test_offline_parser_resolves_simple_references(monkeypatch):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    context = {"current_player": {"slot": "ST", "name": "Z", "age": 31, "nationality": "FR"},
               "lineup": [{"slot": "CB_R", "name": "Bruno Silva", "age": 26, "foot": "right", "nationality": "BR"}]}
    out = llm_client.parse_brief("younger than him, same nationality as CB_R", position="ST", context=context)
    assert {(f["field"], f["value"], f["max"]) for f in out["filters"]} == {("age", None, 30),
                                                                            ("nationality", "BR", None)}


def test_offline_parser_resolves_groups_of_teammates(monkeypatch):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    lineup = [{"slot": "GK", "name": "K", "age": 30, "nationality": "NO"},
              {"slot": "CB_L", "name": "L", "age": 27, "nationality": "IT"},
              {"slot": "CB_R", "name": "R", "age": 26, "nationality": "SN"}]
    context = {"slot": "GK", "lineup": lineup}

    def nat(brief, ctx=context):
        f = next(f for f in llm_client.parse_brief(brief, position="GK", context=ctx)["filters"]
                 if f["field"] == "nationality")
        return f["values"] or [f["value"]]

    assert nat("better reflexes and same nationality as at least one centre back") == ["IT", "SN"]  # any of
    assert nat("same nationality as our right centre-back") == ["SN"]
    assert nat("same nationality as the other centre-back", {"slot": "CB_L", "lineup": lineup}) == ["SN"]


def test_limits_and_foot_come_from_the_words(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "test")
    reply = {"adjustments": [], "hard_foot": None, "filters": [
        {"field": "age", "max": 30, "reason": "under 30"},                                   # off by one
        {"field": "stamina", "min": 82, "reason": "the stamina to last 90 minutes"},          # 82 isn't in the words
        {"field": "pace", "min": 80, "reason": "at least 80 pace"}]}
    monkeypatch.setattr(llm_client, "_chat", lambda *a, **k: (json.dumps(reply), 5, 5))
    out = llm_client.parse_brief("a left-footed winger under 30 with the stamina to last 90 minutes, "
                                 "at least 80 pace", position="WING")
    assert {(f["field"], f["min"], f["max"]) for f in out["filters"]} == {("age", None, 29.0), ("pace", 80.0, None)}
    assert any("corrected" in i for i in out["ignored"]) and any("stamina" in i for i in out["ignored"])
    assert out["hard_foot"] == "left"                                 # named in the request, missed by the reply


def test_foot_words_are_read_carefully():
    foot = llm_client._keyword_foot
    assert foot("an explosive left-footed winger") == "left" and foot("a right-footer") == "right"
    assert foot("right-footed, not left-footed") == "right"
    assert foot("not a left-footer") is None and foot("left or right footed") is None


def test_unusable_llm_reply_falls_back_to_keywords(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "test")
    monkeypatch.setattr(llm_client, "_chat", lambda *a, **k: ("sorry, I can't do that", 5, 5))
    out = llm_client.parse_brief("more aerial ability", position="CB")
    assert out["source"] == "keywords" and "heading_accuracy" in out["reweight"]


def test_keeper_requests_cannot_touch_outfield_attributes(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "test")
    reply = {"adjustments": [{"attribute": "heading_accuracy", "weight": 0.3, "reason": "aerial"},
                             {"attribute": "gk_handling", "weight": 0.35, "reason": "\"safe hands\""}]}
    monkeypatch.setattr(llm_client, "_chat", lambda *a, **k: (json.dumps(reply), 1, 1))
    out = llm_client.parse_brief("safe hands, commanding in the air", position="GK")
    assert out["reweight"] == {"gk_handling": 0.35}
    assert any("heading_accuracy" in item for item in out["ignored"])


@pytest.mark.parametrize("brief, expected", [
    ("a quick winger under 23", [("age", None, 22, None)]),
    ("experienced centre-back", [("age", 29, None, None)]),
    ("u21 striker", [("age", None, 20, None)]),
    ("aged 24-28, two-footed", [("age", 24, 28, None), ("weak_foot_rating", 4, None, None)]),
    ("a natural left back with at least 80 pace", [("natural_position", None, None, "FB"), ("pace", 80, None, None)]),
])
def test_keyword_filters(monkeypatch, brief, expected):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    got = [(f["field"], f["min"], f["max"], f["value"]) for f in llm_client.parse_brief(brief, position="WING")["filters"]]
    assert got == expected


def test_llm_filters_are_validated(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "test")
    reply = {"adjustments": [], "filters": [
        {"field": "age", "max": 24, "reason": "\"under 25\""},
        {"field": "age", "min": 7},                                   # out of range
        {"field": "natural_position", "value": "left back"},         # not a position code
        {"field": "height", "min": 185},                             # unsupported field
        {"field": "nationality", "value": "Brazil", "reason": "\"Brazilian\""},
    ]}
    monkeypatch.setattr(llm_client, "_chat", lambda *a, **k: (json.dumps(reply), 1, 1))
    out = llm_client.parse_brief("Brazilian, under 25", position="CB")
    assert [(f["field"], f["max"], f["value"]) for f in out["filters"]] == [("age", 24, None),
                                                                            ("nationality", None, "Brazil")]
    assert len(out["ignored"]) == 3
