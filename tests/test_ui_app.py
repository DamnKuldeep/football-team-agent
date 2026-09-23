"""Drives the Streamlit UI headlessly through the whole arc (build ->
swap -> simulate -> report -> feedback -> assistant), clicking the same buttons a person
would. Skipped when the optional `ui` extra isn't installed."""
from pathlib import Path

import pytest

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest

from fta import pipeline, storage

APP = str(Path(pipeline.__file__).with_name("ui_app.py"))


@pytest.fixture(autouse=True)
def isolate_state(tmp_path, monkeypatch):
    monkeypatch.setattr(storage, "TEAMS_DIR", tmp_path / "teams")
    monkeypatch.setattr(storage, "INDEX_PATH", tmp_path / "teams" / "index.json")
    monkeypatch.setattr(pipeline, "MATCHES_DIR", tmp_path / "matches")
    monkeypatch.setattr(pipeline, "SCOUTING_DIR", tmp_path / "scouting")
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.setenv("PLAYER_SOURCE", "synthetic")  # small, deterministic pool
    monkeypatch.setattr("fta.config.load_dotenv", lambda *a, **k: None)  # never pick up a real key
    monkeypatch.setattr("fta.agent.EVAL_MATCHES", 20)                    # quick measured tests


def _click(at: AppTest, label: str) -> AppTest:
    next(b for b in at.button if b.label == label).click()
    at.run()
    assert not at.exception, [e.value for e in at.exception]
    return at


def test_deploy_entry_point_reads_streamlit_secrets(monkeypatch):
    """streamlit_app.py (what Streamlit Community Cloud runs) starts the app,
    and settings from the app's Secrets reach the code as environment variables."""
    import os
    monkeypatch.setenv("MODEL", "placeholder")
    monkeypatch.delenv("MODEL")                              # restored to "unset" after the test
    at = AppTest.from_file(str(Path(pipeline.ROOT) / "streamlit_app.py"), default_timeout=60)
    at.secrets["MODEL"] = "google/gemini-3.8-flash"
    at.run()
    assert not at.exception and os.environ["MODEL"] == "google/gemini-3.8-flash"
    assert at.selectbox(key="model").value == "google/gemini-3.8-flash"


def test_ui_full_flow():
    at = AppTest.from_file(APP, default_timeout=60)
    at.run()
    assert not at.exception

    _click(at, "Build team")                                # first team: t_A / Alpha FC
    _click(at, "Build team")                                # "New team": t_B, leaving out t_A's players
    assert set(storage.list_teams()) == {"t_A", "t_B"}
    assert not set(storage.load_team("t_A").player_ids()) & set(storage.load_team("t_B").player_ids())

    # stage two changes, then save them as ONE new version
    at.selectbox(key="squad_team").set_value("t_A").run()
    at.text_input(key="swap_brief").set_value("more aerial ability, stay left-footed")
    for slot in ("CB_L", "ST"):
        at.selectbox(key="swap_slot_t_A").set_value(slot)
        _click(at, "Find players")
        _click(at, f"Stage for {slot}")
    assert storage.list_teams()["t_A"]["latest_version"] == 1  # nothing saved while staging
    at.button(key="save_draft").click().run()
    v2 = storage.load_team("t_A")
    assert v2.version == 2 and {c["slot"] for c in v2.diff["changes"]} == {"CB_L", "ST"}

    _click(at, "Play")                                      # t_A v2 vs t_B, random seed each time
    _click(at, "Play")
    logs = [pipeline.load_event_log(m) for m in pipeline.list_matches()]
    assert len(logs) == 2 and logs[0].seed != logs[1].seed
    assert logs[0].possessions[-1].minute == 90

    at.selectbox(key="m_away_2").set_value("t_A").run()       # t_A v2 vs its own v1
    at.selectbox(key="m_av_t_A_2").set_value(1)
    _click(at, "Play")
    newest = pipeline.load_event_log(pipeline.list_matches()[0])
    assert {newest.team_a, newest.team_b} == {"t_A@v2", "t_A@v1"}

    _click(at, "Find replacement")                          # from a player's feedback card
    newest = pipeline.list_matches()[0]
    at.button(key=f"analyse_{newest}").click().run()        # the assistant's analysis (offline: rules)
    assert not at.exception and pipeline.load_match_analysis(newest)["source"] == "offline"

    at.button(key="review_btn").click().run()               # Squad → Form & history → review
    # 3 matches, but the one against its own v1 counts once per side: 4 games in t_A's history
    assert not at.exception and pipeline.load_team_review("t_A")["matches_seen"] == 4

    # the assistant proposes, the human approves, changes are only staged
    at.text_input(key="assist_goal").set_value("stronger in the air at the back, under 30")
    at.button(key="assist_ask").click().run()
    assert not at.exception
    proposal = at.session_state["assist"]["out"]
    assert proposal["source"] == "offline"
    if proposal["changes"]:
        at.button(key="assist_stage").click().run()
        assert not at.exception
        assert at.session_state["drafts"]["t_A"]["changes"] == {c["slot"]: c["player_id"]
                                                                 for c in proposal["changes"]}
        assert storage.list_teams()["t_A"]["latest_version"] == 2   # nothing saved without "Save"
        at.button(key="discard_draft").click().run()

    at.button(key="confirm_delete_match").click().run()
    assert not at.exception and len(pipeline.list_matches()) == 2
    at.button(key="clear_all").click().run()
    assert not at.exception and storage.list_teams() == {} and pipeline.list_matches() == []
