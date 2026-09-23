"""One end-to-end smoke test: the CLI's full-demo command should run all
nine steps without raising, using the offline (no-API-key) LLM fallback.
Unit tests elsewhere cover each module's logic in isolation; this just
proves the wiring between them holds up.
"""
import pytest
from click.testing import CliRunner

from fta import storage
from fta.cli import cli


@pytest.fixture(autouse=True)
def isolate_state(tmp_path, monkeypatch):
    monkeypatch.setattr(storage, "TEAMS_DIR", tmp_path / "teams")
    monkeypatch.setattr(storage, "INDEX_PATH", tmp_path / "teams" / "index.json")
    monkeypatch.setattr("fta.pipeline.MATCHES_DIR", tmp_path / "matches")
    monkeypatch.setattr("fta.pipeline.SCOUTING_DIR", tmp_path / "scouting")
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)  # force offline LLM fallback
    monkeypatch.setenv("PLAYER_SOURCE", "synthetic")  # small, deterministic pool
    yield


def test_full_demo_runs_without_error():
    runner = CliRunner()
    result = runner.invoke(cli, ["full-demo"], catch_exceptions=False)
    assert result.exit_code == 0
    assert "full-demo complete" in result.output
    assert "HUMAN CHECKPOINT" in result.output  # checkpoints are visibly surfaced, not skipped silently


def test_build_team_then_show_team_round_trips():
    runner = CliRunner()
    r1 = runner.invoke(cli, ["build-team", "--team-id", "t_X", "--name", "X FC"], catch_exceptions=False)
    assert r1.exit_code == 0
    r2 = runner.invoke(cli, ["show-team", "--team-id", "t_X"], catch_exceptions=False)
    assert r2.exit_code == 0
    assert "X FC" in r2.output
