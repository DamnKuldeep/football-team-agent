import pytest

from fta import storage
from fta.models import FormationSlot, Team


@pytest.fixture(autouse=True)
def isolate_teams_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(storage, "TEAMS_DIR", tmp_path / "teams")
    monkeypatch.setattr(storage, "INDEX_PATH", tmp_path / "teams" / "index.json")
    yield


def make_team(team_id="t1", version=1):
    return Team(
        team_id=team_id, name="Test FC", formation="4-3-3", version=version,
        slots=[FormationSlot(slot_id="GK", position="GK", player_id="p1")],
    )


def test_save_and_load_round_trip():
    team = make_team()
    storage.save_version(team)
    loaded = storage.load_team("t1")
    assert loaded.name == "Test FC"
    assert loaded.version == 1


def test_cannot_overwrite_existing_version():
    storage.save_version(make_team(version=1))
    with pytest.raises(FileExistsError):
        storage.save_version(make_team(version=1))


def test_next_version_bumps_and_preserves_parent():
    team_v1 = make_team()
    storage.save_version(team_v1)
    team_v2 = storage.next_version(team_v1, diff={"reason": "test swap"})
    assert team_v2.version == 2
    assert team_v2.parent_version == 1
    storage.save_version(team_v2)

    latest = storage.load_team("t1")
    assert latest.version == 2
    v1_again = storage.load_team("t1", version=1)
    assert v1_again.version == 1  # rollback still accessible


def test_unknown_team_raises_keyerror():
    with pytest.raises(KeyError):
        storage.load_team("does_not_exist")


def test_next_version_does_not_mutate_original():
    team_v1 = make_team()
    storage.next_version(team_v1, diff={})
    assert team_v1.version == 1  # unchanged
