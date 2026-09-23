from pathlib import Path

from fta.kaggle_loader import anonymize, load_and_convert
from fta.models import ATTRIBUTE_NAMES, GK_ONLY

FIXTURE = Path(__file__).parent / "fixtures" / "sample_kaggle.csv"
# Same column headers as the real Kaggle FIFA 22 players_22.csv (fictional rows)
FIFA22_FIXTURE = Path(__file__).parent / "fixtures" / "sample_kaggle_fifa22.csv"


def test_real_fifa22_headers_fill_every_attribute():
    players, warnings = load_and_convert(FIFA22_FIXTURE, anonymize_names=False)
    assert len(players) == 3 and not warnings
    for p in players:
        expected = ATTRIBUTE_NAMES if p.is_gk() else [a for a in ATTRIBUTE_NAMES if a not in GK_ONLY]
        missing = [a for a in expected if p.attributes[a] is None]
        assert not missing, f"{p.player_id} missing {missing}"


def test_derived_attributes_and_specific_columns_win():
    players, _ = load_and_convert(FIFA22_FIXTURE, anonymize_names=False)
    st = next(p for p in players if p.player_id == "k_910001")
    assert st.attributes["pace"] == 86.0            # movement_sprint_speed beats the 'pace' summary
    assert st.attributes["dribbling"] == 88.0       # skill_dribbling beats the 'dribbling' summary
    assert st.attributes["positioning"] == 93.0     # max(mentality_positioning, interceptions)
    assert st.attributes["through_ball"] == 79.0    # mean(vision 80, short passing 78)
    assert st.attributes["heading_power"] == 77.5   # mean(jumping 75, strength 80)
    gk = next(p for p in players if p.is_gk())
    assert gk.attributes["gk_reflexes"] == 88.0     # reflexes, not diving


def test_nationality_kept_in_full_so_prefixes_dont_collide():
    players, _ = load_and_convert(FIFA22_FIXTURE, anonymize_names=False)
    nats = {p.player_id: p.nationality for p in players}
    assert nats["k_910001"] == "Austria" and nats["k_910002"] == "Australia"


def test_anonymize_scales_past_small_name_pool_with_unique_names():
    players = [{"name": f"real {i}", "tags": ["kaggle_derived"]} for i in range(20_000)]
    names = [p["name"] for p in anonymize(players)]
    assert len(set(names)) == len(names)


def test_loads_expected_row_count():
    players, _warnings = load_and_convert(FIXTURE, anonymize_names=False)
    assert len(players) == 5


def test_positions_parsed_and_mapped():
    players, _ = load_and_convert(FIXTURE, anonymize_names=False)
    gk = next(p for p in players if p.is_gk())
    assert gk.natural_positions == ["GK"]
    cb = next(p for p in players if "CB" in p.natural_positions)
    assert cb.player_id == "k_900001"


def test_gk_only_attrs_null_for_outfield_players():
    players, _ = load_and_convert(FIXTURE, anonymize_names=False)
    outfield = [p for p in players if not p.is_gk()]
    assert all(p.attributes["gk_reflexes"] is None for p in outfield)


def test_gk_has_gk_attrs_populated():
    players, _ = load_and_convert(FIXTURE, anonymize_names=False)
    gk = next(p for p in players if p.is_gk())
    assert gk.attributes["gk_reflexes"] is not None


def test_anonymize_replaces_names_and_tags():
    players, _ = load_and_convert(FIXTURE, anonymize_names=True)
    names = [p.name for p in players]
    assert "K. Sample One" not in names  # original raw name gone
    assert all("anonymized" in p.tags for p in players)


def test_anonymize_preserves_stats():
    with_names, _ = load_and_convert(FIXTURE, anonymize_names=False)
    anon, _ = load_and_convert(FIXTURE, anonymize_names=True)
    by_id_named = {p.player_id: p for p in with_names}
    by_id_anon = {p.player_id: p for p in anon}
    for pid, named_player in by_id_named.items():
        assert named_player.attributes == by_id_anon[pid].attributes
        assert named_player.name != by_id_anon[pid].name


def test_missing_source_column_warns_not_crashes():
    players, warnings = load_and_convert(FIXTURE, anonymize_names=False)
    assert len(players) == 5  # didn't crash
    assert any("through_ball" in w for w in warnings)


def test_no_duplicate_player_ids():
    players, _ = load_and_convert(FIXTURE, anonymize_names=False)
    ids = [p.player_id for p in players]
    assert len(ids) == len(set(ids))
