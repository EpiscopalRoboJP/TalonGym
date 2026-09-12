from talongym.presets.loader import load_preset


def test_into_the_deep_auto_volumes():
    field = load_preset("field", "into_the_deep_2024_field")
    ids = {el["id"] for el in field["elements"]}
    assert "red_high_basket" in ids
    assert "red_low_chamber" in ids
    assert "red_ascent_l1" in ids
    types = {p["typeId"] for p in field["gamePieces"]}
    assert "sample_yellow" in types
    assert "clipped_sample" in types
    scoring = load_preset("scoring", "into_the_deep_2024_scoring")
    nodes = {n["id"] for n in scoring["nodes"]}
    assert "high_chamber" in nodes
    assert "park_observation" in nodes


def test_biobuzz_auto_volumes():
    field = load_preset("field", "biobuzz_2026_field_v1")
    ids = {el["id"] for el in field["elements"]}
    assert "red_cell_up" in ids
    assert "red_park" in ids
    assert "leave_interior" in ids
    assert "red_garden" in ids
    flowers = [el["id"] for el in field["elements"] if el["type"] == "flower"]
    assert len(flowers) == 4
    types = {p["typeId"] for p in field["gamePieces"]}
    assert "pollen" in types
    assert "nectar_red" in types
    scoring = load_preset("scoring", "biobuzz_2026_scoring_v1")
    nodes = {n["id"] for n in scoring["nodes"]}
    assert "hive_tip" in nodes
    assert "leave_at_auto_end" in nodes
    assert "park_at_auto_end" in nodes


def test_centerstage_three_spikes():
    field = load_preset("field", "centerstage_2023_field")
    spikes = [el["id"] for el in field["elements"] if el["type"] == "spike_mark"]
    assert spikes == ["red_spike_left", "red_spike_center", "red_spike_right"] or set(spikes) == {
        "red_spike_left",
        "red_spike_center",
        "red_spike_right",
    }
    scoring = load_preset("scoring", "centerstage_2023_scoring")
    assert any(n["id"].startswith("purple_") for n in scoring["nodes"])
    assert any(n["id"] == "park_backstage" for n in scoring["nodes"])
