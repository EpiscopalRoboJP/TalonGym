from talongym.presets.loader import load_preset, list_presets


def test_only_biobuzz_field_ships():
    ids = {row["id"] for row in list_presets("field")}
    assert ids == {"biobuzz_2026_field_v1"}


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
    red = next(el for el in field["elements"] if el["id"] == "red_garden")
    blue = next(el for el in field["elements"] if el["id"] == "blue_garden")
    assert red["pose"]["y"] < -60
    assert blue["pose"]["y"] > 60
    assert red["pose"]["x"] < 0
    assert blue["pose"]["x"] > 0
    assert field.get("backgroundAsset")
    assert field.get("collisionAsset")
    assert "mesh_field_collision" in field["requiredCapabilities"]
