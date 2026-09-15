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
    red_starts = [s for s in field["startSlots"] if s["alliance"] == "red"]
    assert {s["id"] for s in red_starts} == {"red_0", "red_1"}
    park = next(el for el in field["elements"] if el["id"] == "red_park")
    park_y = float(park["pose"]["y"])
    park_hy = float(park["shape"]["depth"]) / 2.0
    for slot in red_starts:
        assert float(slot["pose"]["x"]) < 0
        assert abs(float(slot["pose"]["y"]) - park_y) > park_hy
    assert field.get("backgroundAsset")
    assert field.get("collisionAsset")
    assert "mesh_field_collision" in field["requiredCapabilities"]
    preloads = [s for s in field["spawns"] if s.get("heldByRobotId")]
    assert {s["heldByRobotId"] for s in preloads} == {"red_0", "red_1", "blue_0", "blue_1"}
