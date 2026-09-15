import json

import pytest

from talongym.assets.cad_layout import (
    format_preset_json,
    lab_bounds_to_placement,
    sync_field_to_cad_layout,
)
from talongym.paths import ASSETS_DIR, PRESETS_DIR


def _part(part, x, y, z=0.0, size=(2.8, 2.8, 2.8), n=0):
    return {
        "part": part,
        "instance": f"{part}_{n}" if n else part,
        "x": x,
        "y": y,
        "z": z,
        "sizeIn": list(size),
        "inField": abs(x) < 72 and abs(y) < 72,
    }


def _field():
    return {
        "elements": [
            {"id": "flower_a", "type": "flower", "pose": {"x": -30, "y": -60, "headingDeg": 0}, "shape": {"kind": "aabb", "width": 5, "depth": 5}, "cadPart": "Flower", "isCollider": True},
            {"id": "flower_b", "type": "flower", "pose": {"x": 30, "y": 60, "headingDeg": 0}, "shape": {"kind": "aabb", "width": 5, "depth": 5}, "cadPart": "Flower", "isCollider": True},
            {"id": "loading", "type": "tape", "pose": {"x": 0, "y": 0, "headingDeg": 0}, "shape": {"kind": "aabb", "width": 11, "depth": 23}, "cadPart": ["Tape 11", "Tape 20"]},
            {"id": "park", "type": "zone", "pose": {"x": 5, "y": 5, "headingDeg": 0}, "shape": {"kind": "aabb", "width": 20, "depth": 32}, "cadPart": ["Tape 11", "Tape 20"]},
        ],
        "gamePieces": [{"typeId": "ball", "shape": {"kind": "circle", "radius": 1.4}, "massKg": 0.05, "cadPart": "Ball"}],
        "spawns": [
            {"id": "old_floor", "pieceTypeId": "ball", "poses": [{"x": 0, "y": 0, "headingDeg": 0}]},
            {"id": "preload", "pieceTypeId": "ball", "preloadEligible": True, "poses": [{"x": 1, "y": 1, "headingDeg": 0}]},
        ],
        "startSlots": [{"id": "red_0", "alliance": "red", "slot": 0, "pose": {"x": -63, "y": -24, "headingDeg": 0}}],
    }


def _layout():
    return {
        "parts": [
            _part("Flower", -68.0, -23.4, 20.0, (5, 6, 1)),
            _part("Flower", 68.0, 23.4, 20.0, (5, 6, 1), n=1),
            # Loading zone outline: two short strips joined by a long one.
            _part("Tape 11", -64.6, 24.4, 0.0, (11, 1, 0.01)),
            _part("Tape 11", -64.6, 46.1, 0.0, (11, 1, 0.01), n=1),
            _part("Tape 20", -59.6, 35.25, 0.0, (1, 20.69, 0.01)),
            _part("Ball", -68.0, -23.4, 1.39),
            _part("Ball", -68.0, -23.4, 4.28, n=1),
            _part("Ball", -40.0, 10.0, 1.39, n=2),
            # Outside the perimeter: alliance-area stock, never a spawn.
            _part("Ball", -73.05, 30.0, 0.81, n=3),
        ]
    }


def test_sync_snaps_elements_and_rebuilds_cad_spawns():
    synced, report = sync_field_to_cad_layout(_field(), _layout())
    els = {el["id"]: el for el in synced["elements"]}
    assert (els["flower_a"]["pose"]["x"], els["flower_a"]["pose"]["y"]) == (-68.0, -23.4)
    assert (els["flower_b"]["pose"]["x"], els["flower_b"]["pose"]["y"]) == (68.0, 23.4)
    # Tape group center, shared by the loading zone and the park drawn by the same tape.
    assert (els["loading"]["pose"]["x"], els["loading"]["pose"]["y"]) == (-64.6, 35.25)
    assert (els["park"]["pose"]["x"], els["park"]["pose"]["y"]) == (-64.6, 35.25)

    spawns = {sp["id"]: sp for sp in synced["spawns"]}
    assert "old_floor" not in spawns and "preload" in spawns
    stack = spawns["flower_a_ball"]["poses"]
    assert [p["z"] for p in stack] == [1.39, 4.28]
    assert spawns["ball_cad"]["poses"] == [{"x": -40.0, "y": 10.0, "headingDeg": 0, "z": 1.39}]
    assert all(abs(p["x"]) < 72 for sp in spawns.values() for p in sp["poses"])
    assert report["missingCadParts"] == []


def test_sync_slides_start_slot_off_a_snapped_collider():
    synced, report = sync_field_to_cad_layout(_field(), _layout())
    pose = synced["startSlots"][0]["pose"]
    flower = next(el for el in synced["elements"] if el["id"] == "flower_a")["pose"]
    assert report["startSlots"]["red_0"]["to"] == (pose["x"], pose["y"])
    assert abs(pose["x"] - flower["x"]) >= 9 + 2.5 or abs(pose["y"] - flower["y"]) >= 9 + 2.5


def test_placement_converts_lab_bounds_to_ftc_frame():
    # Lab is Y-up with z = -ftc.y; tile top sits 1.7 in above the CAD floor.
    p = lab_bounds_to_placement("am-5851: Pollen_12", [-69.4, 1.7, 22.0], [-66.6, 4.5, 24.8], 1.7)
    assert p["part"] == "am-5851: Pollen"
    assert (p["x"], p["y"], p["z"]) == pytest.approx((-68.0, -23.4, 1.4))
    assert p["sizeIn"] == pytest.approx([2.8, 2.8, 2.8])
    assert p["inField"] is True


def test_preset_formatter_round_trips_biobuzz_field():
    path = PRESETS_DIR / "seasons" / "biobuzz_2026" / "field.json"
    src = path.read_text(encoding="utf-8")
    assert format_preset_json(json.loads(src)) + "\n" == src


def test_biobuzz_preset_matches_committed_cad_layout():
    field = json.loads((PRESETS_DIR / "seasons" / "biobuzz_2026" / "field.json").read_text(encoding="utf-8"))
    layout = json.loads((ASSETS_DIR / "seasons" / "biobuzz_2026" / "field_layout.json").read_text(encoding="utf-8"))
    synced, report = sync_field_to_cad_layout(field, layout)
    assert report["missingCadParts"] == []
    assert synced == field, "field.json drifted from field_layout.json; rerun import-field-cad --step"
