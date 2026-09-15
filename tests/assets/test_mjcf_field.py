"""CAD MJCF assembly: convex parts, typed free-joint pieces, no AABB field."""

from __future__ import annotations

import json

import pytest

from talongym.assets.mjcf_field import (
    CAD_MJCF_MARKER,
    classify_collision_part,
    piece_body_name,
    piece_slot_plan,
    select_field_collision_parts,
)
from talongym.paths import ASSETS_DIR
from talongym.presets.loader import load_preset


def _manifest():
    field = load_preset("field", "biobuzz_2026_field_v1")
    path = ASSETS_DIR / field["cadManifest"]
    if not path.is_file():
        pytest.skip("derived CAD assets not generated")
    return json.loads(path.read_text(encoding="utf-8")), field


def test_classifier_keeps_hive_and_glass_drops_fasteners_and_pieces():
    assert classify_collision_part({"id": "am-5876_A-Frame_Leg", "minIn": [0, 0, 0], "maxIn": [1, 1, 1]}) == "playable"
    assert classify_collision_part({"id": "FTC_Field_Side_Glass_11in", "minIn": [70, 1, -20], "maxIn": [71, 12, 20]}) == "playable"
    assert classify_collision_part({"id": "Corner_Hinge_Rivet_Field", "minIn": [-72, 1, -72], "maxIn": [-70, 2, -70]}) == "playable"
    assert classify_collision_part({"id": "am-5851_Pollen_4", "minIn": [0, 0, 0], "maxIn": [3, 3, 3]}) == "cad_scoring_element"
    assert classify_collision_part({"id": "am-5706_Artifact_Tray", "minIn": [71, 0, -8], "maxIn": [82, 4, 8]}) == "alliance_area"
    assert classify_collision_part({"id": "Socket_head_cap_screw_10-32_x_1.5", "minIn": [0, 0, 0], "maxIn": [1, 1, 1]}) == "fastener"
    assert classify_collision_part({"id": "am-5892_Flower_Peanut_Support", "minIn": [20, 2, 69], "maxIn": [22, 6, 71]}) == "playable"
    assert classify_collision_part(
        {"id": "Gaffer_Tape_Blue_1in_Wide_94.82in_Long", "minIn": [124, 1, -47], "maxIn": [126, 1.2, 47]}
    ) in {"floor_marking", "off_field"}
    assert classify_collision_part(
        {
            "id": "am-2499-Center_FIRST_Tech_Challenge_Field_Soft_Tiles",
            "minIn": [0, 1, 0],
            "maxIn": [24, 1.7, 24],
        }
    ) == "floor_tile"


def test_select_parts_never_merges_and_keeps_playable_structure():
    manifest, _field = _manifest()
    parts = manifest["field"]["collisionParts"]
    kept, stats = select_field_collision_parts(parts)
    assert stats.manifest_parts == 715
    assert stats.kept >= 80
    assert stats.kept < 400
    assert stats.dropped.get("cad_scoring_element", 0) >= 50
    assert stats.dropped.get("fastener", 0) >= 100
    assert stats.floor_y == pytest.approx(0.0, abs=0.05)
    ids = {p["id"] for p in kept}
    assert any("A-Frame_Leg" in i for i in ids)
    assert not any("Hive_Goal" in i or "Goal_Rib" in i for i in ids)
    assert any("Glass" in i for i in ids)
    assert any("Flower" in i for i in ids)
    assert not any("Pollen" in i or "Nectar" in i for i in ids)
    assert all(p.get("convex") is True for p in kept)


def test_piece_slot_plan_is_typed():
    field = load_preset("field", "biobuzz_2026_field_v1")
    plan = piece_slot_plan(field)
    assert plan["pollen"] >= 40
    assert plan["nectar_red"] >= 8
    assert plan["nectar_blue"] >= 8
    assert piece_body_name("nectar_red", 3) == "gp_nectar_red_03"


def test_cad_mjcf_uses_meshes_not_aabb_hive():
    from talongym.assets.mjcf_field import build_field_mjcf

    manifest, field = _manifest()
    built = build_field_mjcf(field, n_robots=1, robot_hz=7.0)
    xml = built.xml
    assert CAD_MJCF_MARKER in xml
    assert 'type="mesh"' in xml
    assert "gp_pollen_00" in xml
    assert 'type="free"' in xml or "<freejoint" in xml
    assert "hive_frame_west" not in xml
    assert "trig_red_cell_up" in xml or 'name="trig_red_cell_up"' in xml
    assert xml.lower().count("field.glb") == 0
    assert built.cad is True
    assert built.stats["filter"]["kept"] >= 80
    assert built.floor_y == pytest.approx(0.0, abs=0.05)
    assert "pollen_hull.stl" in xml
    assert "nectar_red_hull.stl" in xml
    assert 'joint name="field_mech_red_hive"' in xml
    assert 'joint name="field_mech_blue_hive"' in xml
    assert xml.count('name="perimeter_glass_') == 4
    assert 'size="9.000 7.000 9.000"' in xml
    assert built.stats["fieldMechanismTargets"]["red_hive"] == pytest.approx(1.0472, abs=0.001)
    pollen_mass = next(p for p in field["gamePieces"] if p["typeId"] == "pollen")["massKg"]
    nectar_mass = next(p for p in field["gamePieces"] if p["typeId"] == "nectar_red")["massKg"]
    assert f'mass="{pollen_mass:.5f}"' in xml
    assert f'mass="{nectar_mass:.5f}"' in xml
    assert pollen_mass != nectar_mass
    from talongym.assets.mjcf_field import ftc_point_hits_parts, select_field_collision_parts

    kept, _stats = select_field_collision_parts(manifest["field"]["collisionParts"])
    cell = next(el for el in field["elements"] if el["id"] == "red_cell_up")
    pose = cell["pose"]
    hits = ftc_point_hits_parts(
        kept,
        float(pose["x"]),
        float(pose["y"]),
        float(pose["z"]),
        margin=0.5,
    )
    assert isinstance(hits, list)


@pytest.mark.require_cad
def test_committed_collision_asset_is_cad_assembled():
    from talongym.assets.mjcf_field import CAD_MJCF_MARKER

    field = load_preset("field", "biobuzz_2026_field_v1")
    xml_path = ASSETS_DIR / field["collisionAsset"]
    assert xml_path.is_file()
    text = xml_path.read_text(encoding="utf-8")
    assert CAD_MJCF_MARKER in text
    assert "<freejoint" in text
    assert "hive_frame_west" not in text
    assert "field.glb" not in text.lower()
