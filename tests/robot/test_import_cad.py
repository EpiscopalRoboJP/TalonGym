from __future__ import annotations

from pathlib import Path

import pytest

from talongym.assets.import_robot_cad import (
    RobotCadError,
    delete_robot_assets,
    guess_inch_scale,
    import_robot_cad,
    resolve_robot_asset,
    write_ascii_box_stl,
)
from talongym.presets.loader import load_preset, validate_document


def test_guess_inch_scale():
    scale_m, units_m = guess_inch_scale(0.45)
    assert units_m == "m"
    assert scale_m == pytest.approx(39.37007874015748)
    scale_mm, units_mm = guess_inch_scale(457.0)
    assert units_mm == "mm"
    assert scale_mm == pytest.approx(1.0 / 25.4)
    assert guess_inch_scale(18.0)[1] == "in"


def test_robot_asset_rejects_traversal():
    with pytest.raises(RobotCadError):
        resolve_robot_asset("../pyproject.toml")
    with pytest.raises(RobotCadError):
        resolve_robot_asset("/etc/passwd")


def test_schema_accepts_visual_mesh_fields():
    robot = dict(load_preset("robot", "mecanum_meepmeep_defaults"))
    robot["visualAsset"] = "robots/team_bot/visual.glb"
    robot["collisionAsset"] = "robots/team_bot/collision.stl"
    robot["visualOffset"] = {"x": 0, "y": 0, "z": 0, "yawDeg": 90, "scale": 1}
    chassis = dict(robot["chassis"])
    chassis["collisionShape"] = "mesh"
    chassis["footprint"] = [{"x": 9, "y": 7}, {"x": 9, "y": -7}, {"x": -9, "y": -7}, {"x": -9, "y": 7}]
    robot["chassis"] = chassis
    assert validate_document("robot", robot) == []


def test_import_stl_box_inches(tmp_path: Path):
    trimesh = pytest.importorskip("trimesh")
    stl = write_ascii_box_stl(tmp_path / "box.stl", 18, 14, 10)
    robot_id = "test_cube_cad"
    try:
        result = import_robot_cad(stl, robot_id)
        assert result["unitsGuess"] == "in"
        assert result["bbox"]["lengthIn"] == pytest.approx(18.0, rel=0.05)
        assert result["bbox"]["widthIn"] == pytest.approx(14.0, rel=0.05)
        assert result["bbox"]["heightIn"] == pytest.approx(10.0, rel=0.05)
        assert len(result["footprint"]) >= 3
        glb = resolve_robot_asset(result["visualAsset"])
        stl_out = resolve_robot_asset(result["collisionAsset"])
        assert glb.is_file()
        assert glb.read_bytes()[:4] == b"glTF"
        assert stl_out.is_file()
        loaded = trimesh.load(str(glb), force="mesh")
        assert loaded is not None
    finally:
        delete_robot_assets(robot_id)
