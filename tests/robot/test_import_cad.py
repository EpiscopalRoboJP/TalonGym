from __future__ import annotations

from pathlib import Path

import pytest

from talongym.assets.import_robot_cad import (
    RobotCadError,
    cad_collision_fit,
    collision_size_yup,
    collision_span_in,
    delete_robot_assets,
    extract_cad_archive,
    guess_inch_scale,
    import_robot_cad,
    import_robot_part_cad,
    resolve_catalog_inch_scale,
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


def test_import_part_preserves_origin(tmp_path: Path):
    trimesh = pytest.importorskip("trimesh")
    stl = write_ascii_box_stl(tmp_path / "box.stl", 18, 14, 10)
    robot_id = "test_part_origin"
    try:
        whole = import_robot_cad(stl, robot_id)
        part = import_robot_part_cad(stl, robot_id, "intake", parent_id="chassis")
        assert part["originPreserved"] is True
        assert whole["originPreserved"] is False
        assert part["partId"] == "intake"
        assert part["parentId"] == "chassis"
        assert part["visualAsset"].endswith("parts/intake/visual.glb")
        preserved = trimesh.load(str(resolve_robot_asset(part["visualAsset"])), force="mesh")
        centered = trimesh.load(str(resolve_robot_asset(whole["visualAsset"])), force="mesh")
        assert preserved.bounds[0][1] == pytest.approx(0.0, abs=0.2)
        assert centered.bounds[0][1] == pytest.approx(-centered.extents[1] / 2.0, abs=0.2)
    finally:
        delete_robot_assets(robot_id)


def test_import_zip_of_stl(tmp_path: Path):
    pytest.importorskip("trimesh")
    import zipfile

    stl = write_ascii_box_stl(tmp_path / "box.stl", 18, 14, 10)
    zipped = tmp_path / "box.zip"
    with zipfile.ZipFile(zipped, "w") as zf:
        zf.write(stl, "box.stl")
    inner = extract_cad_archive(zipped, tmp_path / "out")
    assert inner.name == "box.stl"
    robot_id = "test_zip_cad"
    try:
        result = import_robot_cad(zipped, robot_id)
        assert result["unitsGuess"] == "in"
        assert resolve_robot_asset(result["visualAsset"]).is_file()
    finally:
        delete_robot_assets(robot_id)


def test_step_without_cascadio_is_cad_extra(tmp_path: Path, monkeypatch):
    pytest.importorskip("trimesh")
    import builtins

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "cascadio":
            raise ImportError("cascadio missing")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    step = tmp_path / "part.step"
    step.write_text("ISO-10303-21;\nHEADER;\nENDSEC;\nDATA;\nENDSEC;\nEND-ISO-10303-21;\n", encoding="ascii")
    with pytest.raises(RobotCadError, match="cascadio"):
        import_robot_cad(step, "test_step_extra")
    pytest.importorskip("trimesh")
    stl = write_ascii_box_stl(tmp_path / "box.stl", 8, 4, 2)
    with pytest.raises(RobotCadError, match="sha256"):
        import_robot_cad(stl, "test_sha", expected_sha256="ab" * 32)


def test_catalog_transform_keeps_origin(tmp_path: Path):
    pytest.importorskip("trimesh")
    from talongym.assets.import_catalog_cad import convert_catalog_part, invalidate_catalog_cache
    from talongym.robot.catalog import get_part

    stl = write_ascii_box_stl(tmp_path / "proxy.stl", 48, 16, 8)
    sku = "1207-0001-0001"
    part = get_part(sku)
    try:
        result = convert_catalog_part(sku, source=stl, force=True)
        assert result["reused"] is False
        assert result["cache"]["state"] == "ready"
        assert result["import"]["visualAsset"].startswith("robot_parts/gobilda/")
        assert result["import"]["unitsGuess"] in {"mm", "declared"}
        bbox = result["import"]["bbox"]
        assert bbox["lengthIn"] == pytest.approx(1.8898, rel=0.05)
        assert bbox["widthIn"] == pytest.approx(0.6299, rel=0.05)
        assert bbox["heightIn"] == pytest.approx(0.315, rel=0.08)
        reused = convert_catalog_part(sku)
        assert reused["reused"] is True
        assert part["visualTransform"]["rotatedZupToYup"] is True
    finally:
        invalidate_catalog_cache("gobilda", sku)


def test_resolve_catalog_inch_scale_metres_vs_mm():
    target = 1.8898
    scale_m, units_m = resolve_catalog_inch_scale(0.048, declared=1.0 / 25.4, target_in=target)
    assert units_m == "m"
    assert scale_m == pytest.approx(39.37007874015748)
    assert 0.048 * scale_m == pytest.approx(target, rel=0.02)
    scale_mm, units_mm = resolve_catalog_inch_scale(48.0, declared=1.0 / 25.4, target_in=target)
    assert units_mm in {"mm", "declared"}
    assert 48.0 * scale_mm == pytest.approx(target, rel=0.02)
    assert collision_span_in([{"kind": "box", "sizeIn": [1.8898, 0.47, 1.2]}]) == pytest.approx(1.8898)
    assert collision_span_in([{"kind": "cylinder", "radiusIn": 0.5, "lengthIn": 2.0}]) == pytest.approx(2.0)


def test_catalog_metres_mesh_matches_collision(tmp_path: Path):
    pytest.importorskip("trimesh")
    stl = write_ascii_box_stl(tmp_path / "metres.stl", 0.048, 0.016, 0.008)
    robot_id = "test_catalog_metres"
    try:
        result = import_robot_cad(
            stl,
            robot_id,
            preserve_origin=True,
            visual_transform={
                "scaleToInches": 1.0 / 25.4,
                "rotatedZupToYup": True,
                "translationIn": [0.0, 0.0, 0.0],
            },
            collision=[{"kind": "box", "sizeIn": [1.8898, 1.8898, 0.4724]}],
        )
        assert result["unitsGuess"] == "m"
        assert result["bbox"]["lengthIn"] == pytest.approx(1.8898, rel=0.05)
        assert max(result["bbox"].values()) == pytest.approx(1.8898, rel=0.05)
        assert max(result["bbox"].values()) > 0.5
    finally:
        delete_robot_assets(robot_id)


def test_import_bakes_scene_graph_transforms(tmp_path: Path):
    trimesh = pytest.importorskip("trimesh")
    import numpy as np

    box = trimesh.creation.box(extents=[2.0, 2.0, 2.0])
    scene = trimesh.Scene()
    transform = np.eye(4)
    transform[0, 3] = 10.0
    scene.add_geometry(box, transform=transform)
    glb = tmp_path / "offset.glb"
    scene.export(str(glb), file_type="glb")
    robot_id = "test_scene_bake"
    try:
        result = import_robot_cad(
            glb,
            robot_id,
            preserve_origin=True,
            visual_transform={"scaleToInches": 1.0, "rotatedZupToYup": False},
            target_span_in=2.0,
        )
        loaded = trimesh.load(str(resolve_robot_asset(result["visualAsset"])), force="mesh")
        center = loaded.bounds.mean(axis=0)
        assert center[0] == pytest.approx(10.0, abs=0.25)
        assert float(max(loaded.extents)) == pytest.approx(2.0, abs=0.2)
    finally:
        delete_robot_assets(robot_id)


def test_cad_collision_fit_swings_end_origin_bar_onto_x():
    fit = cad_collision_fit([0.00059, 0.01654, 0.00059], [0.0, 0.00827, 0.0], [16.5354, 0.5906, 0.5906])
    assert fit["scale"] == pytest.approx(1000.0)
    rotated_y = [
        fit["rotation"][0][1],
        fit["rotation"][1][1],
        fit["rotation"][2][1],
    ]
    assert rotated_y[0] == pytest.approx(1.0, abs=0.05)
    assert abs(rotated_y[1]) < 0.05
    assert fit["translation"][0] == pytest.approx(-8.27, abs=0.2)
    assert abs(fit["translation"][1]) < 0.2
    assert collision_size_yup([{"kind": "box", "sizeIn": [16.5354, 0.5906, 0.5906]}]) == pytest.approx([16.5354, 0.5906, 0.5906])


def test_catalog_extrusion_aligns_long_axis_to_collision(tmp_path: Path):
    trimesh = pytest.importorskip("trimesh")
    stl = write_ascii_box_stl(tmp_path / "rail.stl", 0.015, 0.015, 0.420)
    robot_id = "test_catalog_rail_axis"
    try:
        result = import_robot_cad(
            stl,
            robot_id,
            preserve_origin=True,
            visual_transform={
                "scaleToInches": 1.0 / 25.4,
                "rotatedZupToYup": True,
                "translationIn": [0.0, 0.0, 0.0],
            },
            collision=[{"kind": "box", "sizeIn": [16.5354, 0.5906, 0.5906]}],
        )
        loaded = trimesh.load(str(resolve_robot_asset(result["visualAsset"])), force="mesh")
        extents = [float(v) for v in loaded.extents]
        center = loaded.bounds.mean(axis=0)
        assert extents[0] == pytest.approx(16.5354, rel=0.08)
        assert max(extents[1], extents[2]) < 1.5
        assert abs(float(center[0])) < 1.0
        assert abs(float(center[1])) < 0.5
    finally:
        delete_robot_assets(robot_id)
