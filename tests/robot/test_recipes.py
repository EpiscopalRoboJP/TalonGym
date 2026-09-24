"""Guided drivebase recipes instantiate schema 1.2 assemblies that compile."""

from __future__ import annotations

import pytest

from talongym.presets.loader import validate_document
from talongym.robot.catalog import get_part
from talongym.robot.contract import PHYSICAL_SCHEMA_VERSION
from talongym.robot.recipes import (
    RECIPE_IDS,
    RecipeError,
    instantiate_and_compile,
    instantiate_recipe,
    list_recipes,
    load_recipe,
)
from talongym.robot.transmissions import part_tags


def test_four_drivebase_recipes_expose_catalog_parameter_choices():
    recipes = {row["id"]: row for row in list_recipes()}
    assert tuple(recipes) == RECIPE_IDS
    for recipe_id in recipes:
        loaded = load_recipe(recipe_id)
        assert loaded["manufacturer"] in {"gobilda", "rev"}
        assert loaded["drivetrainType"] in {"mecanum", "tank"}
        parameters = loaded["parameters"]
        for name in ("lengthSku", "widthSku", "motorSku", "wheelSku"):
            spec = parameters[name]
            assert spec["default"] in spec["choices"]
            for sku in spec["choices"]:
                part = get_part(sku)
                assert part["manufacturer"] == loaded["manufacturer"]
        if loaded["manufacturer"] == "rev":
            spec = parameters["cartridgeSku"]
            assert spec["default"] in spec["choices"]
            for sku in spec["choices"]:
                part = get_part(sku)
                assert part["manufacturer"] == "rev"
                assert "gearbox" in part_tags(part)
                assert float(part["gearRatio"]) in {3.0, 4.0, 5.0}
        motor = get_part(parameters["motorSku"]["default"])
        assert part_tags(motor).intersection({"motor", "gearbox"})
        wheel = get_part(parameters["wheelSku"]["default"])
        if loaded["drivetrainType"] == "mecanum":
            assert "wheel_mecanum" in part_tags(wheel)
        else:
            assert part_tags(wheel).intersection({"wheel_traction", "wheel_omni"})


@pytest.mark.parametrize("recipe_id", RECIPE_IDS)
def test_default_recipes_instantiate_schema_1_2_and_compile(recipe_id: str):
    document = instantiate_recipe(recipe_id)
    assert validate_document("robot", document) == []
    assert document["schemaVersion"] == "1.2.0"
    assert document["assembly"]["recipe"]["id"] == recipe_id
    assert document["assembly"]["recipe"]["parameters"]["motorSku"]
    compiled = instantiate_and_compile(recipe_id, competitive=True)
    assert compiled.compiled.physical is True
    assert compiled.preset["schemaVersion"] == PHYSICAL_SCHEMA_VERSION
    assert compiled.preset["id"] == recipe_id
    assert compiled.preset.get("launchers") == []
    assert compiled.preset["mechanisms"]["launchCapable"] is False
    instance_ids = {row["id"] for row in document["assembly"]["instances"]}
    compiled_ids = {row["id"] for row in compiled.preset["rigidParts"]}
    assert instance_ids <= compiled_ids
    assert compiled.preset["drivetrain"]["type"] == load_recipe(recipe_id)["drivetrainType"]
    for prefix in ("left_rail", "motor_fl", "wheel_fl"):
        assert prefix in instance_ids


def test_gobilda_recipes_include_rails_motors_wheels_brackets_and_electronics():
    mecanum = instantiate_recipe("gobilda_mecanum")
    ids = {row["id"] for row in mecanum["assembly"]["instances"]}
    skus = {row["id"]: row["sku"] for row in mecanum["assembly"]["instances"]}
    assert {"left_rail", "front_rail", "right_rail", "rear_rail"} <= ids
    assert {f"motor_{corner}" for corner in ("fl", "fr", "rl", "rr")} <= ids
    assert {f"wheel_{corner}" for corner in ("fl", "fr", "rl", "rr")} <= ids
    assert {"plate_center", "hub_center", "bracket_fl", "bracket_fr", "sensor_imu", "servo_front"} <= ids
    assert skus["motor_fl"] == "5203-2402-0019"
    assert skus["wheel_fl"] == "3606-0000-0096"
    assert skus["wheel_fr"] == "3606-0100-0096"
    compiled = instantiate_and_compile("gobilda_mecanum")
    assert compiled.report.drivetrain.get("complete") is True
    assert compiled.preset["motors"]["count"] == 4
    tank = instantiate_and_compile("gobilda_tank")
    assert tank.report.drivetrain.get("type") == "tank"
    assert tank.report.drivetrain.get("complete") is True


def test_gobilda_parameter_choices_and_electronics_toggle_recompile():
    swapped = instantiate_recipe(
        "gobilda_mecanum",
        {
            "lengthSku": "1120-0013-0336",
            "widthSku": "1120-0004-0120",
            "motorSku": "5202-2402-0003",
            "includeElectronics": False,
        },
    )
    skus = {row["id"]: row["sku"] for row in swapped["assembly"]["instances"]}
    ids = set(skus)
    assert skus["left_rail"] == "1120-0013-0336"
    assert skus["front_rail"] == "1120-0004-0120"
    assert skus["motor_rr"] == "5202-2402-0003"
    assert skus["wheel_rl"] == "3606-0000-0096"
    assert "sensor_imu" not in ids
    assert "servo_front" not in ids
    compiled = instantiate_and_compile(
        "gobilda_tank",
        {"lengthSku": "1120-0004-0120", "includeElectronics": False},
    )
    assert compiled.compiled.physical is True
    assert compiled.preset["drivetrain"]["type"] == "tank"


def test_rev_recipes_place_rails_motors_wheels_and_compile():
    document = instantiate_recipe("rev_mecanum")
    ids = {row["id"] for row in document["assembly"]["instances"]}
    skus = {row["id"]: row["sku"] for row in document["assembly"]["instances"]}
    assert {"left_rail", "front_rail", "right_rail", "rear_rail"} <= ids
    assert {f"motor_{corner}" for corner in ("fl", "fr", "rl", "rr")} <= ids
    assert {f"wheel_{corner}" for corner in ("fl", "fr", "rl", "rr")} <= ids
    assert {"up_fl", "bracket_fl", "control_hub"} <= ids
    assert skus["left_rail"] == "REV-41-1432"
    assert skus["motor_fl"] == "REV-41-1600"
    assert skus["cartridge_fl"] == "REV-41-1603"
    assert skus["wheel_fl"] == "REV-41-1656"
    assert not any(row.get("code") == "rev_m3_tap_clearance" for row in document.get("warnings") or [])
    compiled = instantiate_and_compile("rev_mecanum")
    compiled_ids = {row["id"] for row in compiled.preset["rigidParts"]}
    assert {"left_rail", "motor_fl", "wheel_fl", "up_rr"} <= compiled_ids
    assert compiled.preset["drivetrain"]["type"] == "mecanum"
    assert compiled.report.drivetrain.get("complete") is True
    codes = {row["code"] for row in compiled.warnings}
    assert "cad_unavailable" in codes or "cad_missing" in codes
    tank = instantiate_and_compile("rev_tank", {"includeElectronics": False})
    tank_ids = {row["id"] for row in tank.preset["rigidParts"]}
    assert "motor_rr" in tank_ids
    assert "control_hub" not in tank_ids
    assert tank.preset["drivetrain"]["type"] == "tank"


def test_recipes_report_real_track_wheelbase_and_selected_gearing():
    expected = {
        "gobilda_mecanum": {"gear": 19.2, "track": 12.913, "wheelbase": 11.654},
        "gobilda_tank": {"gear": 19.2, "track": 4.409, "wheelbase": 5.984},
        "rev_mecanum": {"gear": 5.0, "track": 8.189, "wheelbase": 14.803},
        "rev_tank": {"gear": 5.0, "track": 8.189, "wheelbase": 14.803},
    }
    for recipe_id, want in expected.items():
        compiled = instantiate_and_compile(recipe_id, competitive=True)
        drive = compiled.preset["drivetrain"]
        report = compiled.report.drivetrain
        assert compiled.preset["motors"]["gearRatio"] == pytest.approx(want["gear"], rel=1e-6)
        assert report.get("gearRatio") == pytest.approx(want["gear"], rel=1e-6)
        assert drive["trackWidthIn"] == pytest.approx(want["track"], abs=0.35)
        assert report.get("trackWidthIn") == pytest.approx(want["track"], abs=0.35)
        assert drive["wheelbaseIn"] == pytest.approx(want["wheelbase"], abs=0.5)
        assert drive["trackWidthIn"] != pytest.approx(3.622, abs=0.05)
        if recipe_id.startswith("rev"):
            paths = compiled.report.transmissions
            assert paths
            assert all("cartridge_" in " ".join(path.instance_ids) for path in paths)
            assert all(path.source == "gearbox_stack" for path in paths)
            assert {row["id"]: row["sku"] for row in instantiate_recipe(recipe_id)["assembly"]["instances"]}[
                "cartridge_fl"
            ] == "REV-41-1603"

    three = instantiate_and_compile("rev_mecanum", {"cartridgeSku": "REV-41-1601"}, competitive=True)
    assert three.preset["motors"]["gearRatio"] == pytest.approx(3.0)
    four = instantiate_and_compile("rev_tank", {"cartridgeSku": "REV-41-1602"}, competitive=True)
    assert four.preset["motors"]["gearRatio"] == pytest.approx(4.0)
    full = instantiate_and_compile("gobilda_mecanum", competitive=True)
    narrow = instantiate_and_compile("gobilda_mecanum", {"widthSku": "1120-0004-0120"}, competitive=True)
    assert narrow.preset["drivetrain"]["trackWidthIn"] == pytest.approx(4.409, abs=0.35)
    assert narrow.preset["drivetrain"]["trackWidthIn"] < full.preset["drivetrain"]["trackWidthIn"] - 2.0


def test_recipe_loader_rejects_unknown_ids_and_invalid_choices():
    with pytest.raises(RecipeError, match="unknown drivebase recipe"):
        load_recipe("andymark_swerve")
    with pytest.raises(RecipeError, match="not a recipe choice"):
        instantiate_recipe("gobilda_mecanum", {"motorSku": "REV-41-1600"})
    with pytest.raises(RecipeError, match="not a recipe choice"):
        instantiate_recipe("gobilda_tank", {"wheelSku": "3213-3606-0001"})
    with pytest.raises(RecipeError, match="unknown recipe parameters"):
        instantiate_recipe("rev_mecanum", {"color": "orange"})


def test_recipes_form_compact_planar_drivebases():
    for recipe_id in RECIPE_IDS:
        compiled = instantiate_and_compile(recipe_id, competitive=True)
        assert compiled.report.mass is not None
        lo = compiled.report.mass.aabb_min_in
        hi = compiled.report.mass.aabb_max_in
        span = [hi[i] - lo[i] for i in range(3)]
        assert max(span[0], span[1]) < 20.0, f"{recipe_id} XY span {span} is a stick or oversize"
        assert min(span[0], span[1]) > 6.0, f"{recipe_id} collapsed to a line {span}"
        ids = {row["id"] for row in compiled.preset["rigidParts"]}
        assert {"left_rail", "front_rail", "right_rail", "rear_rail", "motor_fl", "wheel_fl"} <= ids
        assert span[2] >= 2.5, f"{recipe_id} wheels are not standing up, height {span[2]}"


def test_recipe_drive_wheels_spin_about_lateral_axles():
    from talongym.robot.catalog import get_part
    from talongym.robot.mounts import mount_axis, part_mount
    from talongym.robot.transforms import matrix_from_pose, transform_direction

    for recipe_id in RECIPE_IDS:
        document = instantiate_recipe(recipe_id)
        compiled = instantiate_and_compile(recipe_id, competitive=True)
        instances = {row["id"]: row for row in document["assembly"]["instances"]}
        for ident in ("wheel_fl", "wheel_fr", "wheel_rl", "wheel_rr"):
            pose = matrix_from_pose(compiled.instance_poses[ident])
            part = get_part(instances[ident]["sku"])
            axis = transform_direction(pose, mount_axis(part_mount(part, "bore")))
            assert abs(float(axis[1])) > 0.95, f"{recipe_id} {ident} bore {axis} is not lateral"
            assert abs(float(axis[2])) < 0.2, f"{recipe_id} {ident} bore {axis} points at the floor"


def test_recipe_compile_throughput_is_proportional():
    import time

    t0 = time.perf_counter()
    for recipe_id in RECIPE_IDS:
        instantiate_and_compile(recipe_id, competitive=True)
    elapsed = time.perf_counter() - t0
    rate = len(RECIPE_IDS) / max(elapsed, 1e-6)
    print(f"recipe_compiles_per_s={rate:.1f}")
    assert elapsed < 15
