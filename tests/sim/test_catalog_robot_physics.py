"""Catalog assembly MJCF: welded structure, catalog inertia, wheel contact, snapshot IDs."""

from __future__ import annotations

import copy

import pytest
from tests.robot.test_assembly import golden_three_part_assembly

from talongym.assets.mjcf_robot import emit_robot_bodies, kinematic_part_transforms
from talongym.presets.loader import LoadedPresets, load_bundle, load_preset
from talongym.robot.assembly import compile_assembly_to_preset
from talongym.robot.catalog import get_part
from talongym.sim.world import World


def _compiled_golden():
    return compile_assembly_to_preset(golden_three_part_assembly(), competitive=True)


def _robot_xml(preset: dict, *, robot_hz: float = 7.0, body_y: float = 7.2):
    bodies, _assets, stats = emit_robot_bodies(
        1,
        9.0,
        9.0,
        robot_hz,
        mesh_file=None,
        body_y=body_y,
        robot=preset,
    )
    return "\n".join(bodies), stats


def test_golden_mjcf_welds_fixed_structure_and_keeps_instance_ids():
    compiled = _compiled_golden()
    xml, stats = _robot_xml(compiled.preset)
    for ident in ("channel", "motor", "wheel"):
        assert f'name="red_0_part_{ident}"' in xml
    assert 'name="red_0_joint_wheel_joint"' in xml
    assert 'name="red_0_joint_channel_joint"' not in xml
    assert 'name="red_0_joint_motor_joint"' not in xml
    channel_geom_at = xml.find('name="red_0_part_channel_geom_0"')
    motor_geom_at = xml.find('name="red_0_part_motor_geom_0"')
    channel_body_at = xml.find('name="red_0_part_channel"')
    motor_body_at = xml.find('name="red_0_part_motor"')
    wheel_body_at = xml.find('name="red_0_part_wheel"')
    wheel_geom_at = xml.find('name="red_0_part_wheel_geom_0"')
    assert 0 <= channel_geom_at < channel_body_at
    assert 0 <= motor_geom_at < motor_body_at
    assert 0 <= wheel_body_at < wheel_geom_at
    assert "channel" in stats["weldedPartIds"]
    assert "motor" in stats["weldedPartIds"]
    assert "wheel" in stats["articulatedPartIds"]
    assert 'type="cylinder"' in xml
    assert "1.88980" in xml


def test_golden_mjcf_uses_catalog_mass_inertia_and_floor_wheels():
    compiled = _compiled_golden()
    channel = get_part("1120-0001-0048")
    motor = get_part("5203-2402-0019")
    wheel = get_part("3213-3606-0001")
    xml, stats = _robot_xml(compiled.preset, robot_hz=7.0)
    assert f'mass="{float(channel["massKg"]):.6f}"' in xml
    assert f'mass="{float(motor["massKg"]):.6f}"' in xml
    assert f'mass="{float(wheel["massKg"]):.6f}"' in xml
    assert "diaginertia=" in xml
    assert stats["wheelPartIds"] == ["wheel"]
    assert stats["wheelContactZ"] == pytest.approx(-7.0, abs=0.05)
    assert 'name="red_0_part_intake_roller"' not in xml
    assert 'name="red_0_joint_intake_joint"' not in xml


def test_kinematic_snapshot_rows_keep_exact_catalog_ids():
    compiled = _compiled_golden()
    radius = float(get_part("3213-3606-0001")["collision"][0]["radiusIn"])
    rows = kinematic_part_transforms(
        compiled.preset,
        robot_x=12.0,
        robot_y=-8.0,
        heading=0.0,
        origin_z=7.2,
        robot_hz=7.0,
    )
    ids = {row["id"] for row in rows}
    assert {"channel", "motor", "wheel"} <= ids
    assert "intake_roller" not in ids
    wheel_row = next(row for row in rows if row["id"] == "wheel")
    assert wheel_row["x"] == pytest.approx(12.0, abs=3.0)
    assert wheel_row["z"] == pytest.approx(7.2 - 7.0 + radius, abs=0.2)


def test_four_cap_kinematic_rows_keep_mechanism_hull():
    robot = load_preset("robot", "mecanum_biobuzz_4cap")
    rows = kinematic_part_transforms(
        robot,
        robot_x=0.0,
        robot_y=0.0,
        heading=0.0,
        origin_z=7.2,
        robot_hz=7.0,
    )
    ids = {row["id"] for row in rows}
    assert "intake_roller" in ids
    assert "flywheel" in ids


def test_scoring_starter_kinematic_rows_are_catalog_only():
    compiled = compile_assembly_to_preset(load_preset("robot", "gobilda_mecanum_starter"), competitive=True)
    rows = kinematic_part_transforms(
        compiled.preset,
        robot_x=0.0,
        robot_y=0.0,
        heading=0.0,
        origin_z=7.2,
        robot_hz=7.0,
    )
    ids = {row["id"] for row in rows}
    assert "intake_wheel" in ids
    assert "flywheel_wheel" in ids
    assert "intake_roller" not in ids
    assert "flywheel" not in ids


def test_existing_physical_robot_still_emits_articulated_geoms():
    from talongym.presets.loader import load_preset

    robot = load_preset("robot", "mecanum_biobuzz_4cap")
    xml, stats = _robot_xml(robot)
    assert 'name="red_0_part_intake_roller"' in xml
    assert 'name="red_0_part_intake_roller_geom_0"' in xml
    assert xml.find('name="red_0_part_intake_roller"') < xml.find('name="red_0_part_intake_roller_geom_0"')
    assert stats["weldedPartIds"] == []
    assert "intake_roller" in stats["articulatedPartIds"]


@pytest.mark.require_cad
def test_golden_field_mjcf_keeps_free_pieces_and_catalog_ids():
    from talongym.assets.mjcf_field import build_field_mjcf

    compiled = _compiled_golden()
    field = load_bundle("biobuzz_2026_field_v1", "mecanum_biobuzz_4cap", "biobuzz_2026_scoring_v1").field
    mjcf = build_field_mjcf(field, robot=compiled.preset, n_robots=1)
    assert 'inertiafromgeom="auto"' in mjcf.xml
    assert "<freejoint" in mjcf.xml
    assert "gp_pollen_00" in mjcf.xml
    assert 'name="red_0_part_channel"' in mjcf.xml
    assert 'name="red_0_part_wheel"' in mjcf.xml
    assert 'name="red_0_joint_wheel_joint"' in mjcf.xml


@pytest.mark.require_mesh
def test_golden_world_snapshot_preserves_ids_voltage_and_free_pieces():
    compiled = _compiled_golden()
    bundle = load_bundle("biobuzz_2026_field_v1", "mecanum_biobuzz_4cap", "biobuzz_2026_scoring_v1")
    world = World(
        LoadedPresets(field=bundle.field, robot=copy.deepcopy(compiled.preset), scoring=bundle.scoring, training=bundle.training),
        seed=0,
    )
    world.reset(seed=0, static_teammate=False, full_noise=False)
    snap = world.snapshot()
    actor = next(robot for robot in snap["robots"] if robot["id"] == "red_0")
    part_ids = {row["id"] for row in actor["parts"]}
    assert {"channel", "motor", "wheel"} <= part_ids
    assert snap["physicalPieces"] is True
    assert actor.get("batteryVoltageV") is None or actor.get("batteryVoltageV", 0) >= 0
    assert actor.get("batteryCurrentA") is not None or actor.get("batteryVoltageV") is None
    assert "flywheel" not in (actor.get("actuators") or {})
    held = [piece for piece in snap["pieces"] if piece.get("heldBy") == "red_0"]
    assert held == []
    parked = [piece for piece in snap["pieces"] if not piece.get("heldBy") and piece.get("typeId")]
    assert parked
    assert world.backend.name == "mujoco_field"


@pytest.mark.require_mesh
@pytest.mark.parametrize("recipe_id", ["gobilda_mecanum", "gobilda_tank", "rev_mecanum", "rev_tank"])
def test_recipe_world_snapshot_preserves_catalog_ids(recipe_id: str):
    from talongym.robot.recipes import instantiate_and_compile

    compiled = instantiate_and_compile(recipe_id, competitive=True)
    bundle = load_bundle("biobuzz_2026_field_v1", "mecanum_biobuzz_4cap", "biobuzz_2026_scoring_v1")
    world = World(
        LoadedPresets(field=bundle.field, robot=copy.deepcopy(compiled.preset), scoring=bundle.scoring, training=bundle.training),
        seed=0,
    )
    world.reset(seed=0, static_teammate=False, full_noise=False)
    snap = world.snapshot()
    actor = next(robot for robot in snap["robots"] if robot["id"] == "red_0")
    part_ids = {row["id"] for row in actor["parts"]}
    assert {"left_rail", "motor_fl", "wheel_fl"} <= part_ids
    design_ids = {row["id"] for row in (snap.get("robotDesign") or {}).get("rigidParts") or []}
    assert {"left_rail", "motor_fl", "wheel_fl"} <= design_ids


@pytest.mark.parametrize("recipe_id", ["gobilda_mecanum", "gobilda_tank", "rev_mecanum", "rev_tank"])
def test_recipe_mjcf_and_replay_transforms_keep_instance_ids(recipe_id: str):
    from talongym.robot.recipes import instantiate_and_compile

    compiled = instantiate_and_compile(recipe_id, competitive=True)
    xml, stats = _robot_xml(compiled.preset)
    for ident in ("left_rail", "motor_fl", "wheel_fl"):
        assert f'name="red_0_part_{ident}"' in xml
    assert "left_rail" in stats["weldedPartIds"] or "left_rail" in stats["articulatedPartIds"]
    assert "wheel_fl" in stats["articulatedPartIds"] or "wheel_fl" in stats["wheelPartIds"]
    rows = kinematic_part_transforms(
        compiled.preset,
        robot_x=4.0,
        robot_y=2.0,
        heading=0.0,
        origin_z=7.2,
        robot_hz=7.0,
    )
    ids = {row["id"] for row in rows}
    assert {"left_rail", "motor_fl", "wheel_fl"} <= ids
    assert compiled.preset.get("visualAsset") in (None, compiled.preset.get("visualAsset"))
    catalog_ids = {row["id"] for row in compiled.preset["rigidParts"] if row["id"] not in {"chassis"}}
    assert len(catalog_ids) > 4
    for ident in ("left_rail", "front_rail", "right_rail", "rear_rail", "motor_fl", "wheel_fl"):
        assert ident in catalog_ids
    tagged = [row for row in compiled.preset["rigidParts"] if row["id"] == "left_rail"]
    assert tagged and ("channel" in (tagged[0].get("tags") or []) or "extrusion" in (tagged[0].get("tags") or []))
