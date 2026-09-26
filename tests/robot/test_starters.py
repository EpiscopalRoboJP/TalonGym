"""Complete scoring starters compile from catalog assemblies."""

from __future__ import annotations

import copy

import numpy as np
import pytest

from talongym.presets.loader import load_preset, preset_index, validate_document
from talongym.robot.assembly import compile_assembly_to_preset
from talongym.robot.starters import STARTER_IDS, build_starter_document, starter_skus


@pytest.mark.parametrize("starter_id", STARTER_IDS)
def test_scoring_wheels_align_with_motor_outputs(starter_id: str):
    from talongym.robot.catalog import get_part
    from talongym.robot.mounts import hole_in_part, mount_axis, part_mount
    from talongym.robot.transforms import matrix_from_pose, transform_direction, transform_point

    document = load_preset("robot", starter_id)
    compiled = compile_assembly_to_preset(document, competitive=True)
    instances = {row["id"]: row for row in document["assembly"]["instances"]}
    poses = {ident: matrix_from_pose(pose) for ident, pose in compiled.instance_poses.items()}
    for motor_id, wheel_id in (
        ("intake_motor", "intake_wheel"),
        ("conveyor_motor", "conveyor_wheel"),
        ("launcher_motor", "flywheel_wheel"),
    ):
        motor_mount = part_mount(get_part(instances[motor_id]["sku"]), "output")
        wheel_mount = part_mount(get_part(instances[wheel_id]["sku"]), "bore")
        motor_axis = transform_direction(poses[motor_id], mount_axis(motor_mount))
        wheel_axis = transform_direction(poses[wheel_id], mount_axis(wheel_mount))
        motor_hole = transform_point(poses[motor_id], hole_in_part(motor_mount, None))
        wheel_hole = transform_point(poses[wheel_id], hole_in_part(wheel_mount, None))
        assert float(np.dot(motor_axis, wheel_axis)) == pytest.approx(-1.0, abs=1e-8)
        assert float(np.linalg.norm(motor_hole - wheel_hole)) < 1e-8


@pytest.mark.parametrize("starter_id", STARTER_IDS)
def test_starters_are_schema_1_2_scoring_assemblies(starter_id: str):
    document = load_preset("robot", starter_id)
    assert document == build_starter_document(starter_id)
    assert validate_document("robot", document) == []
    assert document["schemaVersion"] == "1.2.0"
    assert document["functionalBindings"]["confirmed"] is True
    assert document["functionalBindings"]["includeScoringTopology"] is True
    ids = {row["id"] for row in document["assembly"]["instances"]}
    assert {"intake_wheel", "flywheel_wheel", "hood_plate", "left_rail", "motor_fl"} <= ids
    roles = document["functionalBindings"]["instanceRoles"]
    assert roles["intake_wheel"] == "intake"
    assert roles["flywheel_wheel"] == "flywheel"
    assert roles["conveyor_wheel"] == "intake"
    compiled = compile_assembly_to_preset(document, competitive=True)
    assert compiled.preset["launchers"]
    assert compiled.preset["intakes"]
    assert compiled.preset["mechanisms"]["capacity"] >= 4
    assert compiled.preset["mechanisms"]["launchCapable"] is not False
    assert float((compiled.preset.get("chassis") or {}).get("heightIn") or 0) >= 14.0
    part_ids = {row["id"] for row in compiled.preset["rigidParts"]}
    assert {"intake_wheel", "flywheel_wheel", "left_rail"} <= part_ids
    assert "intake_roller" not in part_ids
    assert "flywheel" not in part_ids
    chassis = next(row for row in compiled.preset["rigidParts"] if row["id"] == "chassis")
    hull_boxes = [list(item.get("sizeIn") or []) for item in chassis.get("collision") or []]
    assert [18, 18, 1] not in hull_boxes
    actuator_joints = {row["id"]: row.get("jointId") for row in compiled.preset["actuators"]}
    assert actuator_joints.get("intake") in {row["id"] for row in compiled.preset["joints"]}
    assert actuator_joints.get("flywheel") in {row["id"] for row in compiled.preset["joints"]}
    gate_joint = next(row for row in compiled.preset["joints"] if row["id"] == actuator_joints["gate"])
    assert gate_joint["childPartId"] == "gate_servo"
    assert {"launcher_bracket", "launcher_riser", "gate_servo"} <= ids
    for ident in ("flywheel_wheel", "hood_plate", "gate_servo"):
        assert compiled.instance_poses[ident]["z"] > 10.0
    assert not any(row["code"] == "scoring_geometry_unverified" for row in compiled.warnings)
    assert any(row["code"] == "launcher_trajectory_unverified" for row in compiled.warnings)
    from talongym.assets.mjcf_robot import kinematic_part_transforms

    robot_hz = compiled.preset["chassis"]["heightIn"] / 2.0
    flywheel_transform = next(
        row for row in kinematic_part_transforms(
            compiled.preset, robot_x=0.0, robot_y=0.0, heading=0.0,
            origin_z=robot_hz + 0.2, robot_hz=robot_hz,
        ) if row["id"] == "flywheel_wheel"
    )
    assert compiled.preset["piecePath"]["flywheelPose"] == pytest.approx(
        {axis: flywheel_transform[axis] for axis in ("x", "y", "z")}
    )
    assert compiled.compiled.physical is True
    from talongym.robot.catalog import get_part
    from talongym.robot.collision import validate_assembly_collisions
    from talongym.robot.contract import compile_robot_preset
    from talongym.robot.transforms import matrix_from_pose

    instances = {row["id"]: row for row in document["assembly"]["instances"]}
    catalog_parts = {ident: get_part(str(row["sku"])) for ident, row in instances.items()}
    poses = {ident: matrix_from_pose(pose) for ident, pose in compiled.instance_poses.items()}
    validate_assembly_collisions(poses, instances, catalog_parts, document["assembly"]["connections"])
    stamp = compile_robot_preset(compiled.preset, competitive=True)
    assert stamp.physical is True
    assert stamp.compatibility_stamp
    assert starter_id in preset_index(refresh=True)["robot"]
    assert starter_skus(starter_id)


@pytest.mark.require_mesh
@pytest.mark.parametrize("starter_id", STARTER_IDS)
def test_starter_world_spawns_legal_with_preloads_and_actuators(starter_id: str):
    from talongym.presets.loader import LoadedPresets, load_bundle
    from talongym.sim.world import World

    compiled = compile_assembly_to_preset(load_preset("robot", starter_id), competitive=True)
    bundle = load_bundle("biobuzz_2026_field_v1", "mecanum_biobuzz_4cap", "biobuzz_2026_scoring_v1")
    world = World(
        LoadedPresets(field=bundle.field, robot=copy.deepcopy(compiled.preset), scoring=bundle.scoring, training=bundle.training),
        seed=0,
    )
    world.reset(seed=0, static_teammate=False, full_noise=False)
    snap = world.snapshot()
    actor = next(robot for robot in snap["robots"] if robot["id"] == "red_0")
    part_ids = {row["id"] for row in actor["parts"]}
    assert {"left_rail", "intake_wheel", "flywheel_wheel", "wheel_fl"} <= part_ids
    assert "intake_roller" not in part_ids
    assert "flywheel" not in part_ids
    assert "flywheel" in (actor.get("actuators") or {})
    held = [piece for piece in snap["pieces"] if piece.get("heldBy") == "red_0"]
    assert len(held) == 4
    xml, stats = None, None
    from tests.sim.test_catalog_robot_physics import _robot_xml

    xml, stats = _robot_xml(compiled.preset)
    assert 'name="red_0_part_intake_wheel"' in xml
    assert "wheel_fl" in stats["wheelPartIds"] or "wheel_fl" in stats["articulatedPartIds"]
    assert stats["wheelContactZ"] < 0
    parts = actor["parts"]
    assert all("x" in row and "y" in row and "z" in row for row in parts)


@pytest.mark.require_mesh
@pytest.mark.parametrize("starter_id", STARTER_IDS)
def test_starter_scripted_launches_from_launch_pose(starter_id: str):
    from talongym.presets.loader import load_bundle
    from talongym.training.diagnostics import assert_scripted_baseline_scores

    bundle = load_bundle("biobuzz_2026_field_v1", starter_id, "biobuzz_2026_scoring_v1")
    health = assert_scripted_baseline_scores(bundle, seed=1)
    assert health.launches >= 3
    assert health.true_score >= 20 or health.scored_pieces >= 1
    if starter_id.startswith("gobilda"):
        assert health.wall_contact_s <= 8.0


def test_four_cap_reference_keeps_mechanism_hull_in_kinematics():
    from talongym.assets.mjcf_robot import kinematic_part_transforms

    robot = load_preset("robot", "mecanum_biobuzz_4cap")
    part_ids = {row["id"] for row in robot["rigidParts"]}
    assert "intake_roller" in part_ids
    assert "flywheel" in part_ids
    chassis = next(row for row in robot["rigidParts"] if row["id"] == "chassis")
    hull_boxes = [list(item.get("sizeIn") or []) for item in chassis.get("collision") or []]
    assert [18, 18, 1] in hull_boxes
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


def test_scoring_starter_kinematics_omit_topology_hull():
    from talongym.assets.mjcf_robot import kinematic_part_transforms

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
