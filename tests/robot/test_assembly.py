"""Schema 1.2 assemblies: mounts, transforms, graph, collision, and 1.1 compile."""

from __future__ import annotations

import copy

import numpy as np
import pytest

from talongym.presets.loader import load_preset, validate_document
from talongym.robot.assembly import clear_materialize_cache, compile_assembly_to_preset, materialize_sim_robot
from talongym.robot.catalog import get_part
from talongym.robot.collision import connected_overlap_warnings, validate_assembly_collisions
from talongym.robot.contract import (
    ASSEMBLY_SCHEMA_VERSION,
    PHYSICAL_SCHEMA_VERSION,
    AssemblyError,
    RobotContractError,
    compile_robot_preset,
)
from talongym.robot.graph import validate_assembly_graph
from talongym.robot.mounts import hole_in_part, mounts_compatible, part_mount, validate_occupancy
from talongym.robot.transforms import matrix_from_pose, solve_mount_transform, transform_point


def _box_part(sku: str, manufacturer: str, mount_id: str, standard: str, kind: str, size: float = 1.0) -> dict:
    return {
        "sku": sku,
        "manufacturer": manufacturer,
        "massKg": 0.1,
        "tags": ["plate"],
        "collision": [{"kind": "box", "sizeIn": [size, size, size]}],
        "mounts": [
            {
                "id": mount_id,
                "kind": kind,
                "standard": standard,
                "diameterMm": 4.0,
                "axis": [0.0, 0.0, 1.0],
                "transform": {"x": 0, "y": 0, "z": 0},
                "pattern": {"type": "grid", "pitchMm": 8.0, "countU": 3, "countV": 1},
            }
        ],
    }


def golden_three_part_assembly() -> dict:
    return {
        "schemaVersion": ASSEMBLY_SCHEMA_VERSION,
        "id": "golden_three_part",
        "displayName": "Golden three-part catalog assembly",
        "assembly": {
            "rootInstanceId": "channel",
            "instances": [
                {"id": "channel", "sku": "1120-0001-0048"},
                {"id": "motor", "sku": "5203-2402-0019"},
                {"id": "wheel", "sku": "3213-3606-0001"},
            ],
            "connections": [
                {
                    "id": "motor_to_channel",
                    "parent": {"instanceId": "channel", "mountId": "web", "patternIndex": [2, 1]},
                    "child": {"instanceId": "motor", "mountId": "face", "patternIndex": [0, 0]},
                    "secondary": {
                        "parent": {"instanceId": "channel", "mountId": "web", "patternIndex": [4, 1]},
                        "child": {"instanceId": "motor", "mountId": "face", "patternIndex": [1, 0]},
                    },
                },
                {
                    "id": "wheel_to_motor",
                    "parent": {"instanceId": "motor", "mountId": "output"},
                    "child": {"instanceId": "wheel", "mountId": "bore"},
                },
            ],
        },
    }


def test_schema_1_2_requires_assembly_and_keeps_earlier_versions_readable():
    assert validate_document("robot", load_preset("robot", "mecanum_meepmeep_defaults")) == []
    assert validate_document("robot", load_preset("robot", "mecanum_biobuzz_4cap")) == []
    document = golden_three_part_assembly()
    assert validate_document("robot", document) == []
    bare = {"schemaVersion": ASSEMBLY_SCHEMA_VERSION, "id": "no_assembly", "displayName": "Missing assembly"}
    assert any("assembly" in message for message in validate_document("robot", bare))


def test_compile_robot_preset_rejects_uncompiled_1_2_assembly():
    with pytest.raises(RobotContractError, match="compile_assembly_to_preset"):
        compile_robot_preset(golden_three_part_assembly(), competitive=False)


def test_mount_compatibility_accepts_m3_tap_into_clearance():
    extrusion = get_part("REV-41-1432")
    bracket = get_part("REV-41-1480")
    up = get_part("REV-41-1621")
    motor = get_part("REV-41-1600")
    mounts_compatible(extrusion, part_mount(extrusion, "slot"), bracket, part_mount(bracket, "face_a"))
    mounts_compatible(up, part_mount(up, "motor_face"), motor, part_mount(motor, "face"))
    with pytest.raises(AssemblyError, match="incompatible mount diameters"):
        tight = dict(part_mount(motor, "face"))
        tight["diameterMm"] = 3.0
        loose = dict(part_mount(bracket, "face_a"))
        loose["diameterMm"] = 8.0
        mounts_compatible(motor, tight, bracket, loose)
    channel = get_part("1120-0001-0048")
    motor = get_part("5203-2402-0019")
    wheel = get_part("3213-3606-0001")
    rev_extrusion = get_part("REV-41-1432")
    mounts_compatible(channel, part_mount(channel, "web"), motor, part_mount(motor, "face"))
    mounts_compatible(motor, part_mount(motor, "output"), wheel, part_mount(wheel, "bore"))
    with pytest.raises(AssemblyError, match="incompatible mount kinds"):
        mounts_compatible(channel, part_mount(channel, "web"), motor, part_mount(motor, "output"))
    with pytest.raises(AssemblyError, match="mismatched mount standards"):
        mounts_compatible(channel, part_mount(channel, "web"), rev_extrusion, part_mount(rev_extrusion, "ends"))
    gobilda_plate = _box_part("g-plate", "gobilda", "face", "gobilda_pattern", "clearance_hole")
    rev_clone = _box_part("r-plate", "rev", "face", "gobilda_pattern", "threaded_hole")
    with pytest.raises(AssemblyError, match="cross-brand connection requires a catalog adapter"):
        mounts_compatible(gobilda_plate, gobilda_plate["mounts"][0], rev_clone, rev_clone["mounts"][0])


def test_snap_transform_coincides_primary_and_secondary_holes():
    channel = get_part("1120-0001-0048")
    motor = get_part("5203-2402-0019")
    parent_mount = part_mount(channel, "web")
    child_mount = part_mount(motor, "face")
    parent_world = np.eye(4)
    child_world = solve_mount_transform(
        parent_world,
        parent_mount,
        child_mount,
        parent_index=(2, 1),
        child_index=(0, 0),
        secondary={
            "parent_mount": parent_mount,
            "child_mount": child_mount,
            "parent_index": (4, 1),
            "child_index": (1, 0),
        },
    )
    parent_primary = transform_point(parent_world, hole_in_part(parent_mount, (2, 1)))
    child_primary = transform_point(child_world, hole_in_part(child_mount, (0, 0)))
    parent_secondary = transform_point(parent_world, hole_in_part(parent_mount, (4, 1)))
    child_secondary = transform_point(child_world, hole_in_part(child_mount, (1, 0)))
    assert np.allclose(parent_primary, child_primary, atol=1e-6)
    assert np.allclose(parent_secondary, child_secondary, atol=1e-5)


def test_graph_rejects_cycles_and_disconnected_instances():
    assembly = golden_three_part_assembly()["assembly"]
    validate_assembly_graph(assembly)
    cyclic = copy.deepcopy(assembly)
    cyclic["connections"].append(
        {
            "id": "cycle",
            "parent": {"instanceId": "wheel", "mountId": "bore"},
            "child": {"instanceId": "channel", "mountId": "web"},
        }
    )
    with pytest.raises(AssemblyError, match="root instance channel cannot be a connection child"):
        validate_assembly_graph(cyclic)
    disconnected = copy.deepcopy(assembly)
    disconnected["instances"].append({"id": "loose", "sku": "1207-0001-0001"})
    with pytest.raises(AssemblyError, match="disconnected assembly instances"):
        validate_assembly_graph(disconnected)


def test_duplicate_occupancy_and_unconnected_collision_are_hard_errors():
    connections = golden_three_part_assembly()["assembly"]["connections"]
    validate_occupancy(connections)
    duplicate = copy.deepcopy(connections)
    duplicate.append(
        {
            "id": "reuse_hole",
            "parent": {"instanceId": "channel", "mountId": "web", "patternIndex": [2, 1]},
            "child": {"instanceId": "motor", "mountId": "face", "patternIndex": [0, 0]},
        }
    )
    with pytest.raises(AssemblyError, match="duplicate occupancy"):
        validate_occupancy(duplicate)

    left = _box_part("left", "gobilda", "face", "gobilda_pattern", "clearance_hole")
    right = _box_part("right", "gobilda", "face", "gobilda_pattern", "threaded_hole")
    pose = np.eye(4)
    with pytest.raises(AssemblyError, match="interpenetration"):
        validate_assembly_collisions(
            {"a": pose, "b": pose},
            {"a": {"id": "a"}, "b": {"id": "b"}},
            {"a": left, "b": right},
            [],
        )
    validate_assembly_collisions(
        {"a": pose, "b": pose},
        {"a": {"id": "a"}, "b": {"id": "b"}},
        {"a": left, "b": right},
        [
            {
                "parent": {"instanceId": "a", "mountId": "face"},
                "child": {"instanceId": "b", "mountId": "face"},
            }
        ],
    )
    with pytest.raises(AssemblyError, match="coincident collision volumes"):
        validate_assembly_collisions(
            {"a": pose, "b": pose, "c": pose},
            {"a": {"id": "a"}, "b": {"id": "b"}, "c": {"id": "c"}},
            {"a": left, "b": right, "c": left},
            [
                {"parent": {"instanceId": "a"}, "child": {"instanceId": "b"}},
                {"parent": {"instanceId": "b"}, "child": {"instanceId": "c"}},
            ],
        )
    connected = connected_overlap_warnings(
        {"a": pose, "b": pose, "c": pose},
        {"a": {"id": "a"}, "b": {"id": "b"}, "c": {"id": "c"}},
        {"a": left, "b": right, "c": left},
        [
            {"parent": {"instanceId": "a"}, "child": {"instanceId": "b"}},
            {"parent": {"instanceId": "b"}, "child": {"instanceId": "c"}},
        ],
    )
    assert len(connected) == 1
    assert connected[0]["code"] == "connected_proxy_overlap"
    assert "a/c" in connected[0]["message"]


def test_golden_three_part_compiles_through_assembly_and_competitive_validator():
    document = golden_three_part_assembly()
    compiled = compile_assembly_to_preset(document, competitive=True)
    assert compiled.compiled.physical is True
    assert compiled.preset["schemaVersion"] == PHYSICAL_SCHEMA_VERSION
    part_ids = {row["id"] for row in compiled.preset["rigidParts"]}
    assert {"chassis", "channel", "motor", "wheel"} <= part_ids
    assert compiled.preset["id"] == "golden_three_part"
    joints = {row["id"]: row for row in compiled.preset["joints"]}
    assert joints["channel_joint"]["type"] == "fixed"
    assert joints["motor_joint"]["type"] == "fixed"
    assert joints["wheel_joint"]["type"] == "hinge"
    assert joints["wheel_joint"]["parentPartId"] == "motor"
    assert compiled.report.roles["wheel"] == "drive_wheel"
    assert compiled.report.confirmed is False
    assert not any("mecanum_biobuzz_4cap" in note for note in compiled.report.notes)
    assert compiled.preset.get("launchers") == []
    competitive = compile_robot_preset(copy.deepcopy(compiled.preset), competitive=True)
    assert competitive.part_order[0] == "chassis"
    assert "channel" in compiled.compiled.part_order
    channel_pose = matrix_from_pose(compiled.instance_poses["channel"])
    motor_pose = matrix_from_pose(compiled.instance_poses["motor"])
    channel = get_part("1120-0001-0048")
    motor = get_part("5203-2402-0019")
    parent_hole = transform_point(channel_pose, hole_in_part(part_mount(channel, "web"), (2, 1)))
    child_hole = transform_point(motor_pose, hole_in_part(part_mount(motor, "face"), (0, 0)))
    assert np.allclose(parent_hole, child_hole, atol=1e-6)


def test_materialized_robot_updates_after_source_edit_and_is_copy_safe():
    document = golden_three_part_assembly()
    clear_materialize_cache()
    first = materialize_sim_robot(document)
    first["displayName"] = "mutated result"
    cached = materialize_sim_robot(document)
    assert cached["displayName"] != "mutated result"
    document["displayName"] = "edited assembly"
    updated = materialize_sim_robot(document)
    assert updated["displayName"] == "edited assembly"


def test_renamed_flywheel_binds_to_actual_catalog_joint():
    from talongym.presets.loader import load_preset

    document = copy.deepcopy(load_preset("robot", "gobilda_mecanum_starter"))
    document["id"] = "renamed_flywheel_test"
    for row in document["assembly"]["instances"]:
        if row["id"] == "flywheel_wheel":
            row["id"] = "custom_shooter"
    for row in document["assembly"]["connections"]:
        for side in ("parent", "child"):
            if row[side]["instanceId"] == "flywheel_wheel":
                row[side]["instanceId"] = "custom_shooter"
    roles = document["functionalBindings"]["instanceRoles"]
    roles["custom_shooter"] = roles.pop("flywheel_wheel")
    compiled = compile_assembly_to_preset(document)
    actuator = next(row for row in compiled.preset["actuators"] if row["id"] == "flywheel")
    joint = next(row for row in compiled.preset["joints"] if row["id"] == actuator["jointId"])
    assert joint["childPartId"] == "custom_shooter"
    assert "flywheel" not in {row["id"] for row in compiled.preset["rigidParts"]}


def test_gobilda_launcher_uses_catalog_output_motor_curve():
    from talongym.presets.loader import load_preset

    result = compile_assembly_to_preset(load_preset("robot", "gobilda_mecanum_starter"))
    flywheel = next(row for row in result.preset["actuators"] if row["id"] == "flywheel")
    assert flywheel["motor"]["freeSpeedRpm"] == 312
    assert flywheel["targetRpm"] <= 312
    assert flywheel["gearRatio"] == 1
    assert flywheel["loadInertiaKgM2"] == pytest.approx(4.2624e-05)
    assert not any(row["code"] == "launcher_motor_unverified" for row in result.warnings)
    assert any(row["code"] == "scoring_geometry_unverified" for row in result.warnings)


def test_rev_launcher_reports_unverified_output_curve():
    from talongym.presets.loader import load_preset

    result = compile_assembly_to_preset(load_preset("robot", "rev_mecanum_starter"))
    assert any(row["code"] == "launcher_motor_unverified" for row in result.warnings)
