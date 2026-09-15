"""Robot contract: physical schema, legacy compatibility, binding safety."""

from __future__ import annotations

import copy

import pytest

from talongym.presets.loader import load_preset
from talongym.robot.contract import (
    PHYSICAL_SCHEMA_VERSION,
    RobotContractError,
    compatibility_stamp,
    compile_robot_preset,
)
from talongym.sim.world import World


def _physical() -> dict:
    return copy.deepcopy(load_preset("robot", "mecanum_biobuzz_4cap"))


def _legacy() -> dict:
    return copy.deepcopy(load_preset("robot", "mecanum_meepmeep_defaults"))


def test_physical_biobuzz_compiles_when_competitive():
    compiled = compile_robot_preset(_physical(), competitive=True)
    assert compiled.physical is True
    assert compiled.preset["schemaVersion"] == PHYSICAL_SCHEMA_VERSION
    assert compiled.part_order[0] == "chassis"
    assert "flywheel" in compiled.actuator_ids
    assert "hood_position" in compiled.sensor_ids
    assert compiled.compatibility_stamp.startswith("robot-interface:")


def test_physical_schema_still_compiles_in_non_competitive_mode():
    compiled = compile_robot_preset(_physical(), competitive=False)
    assert compiled.physical is True
    assert compiled.actuator_ids == compile_robot_preset(_physical(), competitive=True).actuator_ids


def test_legacy_robot_allowed_when_not_competitive():
    compiled = compile_robot_preset(_legacy(), competitive=False)
    assert compiled.physical is False
    assert compiled.part_order == ()
    assert compiled.actuator_ids == ()
    assert compiled.sensor_ids == ()
    assert compiled.compatibility_stamp == compatibility_stamp(_legacy())


def test_legacy_robot_rejected_when_competitive():
    with pytest.raises(RobotContractError, match="competitive simulation requires robot schema"):
        compile_robot_preset(_legacy(), competitive=True)


def test_mesh_season_competitive_flag_matches_world_mesh_requirement():
    """Training must not silently treat a mesh season as non-competitive."""
    from talongym.assets.cad_common import mesh_required
    from talongym.presets.loader import load_bundle

    bundle = load_bundle("biobuzz_2026_field_v1", "mecanum_biobuzz_4cap", "biobuzz_2026_scoring_v1")
    world = World(bundle, seed=0, allow_missing_mesh=True)
    assert world.needs_mesh() is True
    assert mesh_required(bundle.field) is True
    compile_robot_preset(bundle.robot, competitive=mesh_required(bundle.field))
    with pytest.raises(RobotContractError, match="competitive simulation requires robot schema"):
        compile_robot_preset(_legacy(), competitive=mesh_required(bundle.field))


def test_compatibility_stamp_stable_and_sensitive_to_interface():
    robot = _physical()
    first = compile_robot_preset(robot, competitive=True).compatibility_stamp
    second = compile_robot_preset(copy.deepcopy(robot), competitive=True).compatibility_stamp
    assert first == second
    robot["actuators"][0]["kind"] = "servo"
    changed = compile_robot_preset(robot, competitive=True).compatibility_stamp
    assert changed != first


def test_duplicate_actuator_id_rejected():
    robot = _physical()
    robot["actuators"].append(copy.deepcopy(robot["actuators"][0]))
    with pytest.raises(RobotContractError, match="duplicate actuator"):
        compile_robot_preset(robot, competitive=True)


def test_duplicate_rigid_part_and_joint_ids_rejected():
    robot = _physical()
    robot["rigidParts"].append(copy.deepcopy(robot["rigidParts"][0]))
    with pytest.raises(RobotContractError, match="duplicate rigid part"):
        compile_robot_preset(robot, competitive=True)
    robot = _physical()
    robot["joints"].append(copy.deepcopy(robot["joints"][0]))
    with pytest.raises(RobotContractError, match="duplicate joint"):
        compile_robot_preset(robot, competitive=True)


def test_missing_piece_path_actuator_rejected():
    robot = _physical()
    robot["piecePath"]["flywheelActuatorId"] = "not_a_motor"
    with pytest.raises(RobotContractError, match="piecePath.flywheelActuatorId"):
        compile_robot_preset(robot, competitive=True)


def test_optional_turret_binding_must_exist_when_set():
    robot = _physical()
    robot["piecePath"]["turretActuatorId"] = "missing_turret"
    with pytest.raises(RobotContractError, match="piecePath.turretActuatorId"):
        compile_robot_preset(robot, competitive=True)
    robot = _physical()
    robot["piecePath"]["turretActuatorId"] = None
    compile_robot_preset(robot, competitive=True)


def test_position_actuator_requires_joint_binding():
    robot = _physical()
    hood = next(row for row in robot["actuators"] if row["id"] == "hood")
    hood["jointId"] = None
    with pytest.raises(RobotContractError, match="position actuator hood requires jointId"):
        compile_robot_preset(robot, competitive=True)


def test_actuator_and_sensor_missing_cross_references_rejected():
    robot = _physical()
    robot["actuators"][0]["jointId"] = "no_such_joint"
    with pytest.raises(RobotContractError, match="missing joint"):
        compile_robot_preset(robot, competitive=True)
    robot = _physical()
    robot["mechanismSensors"][0]["actuatorId"] = "ghost"
    with pytest.raises(RobotContractError, match="missing actuator"):
        compile_robot_preset(robot, competitive=True)


def test_child_part_needs_exactly_one_joint():
    robot = _physical()
    robot["joints"] = [row for row in robot["joints"] if row["id"] != "flywheel_joint"]
    with pytest.raises(RobotContractError, match="moving/fixed child parts need one joint"):
        compile_robot_preset(robot, competitive=True)
    robot = _physical()
    extra = copy.deepcopy(next(row for row in robot["joints"] if row["childPartId"] == "flywheel"))
    extra["id"] = "flywheel_joint_dup"
    robot["joints"].append(extra)
    with pytest.raises(RobotContractError, match="bound by multiple joints"):
        compile_robot_preset(robot, competitive=True)


def test_unsafe_muzzle_inside_envelope_rejected():
    robot = _physical()
    robot["piecePath"]["muzzlePose"] = {"x": 0.0, "y": 0.0, "z": 7.0, "pitchDeg": 52}
    robot["piecePath"]["muzzleClearanceIn"] = 0.25
    with pytest.raises(RobotContractError, match="muzzle plus clearance intersects"):
        compile_robot_preset(robot, competitive=True)


def test_muzzle_on_envelope_face_without_clearance_rejected():
    robot = _physical()
    half_length = 0.5 * float(robot["chassis"]["lengthIn"])
    robot["piecePath"]["muzzlePose"] = {"x": half_length, "y": 0.0, "z": 7.0}
    robot["piecePath"]["muzzleClearanceIn"] = 0.25
    with pytest.raises(RobotContractError, match="muzzle plus clearance intersects"):
        compile_robot_preset(robot, competitive=True)


def test_legal_muzzle_outside_length_plus_clearance_accepted():
    robot = _physical()
    compiled = compile_robot_preset(robot, competitive=True)
    muzzle = compiled.preset["piecePath"]["muzzlePose"]
    assert abs(float(muzzle["x"])) > 0.5 * float(robot["chassis"]["lengthIn"])


def test_storage_slots_must_cover_capacity_and_fit_envelope():
    robot = _physical()
    robot["piecePath"]["storageSlots"] = robot["piecePath"]["storageSlots"][:2]
    with pytest.raises(RobotContractError, match="storage slots for capacity"):
        compile_robot_preset(robot, competitive=True)
    robot = _physical()
    robot["piecePath"]["storageSlots"][0] = {"x": 0.0, "y": 0.0, "z": 0.1}
    with pytest.raises(RobotContractError, match="storage slot 0 exceeds"):
        compile_robot_preset(robot, competitive=True)


def test_unsafe_asset_paths_and_part_graph_rejected():
    robot = _physical()
    robot["rigidParts"][0]["visualAsset"] = "../escape.glb"
    with pytest.raises(RobotContractError, match="relative asset path"):
        compile_robot_preset(robot, competitive=True)
    robot = _physical()
    robot["rigidParts"].append(
        {
            "id": "loop_a",
            "parentId": "loop_b",
            "pose": {"x": 0, "y": 0, "z": 0},
            "massKg": 0.1,
            "collision": [{"kind": "box", "sizeIn": [1, 1, 1]}],
        }
    )
    robot["rigidParts"].append(
        {
            "id": "loop_b",
            "parentId": "loop_a",
            "pose": {"x": 0, "y": 0, "z": 0},
            "massKg": 0.1,
            "collision": [{"kind": "box", "sizeIn": [1, 1, 1]}],
        }
    )
    robot["joints"].extend(
        [
            {
                "id": "loop_a_joint",
                "type": "fixed",
                "parentPartId": "loop_b",
                "childPartId": "loop_a",
            },
            {
                "id": "loop_b_joint",
                "type": "fixed",
                "parentPartId": "loop_a",
                "childPartId": "loop_b",
            },
        ]
    )
    with pytest.raises(RobotContractError, match="parent cycle"):
        compile_robot_preset(robot, competitive=True)


def test_root_must_be_exactly_chassis():
    robot = _physical()
    robot["rigidParts"][0]["id"] = "base"
    for row in robot["joints"]:
        if row["parentPartId"] == "chassis":
            row["parentPartId"] = "base"
        if row["childPartId"] == "chassis":
            row["childPartId"] = "base"
    for row in robot["rigidParts"]:
        if row.get("parentId") == "chassis":
            row["parentId"] = "base"
    with pytest.raises(RobotContractError, match="root rigid part must be named chassis"):
        compile_robot_preset(robot, competitive=True)
