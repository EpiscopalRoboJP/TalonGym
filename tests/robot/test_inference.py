"""Catalog assembly inference, confirmation, and physical/FTC validation."""

from __future__ import annotations

import copy

import numpy as np
import pytest

from talongym.robot.assembly import compile_assembly_to_preset
from talongym.robot.contract import ASSEMBLY_SCHEMA_VERSION, AssemblyError
from talongym.robot.inference import aggregate_mass_properties, build_inference_report
from talongym.robot.transforms import matrix_from_pose
from talongym.robot.transmissions import infer_transmissions
from talongym.robot.validation import (
    ftc_soft_warnings,
    validate_coaxiality,
    validate_inferred_assembly,
    validate_travel_envelopes,
    validate_wheel_floor_contact,
)


def test_derived_inertia_is_invariant_under_whole_part_rotation():
    part = {"massKg": 1.0, "collision": [{"kind": "box", "sizeIn": [2.0, 4.0, 6.0]}]}
    traces = []
    for yaw in (0, 45, 90):
        props = aggregate_mass_properties(
            {"box": {}}, {"box": part}, {"box": matrix_from_pose({"yawDeg": yaw})}
        )
        assert props is not None
        traces.append(sum(props.inertia_kg_m2))
    assert traces == pytest.approx([traces[0]] * 3)


def _pose(x: float, y: float, z: float) -> np.ndarray:
    matrix = np.eye(4, dtype=np.float64)
    matrix[0, 3] = x
    matrix[1, 3] = y
    matrix[2, 3] = z
    return matrix


def _rotating_part(
    sku: str,
    tags: list[str],
    *,
    kind: str,
    radius: float = 1.0,
    mass: float = 0.1,
    name: str = "",
    teeth: int | None = None,
) -> dict:
    display = name or sku
    if teeth is not None:
        display = f"{display} {teeth}T"
    return {
        "sku": sku,
        "displayName": display,
        "massKg": mass,
        "tags": tags,
        "collision": [{"kind": "cylinder", "radiusIn": radius, "lengthIn": 0.5}],
        "mounts": [
            {
                "id": "bore",
                "kind": kind,
                "standard": "gobilda_8mm_rex",
                "diameterMm": 8.0,
                "axis": [0.0, 0.0, 1.0],
                "transform": {"x": 0, "y": 0, "z": 0},
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


def test_golden_assembly_infers_transmission_mass_and_confirmation():
    compiled = compile_assembly_to_preset(golden_three_part_assembly(), competitive=True)
    report = compiled.report
    assert report.roles["wheel"] == "drive_wheel"
    assert report.roles["motor"] == "drive_motor"
    assert report.roles["channel"] == "structure"
    assert report.confirmed is False
    assert "transmissions" in report.confirmation_required
    assert "drivetrain" in report.confirmation_required
    assert report.topology_fallback == ()
    assert not any("mecanum_biobuzz_4cap" in note for note in report.notes)
    assert compiled.preset.get("launchers") == []
    assert compiled.preset["mechanisms"]["launchCapable"] is False
    paths = {path.wheel_id: path for path in report.transmissions}
    assert paths["wheel"].motor_id == "motor"
    assert paths["wheel"].gear_ratio == pytest.approx(19.2, rel=1e-6)
    assert paths["wheel"].source == "shaft"
    assert report.drivetrain["type"] == "mecanum"
    assert compiled.preset["drivetrain"]["wheelDiameterIn"] == pytest.approx(3.7796, abs=1e-3)
    assert compiled.preset["motors"]["count"] == 1
    assert compiled.preset["motors"]["gearRatio"] == pytest.approx(19.2)
    assert report.mass is not None
    assert report.mass.mass_kg == pytest.approx(0.028 + 0.315 + 0.165)
    assert compiled.preset["chassis"]["lengthIn"] == 18
    assert compiled.preset["defaultActionTier"] == "high_level_waypoint"
    codes = {row["code"] for row in compiled.warnings}
    assert "ftc_season_parity" not in codes
    assert "optional_electronics" in codes
    assert compiled.compiled.physical is True


def test_scoring_overlay_requires_catalog_intake_and_flywheel():
    document = golden_three_part_assembly()
    document["functionalBindings"] = {"includeScoringTopology": True}
    with pytest.raises(AssemblyError, match="intake_roller"):
        compile_assembly_to_preset(document, competitive=True)


def test_confirmed_bindings_clear_confirmation_gate():
    document = golden_three_part_assembly()
    document["functionalBindings"] = {
        "confirmed": True,
        "instanceRoles": {"wheel": "drive_wheel", "motor": "drive_motor", "channel": "structure"},
        "drivetrain": {"type": "mecanum", "trackWidthIn": 14, "wheelbaseIn": 14},
    }
    compiled = compile_assembly_to_preset(document, competitive=True)
    assert compiled.report.confirmed is True
    assert compiled.report.confirmation_required == ()
    assert compiled.preset["drivetrain"]["trackWidthIn"] == 14
    assert compiled.preset["defaultActionTier"] == "physical_actuators"


def test_confirmed_boolean_without_roles_does_not_enable_physical_actuators():
    document = golden_three_part_assembly()
    document["functionalBindings"] = {"confirmed": True}
    compiled = compile_assembly_to_preset(document, competitive=True)
    assert compiled.report.confirmed is False
    assert "instanceRoles" in compiled.report.confirmation_required
    assert compiled.preset["defaultActionTier"] == "high_level_waypoint"


def test_mecanum_and_tank_from_wheel_tags():
    mecanum_parts = {f"w{i}": _rotating_part(f"mw{i}", ["wheel_mecanum"], kind="hub", radius=2.0) for i in range(4)}
    mecanum_instances = {ident: {"id": ident, "sku": ident} for ident in mecanum_parts}
    poses = {
        "w0": _pose(-6, 6, 2),
        "w1": _pose(-6, -6, 2),
        "w2": _pose(6, 6, 2),
        "w3": _pose(6, -6, 2),
    }
    report = build_inference_report(mecanum_instances, mecanum_parts, {}, poses=poses)
    assert report.drivetrain["type"] == "mecanum"
    assert report.drivetrain["complete"] is True
    assert report.drivetrain["trackWidthIn"] == pytest.approx(12)
    assert report.drivetrain["wheelbaseIn"] == pytest.approx(12)

    tank_parts = {
        "left": _rotating_part("twl", ["wheel_traction"], kind="hub", radius=2.0),
        "right": _rotating_part("twr", ["wheel_traction"], kind="hub", radius=2.0),
    }
    tank_instances = {ident: {"id": ident, "sku": ident} for ident in tank_parts}
    tank_poses = {"left": _pose(0, 5, 2), "right": _pose(0, -5, 2)}
    tank_report = build_inference_report(tank_instances, tank_parts, {}, poses=tank_poses)
    assert tank_report.drivetrain["type"] == "tank"
    assert tank_report.drivetrain["complete"] is True
    assert tank_report.drivetrain["strafeMultiplier"] == 0.0


def test_gear_mesh_compounds_ratio_along_transmission():
    motor = _rotating_part(
        "motor",
        ["motor", "gearbox"],
        kind="shaft",
        radius=0.5,
        name="Test Motor 10.0:1 100 RPM",
    )
    gear_a = _rotating_part("gear-0020", ["gear"], kind="hub", radius=1.0, teeth=20)
    gear_b = _rotating_part("gear-0040", ["gear"], kind="hub", radius=2.0, teeth=40)
    wheel = _rotating_part("wheel", ["wheel_traction"], kind="hub", radius=2.0)
    parts = {"motor": motor, "gear_a": gear_a, "gear_b": gear_b, "wheel": wheel}
    instances = {ident: {"id": ident, "sku": ident} for ident in parts}
    connections = {
        "gear_a": {
            "id": "motor_gear",
            "parent": {"instanceId": "motor", "mountId": "bore"},
            "child": {"instanceId": "gear_a", "mountId": "bore"},
        },
        "wheel": {
            "id": "wheel_gear",
            "parent": {"instanceId": "gear_b", "mountId": "bore"},
            "child": {"instanceId": "wheel", "mountId": "bore"},
        },
    }
    poses = {
        "motor": _pose(0, 0, 0),
        "gear_a": _pose(0, 0, 0),
        "gear_b": _pose(0, 3, 0),
        "wheel": _pose(0, 3, 0),
    }
    paths = infer_transmissions(instances, parts, connections, poses=poses)
    assert len(paths) == 1
    assert paths[0].motor_id == "motor"
    assert paths[0].source == "gear_mesh"
    assert paths[0].gear_ratio == pytest.approx(20.0)
    assert paths[0].free_speed_rpm == pytest.approx(50.0)


def test_ultra_planetary_cartridge_compounds_selected_ratio():
    motor = _rotating_part("REV-41-1600", ["motor"], kind="shaft", radius=0.4, name="HD Hex Motor")
    cartridge = {
        "sku": "REV-41-1603",
        "displayName": "UltraPlanetary Cartridge 5:1",
        "gearRatio": 5.0,
        "massKg": 0.04,
        "tags": ["gearbox"],
        "collision": [{"kind": "cylinder", "radiusIn": 0.8, "lengthIn": 0.7}],
        "mounts": [
            {
                "id": "bore",
                "kind": "hub",
                "standard": "rev_5mm_hex",
                "diameterMm": 5.0,
                "axis": [0.0, 0.0, 1.0],
                "transform": {"x": 0, "y": 0, "z": 0},
            }
        ],
    }
    wheel = _rotating_part("wheel", ["wheel_traction"], kind="hub", radius=2.0)
    parts = {"motor": motor, "cartridge": cartridge, "wheel": wheel}
    instances = {ident: {"id": ident, "sku": ident} for ident in parts}
    connections = {
        "cartridge": {
            "id": "cart",
            "parent": {"instanceId": "motor", "mountId": "bore"},
            "child": {"instanceId": "cartridge", "mountId": "bore"},
        },
        "wheel": {
            "id": "wheel",
            "parent": {"instanceId": "cartridge", "mountId": "bore"},
            "child": {"instanceId": "wheel", "mountId": "bore"},
        },
    }
    poses = {"motor": _pose(0, 0, 0), "cartridge": _pose(0, 0, 0), "wheel": _pose(0, 0, 0)}
    paths = infer_transmissions(instances, parts, connections, poses=poses)
    assert len(paths) == 1
    assert paths[0].motor_id == "motor"
    assert "cartridge" in paths[0].instance_ids
    assert paths[0].source == "gearbox_stack"
    assert paths[0].gear_ratio == pytest.approx(5.0)


def test_coaxiality_is_a_hard_error():
    parent = _rotating_part("shaft", ["shaft"], kind="shaft")
    child = _rotating_part("hub", ["hub"], kind="hub")
    poses = {"a": _pose(0, 0, 0), "b": _pose(0, 1, 0)}
    connections = [
        {
            "id": "bad",
            "parent": {"instanceId": "a", "mountId": "bore"},
            "child": {"instanceId": "b", "mountId": "bore"},
        }
    ]
    with pytest.raises(AssemblyError, match="not coaxial"):
        validate_coaxiality(poses, {"a": parent, "b": child}, connections)


def test_travel_self_intersection_is_a_hard_error():
    parent = {
        "sku": "plate",
        "tags": ["plate"],
        "massKg": 0.2,
        "collision": [{"kind": "box", "sizeIn": [4, 4, 1]}],
        "mounts": [
            {
                "id": "face",
                "kind": "shaft",
                "standard": "gobilda_8mm_rex",
                "diameterMm": 8.0,
                "axis": [0.0, 1.0, 0.0],
                "transform": {"x": 0, "y": 0, "z": 0},
            }
        ],
    }
    child = {
        "sku": "arm",
        "tags": ["plate"],
        "massKg": 0.2,
        "collision": [{"kind": "box", "sizeIn": [6, 1, 1]}],
        "mounts": [
            {
                "id": "hub",
                "kind": "hub",
                "standard": "gobilda_8mm_rex",
                "diameterMm": 8.0,
                "axis": [0.0, 1.0, 0.0],
                "transform": {"x": 0, "y": 0, "z": 0},
            }
        ],
    }
    blocker = {
        "sku": "wall",
        "tags": ["plate"],
        "massKg": 0.2,
        "collision": [{"kind": "box", "sizeIn": [1, 4, 4], "pose": {"x": 3, "y": 0, "z": 0}}],
        "mounts": [
            {
                "id": "face",
                "kind": "mating_face",
                "standard": "gobilda_pattern",
                "axis": [1.0, 0.0, 0.0],
                "transform": {"x": 0, "y": 0, "z": 0},
            }
        ],
    }
    poses = {"root": _pose(0, 0, 0), "arm": _pose(0, 0, 0), "wall": _pose(3, 0, 0)}
    connections = {
        "arm": {
            "id": "hinge",
            "jointType": "hinge",
            "limit": [0, 90],
            "parent": {"instanceId": "root", "mountId": "face"},
            "child": {"instanceId": "arm", "mountId": "hub"},
        }
    }
    instances = {"root": {"id": "root"}, "arm": {"id": "arm"}, "wall": {"id": "wall"}}
    parts = {"root": parent, "arm": child, "wall": blocker}
    with pytest.raises(AssemblyError, match="travel self-intersects"):
        validate_travel_envelopes(poses, instances, parts, connections)


def test_complete_drivetrain_rejects_uneven_wheel_heights():
    parts = {
        "w0": _rotating_part("w0", ["wheel_mecanum"], kind="hub", radius=2.0),
        "w1": _rotating_part("w1", ["wheel_mecanum"], kind="hub", radius=2.0),
        "w2": _rotating_part("w2", ["wheel_mecanum"], kind="hub", radius=2.0),
        "w3": _rotating_part("w3", ["wheel_mecanum"], kind="hub", radius=2.0),
    }
    instances = {ident: {"id": ident} for ident in parts}
    poses = {
        "w0": _pose(-6, 6, 2),
        "w1": _pose(-6, -6, 2),
        "w2": _pose(6, 6, 2),
        "w3": _pose(6, -6, 3),
    }
    report = build_inference_report(instances, parts, {}, poses=poses)
    with pytest.raises(AssemblyError, match="floor contact"):
        validate_wheel_floor_contact(poses, parts, report)


def test_soft_ftc_envelope_mass_season_and_electronics_warnings():
    parts = {
        "heavy": {
            "sku": "brick",
            "tags": ["plate"],
            "massKg": 20.0,
            "collision": [{"kind": "box", "sizeIn": [20, 4, 4]}],
            "mounts": [
                {
                    "id": "face",
                    "kind": "mating_face",
                    "standard": "gobilda_pattern",
                    "axis": [0.0, 0.0, 1.0],
                    "transform": {"x": 0, "y": 0, "z": 0},
                }
            ],
        }
    }
    instances = {"heavy": {"id": "heavy", "sku": "brick"}}
    poses = {"heavy": _pose(0, 0, 0)}
    report = build_inference_report(instances, parts, {}, poses=poses)
    warnings = ftc_soft_warnings(parts, report)
    codes = {row["code"] for row in warnings}
    assert "ftc_starting_envelope" in codes
    assert "ftc_mass" in codes
    assert "optional_electronics" in codes
    validate_inferred_assembly(poses, instances, parts, {}, report)


def test_golden_compile_does_not_require_topology_wheel_diameter():
    compiled = compile_assembly_to_preset(copy.deepcopy(golden_three_part_assembly()), competitive=True)
    assert compiled.preset["drivetrain"]["wheelDiameterIn"] != 4
    assert compiled.preset["motors"]["count"] != 4
    competitive = compile_assembly_to_preset(golden_three_part_assembly(), competitive=True)
    assert competitive.compiled.physical is True
