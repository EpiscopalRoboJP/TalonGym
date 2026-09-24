"""Checked-in complete scoring starters: goBILDA/REV × mecanum/tank."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from talongym.paths import PRESETS_DIR
from talongym.presets.loader import validate_document
from talongym.robot.assembly import CompiledAssembly, compile_assembly_to_preset
from talongym.robot.catalog import get_part
from talongym.robot.contract import ASSEMBLY_SCHEMA_VERSION, AssemblyError
from talongym.robot.inference import infer_role
from talongym.robot.recipes import RecipeError, instantiate_recipe
from talongym.robot.transmissions import part_tags

STARTER_DIR = PRESETS_DIR / "robots"
STARTER_IDS = (
    "gobilda_mecanum_starter",
    "gobilda_tank_starter",
    "rev_mecanum_starter",
    "rev_tank_starter",
)
_RECIPE_FOR_STARTER = {
    "gobilda_mecanum_starter": "gobilda_mecanum",
    "gobilda_tank_starter": "gobilda_tank",
    "rev_mecanum_starter": "rev_mecanum",
    "rev_tank_starter": "rev_tank",
}
_GOBILDA_INTAKE = "3615-4008-0072"
_GOBILDA_FLYWHEEL = "3615-4008-0096"
_GOBILDA_HOOD = "1202-0001-0001"
_GOBILDA_GATE = "2000-0025-0002"
_REV_INTAKE = "REV-41-2034"
_REV_FLYWHEEL = "REV-41-1267"
_REV_HOOD = "REV-41-1305"
_REV_GATE = "REV-41-1097"


class StarterError(RuntimeError):
    pass


def _ref(instance_id: str, mount_id: str, index: tuple[int, int] | None = None) -> dict[str, Any]:
    row: dict[str, Any] = {"instanceId": instance_id, "mountId": mount_id}
    if index is not None:
        row["patternIndex"] = [int(index[0]), int(index[1])]
    return row


def _conn(
    ident: str,
    parent_id: str,
    parent_mount: str,
    parent_index: tuple[int, int] | None,
    child_id: str,
    child_mount: str,
    child_index: tuple[int, int] | None,
    secondary: tuple[tuple[int, int], tuple[int, int]] | None = None,
    *,
    spin_deg: float = 0.0,
    joint_type: str | None = None,
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "id": ident,
        "parent": _ref(parent_id, parent_mount, parent_index),
        "child": _ref(child_id, child_mount, child_index),
    }
    if secondary:
        row["secondary"] = {
            "parent": _ref(parent_id, parent_mount, secondary[0]),
            "child": _ref(child_id, child_mount, secondary[1]),
        }
    if abs(float(spin_deg)) > 1e-12:
        row["spinDeg"] = float(spin_deg)
    if joint_type:
        row["jointType"] = joint_type
    return row


def _instance(ident: str, sku: str) -> dict[str, Any]:
    return {"id": ident, "sku": sku}


def _pattern_counts(sku: str, mount_id: str) -> tuple[int, int]:
    from talongym.robot.mounts import part_mount

    pattern = part_mount(get_part(sku), mount_id).get("pattern") or {}
    return int(pattern.get("countU") or 1), int(pattern.get("countV") or 1)


def _extend_unique(instances: list[dict[str, Any]], extras: list[dict[str, Any]]) -> None:
    ids = {row["id"] for row in instances}
    instances.extend(row for row in extras if row["id"] not in ids)


def _next_flange_pair(flange_u: int, occupied: set[int], start: int = 2) -> int:
    for hole in range(max(start, 0), flange_u - 2):
        footprint = {hole, hole + 1, hole + 2}
        if occupied & footprint:
            continue
        occupied.update(footprint)
        return hole
    raise StarterError(f"U-channel flange {flange_u} has no free 16 mm motor pair")


def _occupied_flange_holes(connections: list[dict[str, Any]], rail_id: str) -> set[int]:
    occupied: set[int] = set()
    refs: list[dict[str, Any]] = []
    for row in connections:
        refs.append(row.get("parent") or {})
        refs.append(row.get("child") or {})
        secondary = row.get("secondary") or {}
        refs.append(secondary.get("parent") or {})
        refs.append(secondary.get("child") or {})
    for ref in refs:
        if str(ref.get("instanceId") or "") != rail_id or str(ref.get("mountId") or "") != "flange_a":
            continue
        index = ref.get("patternIndex") or []
        if index:
            hole = int(index[0])
            occupied.update({hole - 1, hole, hole + 1})
    return occupied


def _attach_gobilda_scoring(document: dict[str, Any]) -> None:
    assembly = document["assembly"]
    instances: list[dict[str, Any]] = assembly["instances"]
    connections: list[dict[str, Any]] = assembly["connections"]
    ids = {row["id"] for row in instances}
    motor_sku = next(row["sku"] for row in instances if row["id"] == "motor_fl")
    length_sku = next(row["sku"] for row in instances if row["id"] == "left_rail")
    flange_u, _ = _pattern_counts(length_sku, "flange_a")
    left_used = _occupied_flange_holes(connections, "left_rail")
    right_used = _occupied_flange_holes(connections, "right_rail")
    intake_u = _next_flange_pair(flange_u, left_used)
    conveyor_u = _next_flange_pair(flange_u, left_used, start=intake_u + 1)
    launcher_u = _next_flange_pair(flange_u, right_used)
    extras = [
        _instance("intake_motor", motor_sku),
        _instance("intake_wheel", _GOBILDA_INTAKE),
        _instance("conveyor_motor", motor_sku),
        _instance("conveyor_wheel", _GOBILDA_INTAKE),
        _instance("launcher_motor", motor_sku),
        _instance("flywheel_wheel", _GOBILDA_FLYWHEEL),
        _instance("hood_plate", _GOBILDA_HOOD),
    ]
    if "servo_front" not in ids:
        extras.append(_instance("gate_servo", _GOBILDA_GATE))
    _extend_unique(instances, extras)
    connections.extend(
        [
            _conn("intake_motor", "left_rail", "flange_a", (intake_u, 0), "intake_motor", "face", (0, 0), ((intake_u + 2, 0), (1, 0))),
            _conn("intake_wheel", "intake_motor", "output", None, "intake_wheel", "bore", None, joint_type="hinge"),
            _conn("conveyor_motor", "left_rail", "flange_a", (conveyor_u, 0), "conveyor_motor", "face", (0, 0), ((conveyor_u + 2, 0), (1, 0))),
            _conn("conveyor_wheel", "conveyor_motor", "output", None, "conveyor_wheel", "bore", None, joint_type="hinge"),
            _conn("launcher_motor", "right_rail", "flange_a", (launcher_u, 0), "launcher_motor", "face", (0, 0), ((launcher_u + 2, 0), (1, 0))),
            _conn(
                "flywheel_wheel",
                "launcher_motor",
                "output",
                None,
                "flywheel_wheel",
                "bore",
                None,
                joint_type="hinge",
            ),
            _conn("hood_plate", "front_rail", "web", (2, 0), "hood_plate", "pattern", (0, 0)),
        ]
    )
    if "servo_front" not in ids:
        connections.append(
            _conn("gate_servo", "hood_plate", "pattern", (1, 1), "gate_servo", "tabs", (0, 0), joint_type="hinge")
        )


def _attach_rev_scoring(document: dict[str, Any]) -> None:
    assembly = document["assembly"]
    instances: list[dict[str, Any]] = assembly["instances"]
    connections: list[dict[str, Any]] = assembly["connections"]
    ids = {row["id"] for row in instances}
    length_sku = next(row["sku"] for row in instances if row["id"] == "left_rail")
    motor_sku = next(row["sku"] for row in instances if row["id"] == "motor_fl")
    length_u, _ = _pattern_counts(length_sku, "slot")
    intake_u = max(2, min(length_u - 3, 6))
    extras = [
        _instance("intake_up", "REV-41-1621"),
        _instance("intake_motor", motor_sku),
        _instance("intake_wheel", _REV_INTAKE),
        _instance("conveyor_up", "REV-41-1621"),
        _instance("conveyor_motor", motor_sku),
        _instance("conveyor_wheel", _REV_INTAKE),
        _instance("launcher_up", "REV-41-1621"),
        _instance("launcher_motor", motor_sku),
        _instance("flywheel_wheel", _REV_FLYWHEEL),
        _instance("hood_plate", _REV_HOOD),
        _instance("gate_servo", _REV_GATE),
    ]
    _extend_unique(instances, extras)
    connections.extend(
        [
            _conn(
                "intake_up",
                "left_rail",
                "slot",
                (intake_u, 0),
                "intake_up",
                "extrusion",
                (0, 0),
                ((intake_u + 1, 0), (1, 0)),
            ),
            _conn("intake_motor", "intake_up", "motor_face", (0, 0), "intake_motor", "face", (0, 0)),
            _conn("intake_wheel", "intake_motor", "output", None, "intake_wheel", "bore", None, joint_type="hinge"),
            _conn(
                "conveyor_up",
                "left_rail",
                "slot",
                (10, 0),
                "conveyor_up",
                "extrusion",
                (0, 0),
                ((11, 0), (1, 0)),
            ),
            _conn("conveyor_motor", "conveyor_up", "motor_face", (0, 0), "conveyor_motor", "face", (0, 0)),
            _conn("conveyor_wheel", "conveyor_motor", "output", None, "conveyor_wheel", "bore", None, joint_type="hinge"),
            _conn("launcher_up", "right_rail", "slot", (8, 0), "launcher_up", "extrusion", (0, 0), ((9, 0), (1, 0))),
            _conn("launcher_motor", "launcher_up", "motor_face", (0, 0), "launcher_motor", "face", (0, 0)),
            _conn(
                "flywheel_wheel",
                "launcher_motor",
                "output",
                None,
                "flywheel_wheel",
                "bore",
                None,
                joint_type="hinge",
            ),
            _conn("hood_plate", "front_rail", "slot", (2, 0), "hood_plate", "face_a", (0, 0)),
        ]
    )
    if "gate_servo" not in ids:
        connections.append(
            _conn("gate_servo", "hood_plate", "face_b", (0, 0), "gate_servo", "tabs", (0, 0), joint_type="hinge")
        )


def _instance_roles(document: dict[str, Any]) -> dict[str, str]:
    roles: dict[str, str] = {}
    for row in document["assembly"]["instances"]:
        ident = str(row["id"])
        part = get_part(str(row["sku"]))
        tags = part_tags(part)
        if ident in {"intake_wheel", "conveyor_wheel"} or "intake_roller" in tags:
            roles[ident] = "intake"
        elif ident == "flywheel_wheel" or "flywheel" in tags:
            roles[ident] = "flywheel"
        elif ident in {"gate_servo", "servo_front"} or "servo" in tags:
            roles[ident] = "servo"
        elif ident in {"intake_motor", "conveyor_motor", "launcher_motor"}:
            roles[ident] = "drive_motor"
        elif ident == "hood_plate":
            roles[ident] = "structure"
        else:
            roles[ident] = infer_role(part, instance_id=ident)
    return roles


def build_starter_document(starter_id: str) -> dict[str, Any]:
    if starter_id not in _RECIPE_FOR_STARTER:
        raise StarterError(f"unknown starter {starter_id}")
    recipe_id = _RECIPE_FOR_STARTER[starter_id]
    document = instantiate_recipe(recipe_id)
    if recipe_id.startswith("gobilda"):
        _attach_gobilda_scoring(document)
    else:
        _attach_rev_scoring(document)
    drivetrain = dict((document.get("functionalBindings") or {}).get("drivetrain") or {})
    roles = _instance_roles(document)
    document["id"] = starter_id
    document["displayName"] = {
        "gobilda_mecanum_starter": "goBILDA mecanum scoring starter",
        "gobilda_tank_starter": "goBILDA tank scoring starter",
        "rev_mecanum_starter": "REV mecanum scoring starter",
        "rev_tank_starter": "REV tank scoring starter",
    }[starter_id]
    document["schemaVersion"] = ASSEMBLY_SCHEMA_VERSION
    document["drivetrain"] = drivetrain
    document["functionalBindings"] = {
        "confirmed": True,
        "includeScoringTopology": True,
        "drivetrain": drivetrain,
        "instanceRoles": roles,
    }
    compiled = compile_assembly_to_preset(document, competitive=True)
    for key in ("chassis", "motors", "constraints", "sensors", "mechanisms", "defaultActionTier"):
        value = compiled.preset.get(key)
        if value is not None:
            document[key] = value
    errors = validate_document("robot", document)
    if errors:
        raise StarterError(f"{starter_id} failed schema: " + "; ".join(errors[:8]))
    return document


def compile_starter(starter_id: str, *, competitive: bool = True) -> CompiledAssembly:
    document = build_starter_document(starter_id)
    try:
        compiled = compile_assembly_to_preset(document, competitive=competitive)
    except (AssemblyError, RecipeError) as exc:
        raise StarterError(f"{starter_id} failed compile: {exc}") from exc
    if not compiled.preset.get("launchers"):
        raise StarterError(f"{starter_id} is missing launchers after scoring overlay")
    if int((compiled.preset.get("mechanisms") or {}).get("capacity") or 0) < 4:
        raise StarterError(f"{starter_id} must hold a 4-piece preload")
    if not compiled.preset.get("intakes"):
        raise StarterError(f"{starter_id} is missing intakes after scoring overlay")
    return compiled


def write_starter_preset(starter_id: str, dest: Path | None = None) -> Path:
    document = build_starter_document(starter_id)
    path = dest or (STARTER_DIR / f"{starter_id}.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")
    return path


def write_all_starters() -> list[Path]:
    return [write_starter_preset(starter_id) for starter_id in STARTER_IDS]


def starter_skus(starter_id: str) -> list[str]:
    document = build_starter_document(starter_id)
    return sorted({str(row["sku"]) for row in document["assembly"]["instances"]})
