"""Validation and compatibility stamps for physical robot presets."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import PurePosixPath
from typing import Any

PHYSICAL_SCHEMA_VERSION = "1.1.0"
ASSEMBLY_SCHEMA_VERSION = "1.2.0"


class RobotContractError(ValueError):
    """A robot preset cannot be represented by the physical simulator."""


class AssemblyError(RobotContractError):
    """A schema 1.2 catalog assembly cannot be compiled into the 1.1 physical contract."""


@dataclass(frozen=True)
class CompiledRobot:
    preset: dict[str, Any]
    part_order: tuple[str, ...]
    actuator_ids: tuple[str, ...]
    sensor_ids: tuple[str, ...]
    compatibility_stamp: str
    physical: bool


def _rows_by_id(rows: list[dict[str, Any]], label: str) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for row in rows:
        ident = str(row.get("id") or "")
        if not ident:
            raise RobotContractError(f"{label} entry is missing id")
        if ident in out:
            raise RobotContractError(f"duplicate {label} id {ident}")
        out[ident] = row
    return out


def _safe_asset_path(value: object, label: str) -> None:
    if value is None:
        return
    path = PurePosixPath(str(value))
    if path.is_absolute() or ".." in path.parts:
        raise RobotContractError(f"{label} must be a relative asset path")


def _part_order(parts: dict[str, dict[str, Any]]) -> tuple[str, ...]:
    result: list[str] = []
    visiting: set[str] = set()

    def visit(part_id: str) -> None:
        if part_id in result:
            return
        if part_id in visiting:
            raise RobotContractError(f"rigid part parent cycle at {part_id}")
        visiting.add(part_id)
        parent = parts[part_id].get("parentId")
        if parent is not None:
            if parent not in parts:
                raise RobotContractError(f"part {part_id} references missing parent {parent}")
            visit(str(parent))
        visiting.remove(part_id)
        result.append(part_id)

    for ident in parts:
        visit(ident)
    return tuple(result)


def compatibility_stamp(preset: dict[str, Any]) -> str:
    payload = {
        "schemaVersion": preset.get("schemaVersion"),
        "policyInterfaceVersion": preset.get("policyInterfaceVersion"),
        "actuators": [
            {"id": row.get("id"), "kind": row.get("kind")}
            for row in preset.get("actuators") or []
        ],
        "sensors": [
            {"id": row.get("id"), "kind": row.get("kind")}
            for row in preset.get("mechanismSensors") or []
        ],
        "defaultActionTier": preset.get("defaultActionTier"),
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return f"robot-interface:{hashlib.sha256(encoded).hexdigest()[:16]}"


def compile_robot_preset(
    preset: dict[str, Any],
    *,
    competitive: bool,
) -> CompiledRobot:
    """Validate cross-references and physical geometry omitted from JSON Schema."""
    version = str(preset.get("schemaVersion"))
    if version == ASSEMBLY_SCHEMA_VERSION:
        raise RobotContractError(
            "schema 1.2.0 assemblies must be compiled with compile_assembly_to_preset"
        )
    physical = version == PHYSICAL_SCHEMA_VERSION
    if not physical:
        if competitive:
            raise RobotContractError(
                f"competitive simulation requires robot schema {PHYSICAL_SCHEMA_VERSION}"
            )
        return CompiledRobot(
            preset=preset,
            part_order=(),
            actuator_ids=(),
            sensor_ids=(),
            compatibility_stamp=compatibility_stamp(preset),
            physical=False,
        )

    parts = _rows_by_id(list(preset.get("rigidParts") or []), "rigid part")
    joints = _rows_by_id(list(preset.get("joints") or []), "joint")
    actuators = _rows_by_id(list(preset.get("actuators") or []), "actuator")
    sensors = _rows_by_id(list(preset.get("mechanismSensors") or []), "mechanism sensor")
    order = _part_order(parts)

    roots = [ident for ident, row in parts.items() if row.get("parentId") is None]
    if len(roots) != 1:
        raise RobotContractError("physical robot must define exactly one root rigid part")
    if roots[0] != "chassis":
        raise RobotContractError("physical robot root rigid part must be named chassis")

    child_joint: dict[str, str] = {}
    for ident, joint in joints.items():
        parent = str(joint.get("parentPartId") or "")
        child = str(joint.get("childPartId") or "")
        if parent not in parts or child not in parts:
            raise RobotContractError(f"joint {ident} references a missing rigid part")
        if parts[child].get("parentId") != parent:
            raise RobotContractError(f"joint {ident} disagrees with {child}.parentId")
        if child in child_joint:
            raise RobotContractError(f"part {child} is bound by multiple joints")
        child_joint[child] = ident
        if joint.get("type") != "fixed":
            axis = list(joint.get("axis") or [])
            if len(axis) != 3 or sum(float(v) ** 2 for v in axis) < 0.99:
                raise RobotContractError(f"joint {ident} must have a normalized axis")
            limit = list(joint.get("limit") or [])
            if len(limit) != 2 or float(limit[0]) >= float(limit[1]):
                raise RobotContractError(f"joint {ident} has invalid limits")
    if set(parts) - {roots[0]} != set(child_joint):
        missing = sorted(set(parts) - {roots[0]} - set(child_joint))
        raise RobotContractError(f"moving/fixed child parts need one joint: {missing}")

    for ident, part in parts.items():
        _safe_asset_path(part.get("visualAsset"), f"part {ident} visualAsset")
        for collision in part.get("collision") or []:
            if collision.get("kind") == "convex_mesh":
                _safe_asset_path(collision.get("asset"), f"part {ident} collision asset")

    for ident, actuator in actuators.items():
        joint_id = actuator.get("jointId")
        if joint_id is not None and joint_id not in joints:
            raise RobotContractError(f"actuator {ident} references missing joint {joint_id}")
        if actuator.get("kind") in {"position_motor", "servo"} and joint_id is None:
            raise RobotContractError(f"position actuator {ident} requires jointId")

    piece_path = dict(preset.get("piecePath") or {})
    launch_capable = bool((preset.get("mechanisms") or {}).get("launchCapable"))
    if launch_capable or piece_path:
        if not piece_path:
            raise RobotContractError("launch-capable robots require piecePath")
        required_actuators = (
            "intakeActuatorId",
            "conveyorActuatorId",
            "flywheelActuatorId",
            "gateActuatorId",
        )
        for key in required_actuators:
            if piece_path.get(key) not in actuators:
                raise RobotContractError(f"piecePath.{key} references a missing actuator")
        for key in ("hoodActuatorId", "turretActuatorId"):
            if piece_path.get(key) is not None and piece_path.get(key) not in actuators:
                raise RobotContractError(f"piecePath.{key} references a missing actuator")

        slots = list(piece_path.get("storageSlots") or [])
        capacity = int((preset.get("mechanisms") or {}).get("capacity") or 0)
        if len(slots) < capacity:
            raise RobotContractError(
                f"piece path has {len(slots)} storage slots for capacity {capacity}"
            )
        chassis = preset.get("chassis") or {}
        half_length = 0.5 * float(chassis.get("lengthIn") or 0)
        half_width = 0.5 * float(chassis.get("widthIn") or 0)
        height = float(chassis.get("heightIn") or 0)
        piece_radius = 1.4
        for index, slot in enumerate(slots[:capacity]):
            if (
                abs(float(slot.get("x") or 0)) + piece_radius > half_length
                or abs(float(slot.get("y") or 0)) + piece_radius > half_width
                or not piece_radius <= float(slot.get("z") or 0) <= height - piece_radius
            ):
                raise RobotContractError(f"storage slot {index} exceeds the legal robot envelope")

        muzzle = piece_path.get("muzzlePose") or {}
        muzzle_x = float(muzzle.get("x") or 0)
        muzzle_y = float(muzzle.get("y") or 0)
        muzzle_z = float(muzzle.get("z") or 0)
        clearance = float(piece_path.get("muzzleClearanceIn") or 0)
        outside_envelope = (
            abs(muzzle_x) >= half_length + clearance
            or abs(muzzle_y) >= half_width + clearance
            or muzzle_z >= height + clearance
        )
        if not outside_envelope:
            raise RobotContractError("muzzle plus clearance intersects the robot envelope")
    elif competitive:
        mechanisms = preset.get("mechanisms") or {}
        if mechanisms.get("launchCapable") is None:
            pass

    for ident, sensor in sensors.items():
        actuator_id = sensor.get("actuatorId")
        joint_id = sensor.get("jointId")
        if actuator_id is not None and actuator_id not in actuators:
            raise RobotContractError(f"sensor {ident} references missing actuator")
        if joint_id is not None and joint_id not in joints:
            raise RobotContractError(f"sensor {ident} references missing joint")

    return CompiledRobot(
        preset=preset,
        part_order=order,
        actuator_ids=tuple(actuators),
        sensor_ids=tuple(sensors),
        compatibility_stamp=compatibility_stamp(preset),
        physical=True,
    )
