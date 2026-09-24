"""Infer catalog roles, transmissions, mass, and confirmation requirements."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from talongym.robot.collision import assembly_aabb, part_aabbs, union_aabb
from talongym.robot.mounts import inferred_joint_type, part_mount
from talongym.robot.transmissions import (
    STRUCTURE_TAGS,
    WHEEL_TAGS,
    TransmissionPath,
    infer_drivetrain,
    infer_transmissions,
    part_tags,
)

TOPOLOGY_NOTE = "competitive piecePath/actuators from mecanum_biobuzz_4cap (parity fallback; unconfirmed)"
INCHES_TO_METERS = 0.0254
DRIVE_TAGS = frozenset({"motor", "gearbox"})
INTAKE_TAGS = frozenset({"intake_roller"})
FLY_TAGS = frozenset({"flywheel"})
SENSOR_TAGS = frozenset({"sensor"})
SERVO_TAGS = frozenset({"servo"})
TOPOLOGY_FALLBACK_FIELDS = (
    "piecePath",
    "actuators",
    "intakes",
    "launchers",
    "mechanismSensors",
    "mechanisms",
    "powerSystem",
)


@dataclass(frozen=True)
class MassProperties:
    mass_kg: float
    com_in: tuple[float, float, float]
    inertia_kg_m2: tuple[float, float, float]
    aabb_min_in: tuple[float, float, float]
    aabb_max_in: tuple[float, float, float]


@dataclass(frozen=True)
class InferenceReport:
    confirmed: bool
    roles: dict[str, str]
    joints: tuple[dict[str, str], ...]
    notes: tuple[str, ...]
    welded_instance_ids: tuple[str, ...]
    articulated_instance_ids: tuple[str, ...]
    drivetrain: dict[str, Any]
    transmissions: tuple[TransmissionPath, ...]
    mechanisms: dict[str, tuple[str, ...]]
    sensors: tuple[dict[str, Any], ...]
    mass: MassProperties | None
    confirmation_required: tuple[str, ...]
    topology_fallback: tuple[str, ...]


def infer_role(part: dict[str, Any], bindings: dict[str, Any] | None = None, instance_id: str = "") -> str:
    bound = (bindings or {}).get("instanceRoles") or {}
    if instance_id and instance_id in bound:
        return str(bound[instance_id])
    tags = part_tags(part)
    if tags & INTAKE_TAGS:
        return "intake"
    if tags & FLY_TAGS:
        return "flywheel"
    if tags & WHEEL_TAGS:
        return "drive_wheel"
    if tags & SERVO_TAGS:
        return "servo"
    if tags & SENSOR_TAGS:
        return "sensor"
    if "motor" in tags:
        return "drive_motor"
    if tags & DRIVE_TAGS:
        return "gearbox"
    if tags & {"gear", "sprocket", "pulley"}:
        return "transmission"
    if "shaft" in tags:
        return "shaft"
    if tags & {"hub", "bearing", "collar"}:
        return "hub"
    if tags & {"control_hub", "battery", "electronics"}:
        return "electronics"
    if tags & STRUCTURE_TAGS:
        return "structure"
    return "structure"


def infer_connection_joint(
    connection: dict[str, Any],
    parent_part: dict[str, Any],
    child_part: dict[str, Any],
) -> str:
    explicit = connection.get("jointType")
    if explicit in {"fixed", "hinge", "slide"}:
        return str(explicit)
    parent_mount = part_mount(parent_part, str(connection["parent"]["mountId"]))
    child_mount = part_mount(child_part, str(connection["child"]["mountId"]))
    return inferred_joint_type(parent_mount, child_mount)


def _box_inertia_kg_m2(mass_kg: float, size_in: tuple[float, float, float]) -> np.ndarray:
    sx, sy, sz = (value * INCHES_TO_METERS for value in size_in)
    return np.array(
        [
            mass_kg / 12.0 * (sy * sy + sz * sz),
            mass_kg / 12.0 * (sx * sx + sz * sz),
            mass_kg / 12.0 * (sx * sx + sy * sy),
        ],
        dtype=np.float64,
    )


def _part_size_in(part: dict[str, Any], pose: np.ndarray) -> tuple[float, float, float]:
    boxes = part_aabbs(pose, part)
    combined = union_aabb(boxes)
    if combined is None:
        return (1.0, 1.0, 1.0)
    span = combined[1] - combined[0]
    return float(max(span[0], 1e-3)), float(max(span[1], 1e-3)), float(max(span[2], 1e-3))


def aggregate_mass_properties(
    instances: dict[str, dict[str, Any]],
    catalog_parts: dict[str, dict[str, Any]],
    poses: dict[str, np.ndarray] | None,
) -> MassProperties | None:
    if not poses:
        return None
    total_mass = 0.0
    moment = np.zeros(3, dtype=np.float64)
    for ident in instances:
        part = catalog_parts[ident]
        mass = float(part.get("massKg") or 0.0)
        if mass <= 0 or ident not in poses:
            continue
        total_mass += mass
        moment += mass * poses[ident][:3, 3]
    if total_mass <= 0:
        return None
    com = moment / total_mass
    inertia = np.zeros((3, 3), dtype=np.float64)
    for ident in instances:
        part = catalog_parts[ident]
        mass = float(part.get("massKg") or 0.0)
        if mass <= 0 or ident not in poses:
            continue
        pose = poses[ident]
        offset_m = (pose[:3, 3] - com) * INCHES_TO_METERS
        listed = list(part.get("inertiaKgM2") or [])
        if len(listed) == 3:
            local = np.diag([float(listed[0]), float(listed[1]), float(listed[2])])
            rotated = pose[:3, :3] @ local @ pose[:3, :3].T
        else:
            local = np.diag(_box_inertia_kg_m2(mass, _part_size_in(part, np.eye(4))))
            rotated = pose[:3, :3] @ local @ pose[:3, :3].T
        inertia += rotated + mass * (float(np.dot(offset_m, offset_m)) * np.eye(3) - np.outer(offset_m, offset_m))
    aabb = assembly_aabb(poses, {ident: catalog_parts[ident] for ident in instances if ident in poses})
    if aabb is None:
        lo = com.copy()
        hi = com.copy()
    else:
        lo, hi = aabb
    return MassProperties(
        mass_kg=float(total_mass),
        com_in=(float(com[0]), float(com[1]), float(com[2])),
        inertia_kg_m2=(float(inertia[0, 0]), float(inertia[1, 1]), float(inertia[2, 2])),
        aabb_min_in=(float(lo[0]), float(lo[1]), float(lo[2])),
        aabb_max_in=(float(hi[0]), float(hi[1]), float(hi[2])),
    )


def infer_mechanisms(roles: dict[str, str]) -> dict[str, tuple[str, ...]]:
    grouped: dict[str, list[str]] = {
        "drivetrain": [],
        "intake": [],
        "launcher": [],
        "sensors": [],
        "electronics": [],
    }
    for ident, role in roles.items():
        if role in {"drive_wheel", "drive_motor", "gearbox", "transmission", "shaft", "hub"}:
            grouped["drivetrain"].append(ident)
        elif role in {"intake"}:
            grouped["intake"].append(ident)
        elif role in {"servo", "flywheel", "launcher"}:
            grouped["launcher"].append(ident)
        elif role == "sensor":
            grouped["sensors"].append(ident)
        elif role == "electronics":
            grouped["electronics"].append(ident)
    return {key: tuple(values) for key, values in grouped.items() if values}


def infer_sensors(
    instances: dict[str, dict[str, Any]],
    catalog_parts: dict[str, dict[str, Any]],
    poses: dict[str, np.ndarray] | None,
) -> tuple[dict[str, Any], ...]:
    rows: list[dict[str, Any]] = []
    for ident, part in catalog_parts.items():
        if ident not in instances or not (part_tags(part) & SENSOR_TAGS):
            continue
        row: dict[str, Any] = {
            "instanceId": ident,
            "role": "sensor",
            "sku": str(part.get("sku") or ""),
            "kind": "unconfirmed",
        }
        if poses and ident in poses:
            origin = poses[ident][:3, 3]
            row["poseIn"] = {"x": float(origin[0]), "y": float(origin[1]), "z": float(origin[2])}
        rows.append(row)
    return tuple(rows)


def _confirmation_required(
    *,
    drivetrain: dict[str, Any],
    transmissions: tuple[TransmissionPath, ...],
    mechanisms: dict[str, tuple[str, ...]],
    sensors: tuple[dict[str, Any], ...],
    topology_fallback: tuple[str, ...],
) -> tuple[str, ...]:
    required: list[str] = ["instanceRoles"]
    if drivetrain:
        required.append("drivetrain")
    if transmissions:
        required.append("transmissions")
    if "intake" in mechanisms or "launcher" in mechanisms or topology_fallback:
        required.append("mechanisms")
    if sensors or "mechanismSensors" in topology_fallback:
        required.append("sensors")
    return tuple(required)


def bindings_are_complete(
    bindings: dict[str, Any],
    roles: dict[str, str],
    required: tuple[str, ...],
) -> bool:
    if not bindings.get("confirmed"):
        return False
    bound_roles = bindings.get("instanceRoles") if isinstance(bindings.get("instanceRoles"), dict) else {}
    if "instanceRoles" in required:
        if not bound_roles or any(ident not in bound_roles for ident in roles):
            return False
    if "drivetrain" in required:
        bound_drive = bindings.get("drivetrain") if isinstance(bindings.get("drivetrain"), dict) else {}
        if not bound_drive.get("type"):
            return False
    return True


def build_inference_report(
    instances: dict[str, dict[str, Any]],
    catalog_parts: dict[str, dict[str, Any]],
    connections: dict[str, dict[str, Any]],
    *,
    bindings: dict[str, Any] | None = None,
    poses: dict[str, np.ndarray] | None = None,
) -> InferenceReport:
    bindings = bindings if isinstance(bindings, dict) else {}
    roles = {ident: infer_role(catalog_parts[ident], bindings, ident) for ident in instances}
    joints: list[dict[str, str]] = []
    welded: list[str] = []
    articulated: list[str] = []
    for connection in connections.values():
        child_id = str(connection["child"]["instanceId"])
        parent_id = str(connection["parent"]["instanceId"])
        joint_type = infer_connection_joint(connection, catalog_parts[parent_id], catalog_parts[child_id])
        joints.append(
            {
                "connectionId": str(connection.get("id") or ""),
                "parentId": parent_id,
                "childId": child_id,
                "type": joint_type,
            }
        )
        if joint_type == "fixed":
            welded.append(child_id)
        else:
            articulated.append(child_id)
    transmissions = infer_transmissions(instances, catalog_parts, connections, poses=poses)
    drivetrain = infer_drivetrain(catalog_parts, transmissions, poses=poses)
    bound_drive = bindings.get("drivetrain") if isinstance(bindings.get("drivetrain"), dict) else None
    if bound_drive:
        drivetrain = {**drivetrain, **bound_drive}
    mechanisms = infer_mechanisms(roles)
    sensors = infer_sensors(instances, catalog_parts, poses)
    mass = aggregate_mass_properties(instances, catalog_parts, poses)
    topology_fallback = TOPOLOGY_FALLBACK_FIELDS if bindings.get("includeScoringTopology") else ()
    required = _confirmation_required(
        drivetrain=drivetrain,
        transmissions=transmissions,
        mechanisms=mechanisms,
        sensors=sensors,
        topology_fallback=topology_fallback,
    )
    confirmed = bindings_are_complete(bindings, roles, required)
    notes = []
    if topology_fallback:
        notes.append(TOPOLOGY_NOTE)
    if drivetrain.get("type"):
        notes.append(f"inferred drivetrain type={drivetrain['type']} from catalog wheels")
    if transmissions:
        notes.append("inferred wheel/motor transmission paths from rotating mounts")
    if not confirmed:
        notes.append("functional bindings are inferred and not confirmed")
        notes.append("unconfirmed inference; competitive physical_actuators remain disabled")
    if not any(part_tags(part) & {"control_hub", "battery"} for part in catalog_parts.values()):
        notes.append("optional electronics omitted; function is assumed")
    return InferenceReport(
        confirmed=confirmed,
        roles=roles,
        joints=tuple(joints),
        notes=tuple(notes),
        welded_instance_ids=tuple(welded),
        articulated_instance_ids=tuple(articulated),
        drivetrain=drivetrain,
        transmissions=transmissions,
        mechanisms=mechanisms,
        sensors=sensors,
        mass=mass,
        confirmation_required=() if confirmed else required,
        topology_fallback=topology_fallback,
    )
