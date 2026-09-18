"""Wheel/motor transmission paths, gear ratios, and drivetrain geometry."""

from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

import numpy as np

from talongym.robot.contract import AssemblyError
from talongym.robot.graph import undirected_adjacency
from talongym.robot.mounts import connection_is_rotating, mount_axis, part_mount
from talongym.robot.transforms import transform_direction, transform_point


def round_measure(value: float, digits: int = 3) -> float:
    return round(float(value), digits)


WHEEL_TAGS = frozenset({"wheel_mecanum", "wheel_traction", "wheel_omni", "wheel_compliant"})
GEAR_TAGS = frozenset({"gear", "sprocket", "pulley"})
MOTION_TAGS = frozenset({"shaft", "hub", "bearing", "collar", "gearbox", "motor"}) | GEAR_TAGS | WHEEL_TAGS
STRUCTURE_TAGS = frozenset({"channel", "extrusion", "plate", "bracket"})
_RATIO_RE = re.compile(r"(\d+(?:\.\d+)?)\s*:\s*1")
_RPM_RE = re.compile(r"(\d+(?:\.\d+)?)\s*RPM", re.I)
_TEETH_RE = re.compile(r"(\d+)\s*T\b", re.I)
_MESH_TOL_IN = 0.2
_AXIS_ALIGN = 0.998


@dataclass(frozen=True)
class MotorSpecs:
    gear_ratio: float | None
    free_speed_rpm: float | None


@dataclass(frozen=True)
class TransmissionPath:
    wheel_id: str
    motor_id: str | None
    instance_ids: tuple[str, ...]
    gear_ratio: float
    free_speed_rpm: float | None
    source: str


def part_tags(part: dict[str, Any]) -> set[str]:
    return {str(tag) for tag in (part.get("tags") or [])}


def motor_specs_from_part(part: dict[str, Any]) -> MotorSpecs:
    name = str(part.get("displayName") or "")
    ratio_match = _RATIO_RE.search(name)
    rpm_match = _RPM_RE.search(name)
    listed = part.get("gearRatio")
    if listed is not None:
        ratio = float(listed)
    else:
        ratio = float(ratio_match.group(1)) if ratio_match else None
    rpm = float(rpm_match.group(1)) if rpm_match else None
    return MotorSpecs(gear_ratio=ratio, free_speed_rpm=rpm)


def part_gear_ratio(part: dict[str, Any]) -> float:
    specs = motor_specs_from_part(part)
    if specs.gear_ratio and specs.gear_ratio > 0:
        return float(specs.gear_ratio)
    return 1.0


def tooth_count(part: dict[str, Any]) -> int | None:
    if not (part_tags(part) & GEAR_TAGS):
        return None
    name = str(part.get("displayName") or "")
    named = _TEETH_RE.search(name)
    if named:
        return int(named.group(1))
    sku = str(part.get("sku") or "")
    tail = sku.replace("_", "-").split("-")[-1]
    if tail.isdigit() and int(tail) >= 8:
        return int(tail)
    return None


def pitch_radius_in(part: dict[str, Any]) -> float:
    for proxy in part.get("collision") or []:
        if proxy.get("radiusIn"):
            return float(proxy["radiusIn"])
        size = list(proxy.get("sizeIn") or [])
        if len(size) >= 2:
            return 0.5 * max(float(size[0]), float(size[1]))
    return 0.0


def wheel_radius_in(part: dict[str, Any]) -> float | None:
    if not (part_tags(part) & WHEEL_TAGS):
        return None
    radius = pitch_radius_in(part)
    return radius if radius > 0 else None


def _primary_axis_world(part: dict[str, Any], pose: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mounts = list(part.get("mounts") or [])
    rotating = [row for row in mounts if str(row.get("kind") or "") in {"shaft", "hub", "bearing", "clamp"}]
    mount = rotating[0] if rotating else (mounts[0] if mounts else None)
    if mount is None:
        return pose[:3, 3].copy(), transform_direction(pose, np.array([1.0, 0.0, 0.0]))
    origin = transform_point(
        pose,
        np.array(
            [
                float((mount.get("transform") or {}).get("x") or 0.0),
                float((mount.get("transform") or {}).get("y") or 0.0),
                float((mount.get("transform") or {}).get("z") or 0.0),
            ]
        ),
    )
    axis = transform_direction(pose, mount_axis(mount))
    norm = float(np.linalg.norm(axis))
    return origin, (axis / norm if norm > 1e-9 else axis)


def gears_mesh(part_a: dict[str, Any], pose_a: np.ndarray, part_b: dict[str, Any], pose_b: np.ndarray) -> bool:
    if not ((part_tags(part_a) & GEAR_TAGS) and (part_tags(part_b) & GEAR_TAGS)):
        return False
    origin_a, axis_a = _primary_axis_world(part_a, pose_a)
    origin_b, axis_b = _primary_axis_world(part_b, pose_b)
    if abs(abs(float(np.dot(axis_a, axis_b))) - 1.0) > (1.0 - _AXIS_ALIGN):
        return False
    offset = origin_b - origin_a
    radial = offset - float(np.dot(offset, axis_a)) * axis_a
    distance = float(np.linalg.norm(radial))
    expected = pitch_radius_in(part_a) + pitch_radius_in(part_b)
    if expected <= 1e-6:
        return False
    return abs(distance - expected) <= _MESH_TOL_IN


def mesh_pairs(
    catalog_parts: dict[str, dict[str, Any]],
    poses: dict[str, np.ndarray] | None,
) -> tuple[tuple[str, str], ...]:
    if not poses:
        return ()
    gear_ids = [ident for ident, part in catalog_parts.items() if part_tags(part) & GEAR_TAGS and ident in poses]
    pairs: list[tuple[str, str]] = []
    for index, left in enumerate(gear_ids):
        for right in gear_ids[index + 1 :]:
            if gears_mesh(catalog_parts[left], poses[left], catalog_parts[right], poses[right]):
                pairs.append((left, right))
    return tuple(pairs)


def _connection_mounts(
    connection: dict[str, Any],
    catalog_parts: dict[str, dict[str, Any]],
) -> tuple[dict[str, Any], dict[str, Any]] | None:
    parent_id = str(connection["parent"]["instanceId"])
    child_id = str(connection["child"]["instanceId"])
    try:
        parent_mount = part_mount(catalog_parts[parent_id], str(connection["parent"]["mountId"]))
        child_mount = part_mount(catalog_parts[child_id], str(connection["child"]["mountId"]))
    except (KeyError, AssemblyError):
        return None
    return parent_mount, child_mount


def _is_motion_edge(
    ident: str,
    other_id: str,
    connection: dict[str, Any],
    catalog_parts: dict[str, dict[str, Any]],
) -> bool:
    mounts = _connection_mounts(connection, catalog_parts)
    if mounts and connection_is_rotating(*mounts):
        return True
    other = catalog_parts.get(other_id) or {}
    current = catalog_parts.get(ident) or {}
    other_tags = part_tags(other)
    current_tags = part_tags(current)
    if other_tags & MOTION_TAGS and current_tags & MOTION_TAGS:
        return True
    if other_tags <= STRUCTURE_TAGS and not (other_tags & MOTION_TAGS):
        return False
    return bool(other_tags & MOTION_TAGS)


def _mesh_adjacency(pairs: Iterable[tuple[str, str]]) -> dict[str, list[str]]:
    adjacency: dict[str, list[str]] = {}
    for left, right in pairs:
        adjacency.setdefault(left, []).append(right)
        adjacency.setdefault(right, []).append(left)
    return adjacency


def _ratio_between_gears(driver: dict[str, Any], driven: dict[str, Any]) -> float:
    driver_teeth = tooth_count(driver)
    driven_teeth = tooth_count(driven)
    if driver_teeth and driven_teeth:
        return float(driven_teeth) / float(driver_teeth)
    driver_r = pitch_radius_in(driver)
    driven_r = pitch_radius_in(driven)
    if driver_r > 1e-6 and driven_r > 1e-6:
        return driven_r / driver_r
    return 1.0


def infer_transmissions(
    instances: dict[str, dict[str, Any]],
    catalog_parts: dict[str, dict[str, Any]],
    connections: dict[str, dict[str, Any]] | list[dict[str, Any]],
    *,
    poses: dict[str, np.ndarray] | None = None,
) -> tuple[TransmissionPath, ...]:
    adjacency = undirected_adjacency(connections)
    meshes = _mesh_adjacency(mesh_pairs(catalog_parts, poses))
    wheels = [ident for ident, part in catalog_parts.items() if ident in instances and part_tags(part) & WHEEL_TAGS]
    paths: list[TransmissionPath] = []
    for wheel_id in wheels:
        queue: list[tuple[str, tuple[str, ...]]] = [(wheel_id, (wheel_id,))]
        seen = {wheel_id}
        motor_id: str | None = None
        motor_path: tuple[str, ...] = (wheel_id,)
        while queue:
            ident, path = queue.pop(0)
            if part_tags(catalog_parts[ident]) & {"motor"} and ident != wheel_id:
                motor_id = ident
                motor_path = path
                break
            for other_id, connection in adjacency.get(ident, []):
                if other_id in seen or other_id not in catalog_parts:
                    continue
                if not _is_motion_edge(ident, other_id, connection, catalog_parts):
                    continue
                seen.add(other_id)
                queue.append((other_id, path + (other_id,)))
            for other_id in meshes.get(ident, []):
                if other_id in seen or other_id not in catalog_parts:
                    continue
                seen.add(other_id)
                queue.append((other_id, path + (other_id,)))
        extra = 1.0
        mesh_used = False
        stack_used = False
        if motor_id:
            for index, ident in enumerate(motor_path[:-1]):
                nxt = motor_path[index + 1]
                if nxt in meshes.get(ident, []):
                    extra *= _ratio_between_gears(catalog_parts[nxt], catalog_parts[ident])
                    mesh_used = True
            for ident in motor_path:
                if ident == motor_id:
                    continue
                if part_tags(catalog_parts[ident]) & {"gearbox"}:
                    stage = part_gear_ratio(catalog_parts[ident])
                    if abs(stage - 1.0) > 1e-9:
                        extra *= stage
                        stack_used = True
            specs = motor_specs_from_part(catalog_parts[motor_id])
            internal = part_gear_ratio(catalog_parts[motor_id])
            total = internal * extra
            rpm = specs.free_speed_rpm
            if rpm is not None and extra > 1e-9:
                rpm = float(rpm) / extra
            source = "gear_mesh" if mesh_used else ("gearbox_stack" if stack_used else "shaft")
            paths.append(
                TransmissionPath(
                    wheel_id=wheel_id,
                    motor_id=motor_id,
                    instance_ids=motor_path,
                    gear_ratio=total,
                    free_speed_rpm=rpm,
                    source=source,
                )
            )
        else:
            paths.append(
                TransmissionPath(
                    wheel_id=wheel_id,
                    motor_id=None,
                    instance_ids=(wheel_id,),
                    gear_ratio=1.0,
                    free_speed_rpm=None,
                    source="unconfirmed",
                )
            )
    return tuple(paths)


def infer_drivetrain(
    catalog_parts: dict[str, dict[str, Any]],
    transmissions: tuple[TransmissionPath, ...],
    poses: dict[str, np.ndarray] | None = None,
) -> dict[str, Any]:
    wheels = [ident for ident, part in catalog_parts.items() if part_tags(part) & WHEEL_TAGS]
    mecanum = [ident for ident in wheels if "wheel_mecanum" in part_tags(catalog_parts[ident])]
    tankish = [ident for ident in wheels if part_tags(catalog_parts[ident]) & {"wheel_traction", "wheel_omni"}]
    drive: dict[str, Any] = {}
    if mecanum and not tankish:
        drive["type"] = "mecanum"
        drive["strafeMultiplier"] = 1.0
    elif tankish and not mecanum:
        drive["type"] = "tank"
        drive["strafeMultiplier"] = 0.0
    diameters = [2.0 * float(wheel_radius_in(catalog_parts[ident]) or 0.0) for ident in wheels]
    diameters = [value for value in diameters if value > 0]
    if diameters:
        drive["wheelDiameterIn"] = round_measure(sum(diameters) / len(diameters))
    if poses:
        left = [ident for ident in wheels if ident in poses and ident.endswith(("_fl", "_rl"))]
        right = [ident for ident in wheels if ident in poses and ident.endswith(("_fr", "_rr"))]
        placed = [ident for ident in wheels if ident in poses]
        if left and right:
            left_y = float(np.mean([poses[ident][1, 3] for ident in left]))
            right_y = float(np.mean([poses[ident][1, 3] for ident in right]))
            drive["trackWidthIn"] = round_measure(abs(left_y - right_y))
            xs = [float(poses[ident][0, 3]) for ident in left + right]
            drive["wheelbaseIn"] = round_measure(abs(max(xs) - min(xs)))
        elif len(placed) >= 2:
            points = np.vstack([poses[ident][:3, 3] for ident in placed])
            drive["trackWidthIn"] = round_measure(float(points[:, 1].max() - points[:, 1].min()))
            drive["wheelbaseIn"] = round_measure(float(points[:, 0].max() - points[:, 0].min()))
    drive["complete"] = (drive.get("type") == "mecanum" and len(mecanum) >= 4) or (
        drive.get("type") == "tank" and len(tankish) >= 2
    )
    motor_ids = {path.motor_id for path in transmissions if path.motor_id}
    drive["motorCount"] = len(motor_ids)
    if transmissions:
        drive["gearRatio"] = round_measure(transmissions[0].gear_ratio)
        rpms = [path.free_speed_rpm for path in transmissions if path.free_speed_rpm]
        if rpms:
            drive["freeSpeedRpm"] = round_measure(rpms[0])
    return drive
