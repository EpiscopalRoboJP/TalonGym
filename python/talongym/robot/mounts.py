"""Exact mount compatibility, hole-pattern occupancy, and local hole frames."""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from talongym.robot.catalog import MM_TO_IN
from talongym.robot.contract import AssemblyError
from talongym.robot.transforms import plane_basis

COMPATIBLE_KIND_PAIRS = frozenset(
    {
        frozenset({"threaded_hole", "clearance_hole"}),
        frozenset({"clearance_hole"}),
        frozenset({"clearance_hole", "mating_face"}),
        frozenset({"shaft", "hub"}),
        frozenset({"shaft", "bearing"}),
        frozenset({"shaft", "clamp"}),
        frozenset({"mating_face"}),
    }
)
ROTATING_KIND_PAIRS = frozenset(
    {
        frozenset({"shaft", "hub"}),
        frozenset({"shaft", "bearing"}),
        frozenset({"shaft", "clamp"}),
    }
)
HOLE_KINDS = frozenset({"threaded_hole", "clearance_hole"})
DIAMETER_TOL_MM = 0.2
# M3 close-fit is ~3.4 mm, normal ~3.6 mm; M4 clearance is ~4.5 mm over a 4.0 mm thread.
MAX_CLEARANCE_OVER_THREAD_MM = 1.5
ADAPTER_STANDARD = "adapter_required"
HARDWARE_FAMILIES = {
    "m3_screw": "m3",
    "m3_t_nut": "m3",
    "m4_screw": "m4",
    "m4_t_nut": "m4",
}
STANDARD_FAMILIES = {
    "rev_m3": "m3",
    "rev_15mm": "m3",
    "gobilda_pattern": "m4",
}


def part_mount(part: dict[str, Any], mount_id: str) -> dict[str, Any]:
    sku = str(part.get("sku") or part.get("id") or "part")
    for mount in part.get("mounts") or []:
        if str(mount.get("id") or "") == mount_id:
            return mount
    raise AssemblyError(f"part {sku} has no mount {mount_id}")


def mount_axis(mount: dict[str, Any]) -> np.ndarray:
    axis = np.asarray(mount.get("axis") or [0.0, 0.0, 1.0], dtype=np.float64)
    return axis / float(np.linalg.norm(axis))


def _pattern_counts(pattern: dict[str, Any] | None) -> tuple[int, int]:
    row = pattern or {}
    count_u = int(row.get("countU") or row.get("count") or 1)
    count_v = int(row.get("countV") or 1)
    if str(row.get("type") or "single") == "single":
        return 1, 1
    if str(row.get("type") or "") == "linear":
        return max(count_u, 1), 1
    if str(row.get("type") or "") == "circle":
        return max(int(row.get("count") or count_u), 1), 1
    return max(count_u, 1), max(count_v, 1)


def _pattern_offset_uv(pattern: dict[str, Any] | None, index: tuple[int, int] | None) -> np.ndarray:
    if index is None or not pattern:
        return np.zeros(2, dtype=np.float64)
    kind = str(pattern.get("type") or "single")
    count_u, count_v = _pattern_counts(pattern)
    u_i, v_i = int(index[0]), int(index[1])
    if u_i < 0 or v_i < 0 or u_i >= count_u or v_i >= count_v:
        raise AssemblyError(f"pattern index {list(index)} is outside {kind} {count_u}x{count_v}")
    pitch = float(pattern.get("pitchMm") or 0.0) * MM_TO_IN
    if kind == "single":
        return np.zeros(2, dtype=np.float64)
    if kind == "circle":
        angle = 2.0 * math.pi * u_i / float(count_u)
        return np.array([pitch * math.cos(angle), pitch * math.sin(angle)], dtype=np.float64)
    du = (u_i - (count_u - 1) / 2.0) * pitch
    dv = 0.0 if kind == "linear" else (v_i - (count_v - 1) / 2.0) * pitch
    return np.array([du, dv], dtype=np.float64)


def pattern_basis(mount: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    axis = mount_axis(mount)
    listed = mount.get("uAxis")
    if listed is None:
        return plane_basis(axis)
    raw = np.array([float(listed[0]), float(listed[1]), float(listed[2])], dtype=np.float64)
    projected = raw - float(np.dot(raw, axis)) * axis
    norm = float(np.linalg.norm(projected))
    if norm < 1e-9:
        raise AssemblyError("mount uAxis must be non-zero and not parallel to the mount axis")
    u_axis = projected / norm
    return u_axis, np.cross(axis, u_axis)


def hole_in_part(mount: dict[str, Any], index: tuple[int, int] | None) -> np.ndarray:
    pose = mount.get("transform") or {}
    origin = np.array(
        [float(pose.get("x") or 0.0), float(pose.get("y") or 0.0), float(pose.get("z") or 0.0)],
        dtype=np.float64,
    )
    u_axis, v_axis = pattern_basis(mount)
    offset = _pattern_offset_uv(mount.get("pattern") if isinstance(mount.get("pattern"), dict) else None, index)
    return origin + u_axis * float(offset[0]) + v_axis * float(offset[1])


def connection_is_rotating(parent_mount: dict[str, Any], child_mount: dict[str, Any]) -> bool:
    pair = frozenset({str(parent_mount.get("kind") or ""), str(child_mount.get("kind") or "")})
    return pair in ROTATING_KIND_PAIRS


def inferred_joint_type(parent_mount: dict[str, Any], child_mount: dict[str, Any]) -> str:
    if connection_is_rotating(parent_mount, child_mount):
        return "hinge"
    return "fixed"


def _kinds_compatible(parent_kind: str, child_kind: str) -> bool:
    return frozenset({parent_kind, child_kind}) in COMPATIBLE_KIND_PAIRS


def hardware_families(mount: dict[str, Any]) -> set[str]:
    families: set[str] = set()
    for item in mount.get("allowedHardware") or []:
        family = HARDWARE_FAMILIES.get(str(item).lower())
        if family:
            families.add(family)
    standard = str(mount.get("standard") or "")
    family = STANDARD_FAMILIES.get(standard)
    if family:
        families.add(family)
    return families


def _standards_compatible(parent_mount: dict[str, Any], child_mount: dict[str, Any]) -> bool:
    parent_standard = str(parent_mount.get("standard") or "")
    child_standard = str(child_mount.get("standard") or "")
    if parent_standard == child_standard:
        return True
    if ADAPTER_STANDARD in {parent_standard, child_standard}:
        return True
    kinds = {str(parent_mount.get("kind") or ""), str(child_mount.get("kind") or "")}
    if kinds == {"clearance_hole", "mating_face"} and hardware_families(parent_mount) & hardware_families(child_mount):
        return True
    if kinds == {"threaded_hole", "clearance_hole"} and hardware_families(parent_mount) & hardware_families(child_mount):
        return True
    return False


def _thread_and_clearance(parent_mount: dict[str, Any], child_mount: dict[str, Any]) -> tuple[float, float] | None:
    parent_kind = str(parent_mount.get("kind") or "")
    child_kind = str(child_mount.get("kind") or "")
    if {parent_kind, child_kind} != {"threaded_hole", "clearance_hole"}:
        return None
    parent_d = parent_mount.get("diameterMm")
    child_d = child_mount.get("diameterMm")
    if parent_d is None or child_d is None:
        return None
    if parent_kind == "threaded_hole":
        return float(parent_d), float(child_d)
    return float(child_d), float(parent_d)


def _diameters_compatible(parent_mount: dict[str, Any], child_mount: dict[str, Any]) -> bool:
    parent = parent_mount.get("diameterMm")
    child = child_mount.get("diameterMm")
    if parent is None or child is None:
        return True
    pair = _thread_and_clearance(parent_mount, child_mount)
    if pair is not None:
        thread_d, clear_d = pair
        if abs(thread_d - clear_d) <= DIAMETER_TOL_MM:
            return True
        over = clear_d - thread_d
        return 0.0 < over <= MAX_CLEARANCE_OVER_THREAD_MM
    return abs(float(parent) - float(child)) <= DIAMETER_TOL_MM


def mounts_compatible(
    parent_part: dict[str, Any],
    parent_mount: dict[str, Any],
    child_part: dict[str, Any],
    child_mount: dict[str, Any],
) -> None:
    parent_kind = str(parent_mount.get("kind") or "")
    child_kind = str(child_mount.get("kind") or "")
    if not _kinds_compatible(parent_kind, child_kind):
        raise AssemblyError(f"incompatible mount kinds {parent_kind} and {child_kind}")
    parent_standard = str(parent_mount.get("standard") or "")
    child_standard = str(child_mount.get("standard") or "")
    if not _standards_compatible(parent_mount, child_mount):
        raise AssemblyError(f"mismatched mount standards {parent_standard} and {child_standard}")
    parent_brand = str(parent_part.get("manufacturer") or "")
    child_brand = str(child_part.get("manufacturer") or "")
    if parent_brand and child_brand and parent_brand != child_brand:
        if ADAPTER_STANDARD not in {parent_standard, child_standard}:
            raise AssemblyError("cross-brand connection requires a catalog adapter")
    if {parent_kind, child_kind} <= HOLE_KINDS or {parent_kind, child_kind} == {"clearance_hole", "mating_face"}:
        families = hardware_families(parent_mount) & hardware_families(child_mount)
        if hardware_families(parent_mount) and hardware_families(child_mount) and not families:
            raise AssemblyError(
                f"incompatible hardware {parent_mount.get('allowedHardware')} and {child_mount.get('allowedHardware')}"
            )
    if not _diameters_compatible(parent_mount, child_mount):
        raise AssemblyError(
            f"incompatible mount diameters {parent_mount.get('diameterMm')} mm and {child_mount.get('diameterMm')} mm"
        )


def _occupancy_key(instance_id: str, mount_id: str, index: tuple[int, int] | None) -> tuple[str, str, int | None, int | None]:
    if index is None:
        return (instance_id, mount_id, None, None)
    return (instance_id, mount_id, int(index[0]), int(index[1]))


def _pattern_index(ref: dict[str, Any]) -> tuple[int, int] | None:
    index = ref.get("patternIndex")
    if not index:
        return None
    return int(index[0]), int(index[1])


def _conflicts(existing: set[tuple[str, str, int | None, int | None]], key: tuple[str, str, int | None, int | None]) -> bool:
    instance_id, mount_id, u_i, v_i = key
    if (instance_id, mount_id, None, None) in existing:
        return True
    if u_i is None and v_i is None:
        return any(item[0] == instance_id and item[1] == mount_id for item in existing)
    return key in existing


def _claim(
    occupied: set[tuple[str, str, int | None, int | None]],
    ref: dict[str, Any],
    connection_id: str,
) -> None:
    key = _occupancy_key(str(ref["instanceId"]), str(ref["mountId"]), _pattern_index(ref))
    if _conflicts(occupied, key):
        raise AssemblyError(
            f"duplicate occupancy of {key[0]}.{key[1]}"
            + ("" if key[2] is None else f" hole {list(key[2:])}")
            + f" by connection {connection_id}"
        )
    occupied.add(key)


def validate_connection_mounts(
    connection: dict[str, Any],
    parent_part: dict[str, Any],
    child_part: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    parent_ref = connection["parent"]
    child_ref = connection["child"]
    parent_mount = part_mount(parent_part, str(parent_ref["mountId"]))
    child_mount = part_mount(child_part, str(child_ref["mountId"]))
    mounts_compatible(parent_part, parent_mount, child_part, child_mount)
    secondary = connection.get("secondary")
    if secondary:
        parent_secondary_ref = secondary["parent"]
        child_secondary_ref = secondary["child"]
        if str(parent_secondary_ref.get("instanceId") or "") != str(parent_ref["instanceId"]):
            raise AssemblyError(f"connection {connection.get('id')} secondary parent must use the parent instance")
        if str(child_secondary_ref.get("instanceId") or "") != str(child_ref["instanceId"]):
            raise AssemblyError(f"connection {connection.get('id')} secondary child must use the child instance")
        part_mount(parent_part, str(parent_secondary_ref["mountId"]))
        part_mount(child_part, str(child_secondary_ref["mountId"]))
    return parent_mount, child_mount


def validate_occupancy(connections: list[dict[str, Any]]) -> None:
    occupied: set[tuple[str, str, int | None, int | None]] = set()
    for connection in connections:
        ident = str(connection.get("id") or "")
        _claim(occupied, connection["parent"], ident)
        _claim(occupied, connection["child"], ident)
        secondary = connection.get("secondary")
        if secondary:
            _claim(occupied, secondary["parent"], ident)
            _claim(occupied, secondary["child"], ident)
