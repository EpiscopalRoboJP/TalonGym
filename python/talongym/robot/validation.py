"""Hard physical validity and soft FTC envelope/mass/season warnings."""

from __future__ import annotations

from typing import Any

import numpy as np

from talongym.robot.collision import aabbs_overlap, part_aabbs
from talongym.robot.contract import AssemblyError
from talongym.robot.graph import subtree_ids
from talongym.robot.inference import FLY_TAGS, INTAKE_TAGS, SERVO_TAGS, InferenceReport
from talongym.robot.mounts import connection_is_rotating, hole_in_part, mount_axis, part_mount
from talongym.robot.transforms import axis_angle, transform_direction, transform_point
from talongym.robot.transmissions import WHEEL_TAGS, part_tags, wheel_radius_in

FTC_STARTING_ENVELOPE_IN = 18.0
FTC_MASS_KG = 42.0 * 0.45359237
COAXIAL_COS_MIN = 0.998
COAXIAL_OFFSET_IN = 0.05
WHEEL_PLANE_TOL_IN = 0.25
FLOOR_STRUCTURE_TOL_IN = 0.1
CONTINUOUS_TRAVEL = 1.0e5
TRAVEL_SAMPLES = 5


def _connection_rows(
    connections: dict[str, dict[str, Any]] | list[dict[str, Any]],
) -> list[dict[str, Any]]:
    return list(connections.values() if isinstance(connections, dict) else connections)


def _pattern_index(ref: dict[str, Any]) -> tuple[int, int] | None:
    index = ref.get("patternIndex")
    if not index:
        return None
    return int(index[0]), int(index[1])


def _normalize(vector: np.ndarray) -> np.ndarray:
    vector = np.asarray(vector, dtype=np.float64).reshape(3)
    norm = float(np.linalg.norm(vector))
    if norm < 1e-12:
        raise AssemblyError("mount axis must be a non-zero vector")
    return vector / norm


def _world_mount(
    pose: np.ndarray,
    part: dict[str, Any],
    ref: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray]:
    mount = part_mount(part, str(ref["mountId"]))
    origin = transform_point(pose, hole_in_part(mount, _pattern_index(ref)))
    axis = _normalize(transform_direction(pose, mount_axis(mount)))
    return origin, axis


def validate_coaxiality(
    poses: dict[str, np.ndarray],
    catalog_parts: dict[str, dict[str, Any]],
    connections: dict[str, dict[str, Any]] | list[dict[str, Any]],
) -> None:
    for connection in _connection_rows(connections):
        parent_id = str(connection["parent"]["instanceId"])
        child_id = str(connection["child"]["instanceId"])
        if parent_id not in poses or child_id not in poses:
            continue
        parent_part = catalog_parts[parent_id]
        child_part = catalog_parts[child_id]
        parent_mount = part_mount(parent_part, str(connection["parent"]["mountId"]))
        child_mount = part_mount(child_part, str(connection["child"]["mountId"]))
        if not connection_is_rotating(parent_mount, child_mount):
            continue
        parent_origin, parent_axis = _world_mount(poses[parent_id], parent_part, connection["parent"])
        child_origin, child_axis = _world_mount(poses[child_id], child_part, connection["child"])
        if abs(abs(float(np.dot(parent_axis, child_axis))) - 1.0) > (1.0 - COAXIAL_COS_MIN):
            raise AssemblyError(f"shaft/bearing axes are not coaxial between {parent_id} and {child_id}")
        offset = child_origin - parent_origin
        miss = float(np.linalg.norm(np.cross(offset, parent_axis)))
        if miss > COAXIAL_OFFSET_IN:
            raise AssemblyError(f"shaft/bearing axes are not coaxial between {parent_id} and {child_id}")


def _rotate_about(world: np.ndarray, origin: np.ndarray, axis: np.ndarray, theta_rad: float) -> np.ndarray:
    rotation = axis_angle(axis, theta_rad)
    matrix = np.eye(4, dtype=np.float64)
    matrix[:3, :3] = rotation
    matrix[:3, 3] = origin - rotation @ origin
    return matrix @ world


def validate_travel_envelopes(
    poses: dict[str, np.ndarray],
    instances: dict[str, dict[str, Any]],
    catalog_parts: dict[str, dict[str, Any]],
    connections: dict[str, dict[str, Any]],
) -> None:
    for child_id, connection in connections.items():
        explicit = connection.get("jointType")
        parent_id = str(connection["parent"]["instanceId"])
        parent_part = catalog_parts[parent_id]
        child_part = catalog_parts[child_id]
        joint_type = (
            explicit
            if explicit in {"fixed", "hinge", "slide"}
            else inferred_joint_name(connection, parent_part, child_part)
        )
        limit = connection.get("limit") or connection.get("travelLimit")
        if joint_type not in {"hinge", "slide"} or not isinstance(limit, (list, tuple)) or len(limit) != 2:
            continue
        lo, hi = float(limit[0]), float(limit[1])
        if hi - lo >= CONTINUOUS_TRAVEL:
            continue
        parent_origin, parent_axis = _world_mount(poses[parent_id], parent_part, connection["parent"])
        moving = set(subtree_ids(child_id, connections))
        static = [ident for ident in instances if ident not in moving]
        for theta_deg in np.linspace(lo, hi, TRAVEL_SAMPLES):
            theta = float(np.radians(theta_deg if joint_type == "hinge" else 0.0))
            offset = parent_axis * float(theta_deg if joint_type == "slide" else 0.0)
            moved: dict[str, np.ndarray] = {}
            for ident in moving:
                pose = poses[ident]
                if joint_type == "hinge":
                    moved[ident] = _rotate_about(pose, parent_origin, parent_axis, theta)
                else:
                    shifted = pose.copy()
                    shifted[:3, 3] = pose[:3, 3] + offset
                    moved[ident] = shifted
            for ident in moving:
                for other_id in static:
                    overlapping = any(
                        aabbs_overlap(left, right)
                        for left in part_aabbs(moved[ident], catalog_parts[ident])
                        for right in part_aabbs(poses[other_id], catalog_parts[other_id])
                    )
                    if overlapping:
                        raise AssemblyError(
                            f"mechanism travel self-intersects between {ident} and {other_id} at {theta_deg:.1f}"
                        )


def inferred_joint_name(
    connection: dict[str, Any],
    parent_part: dict[str, Any],
    child_part: dict[str, Any],
) -> str:
    from talongym.robot.inference import infer_connection_joint

    return infer_connection_joint(connection, parent_part, child_part)


def validate_wheel_floor_contact(
    poses: dict[str, np.ndarray],
    catalog_parts: dict[str, dict[str, Any]],
    report: InferenceReport,
) -> None:
    if not report.drivetrain.get("complete"):
        return
    bottoms: list[float] = []
    for ident, part in catalog_parts.items():
        if ident not in poses or not (part_tags(part) & WHEEL_TAGS):
            continue
        radius = float(wheel_radius_in(part) or 0.0)
        boxes = part_aabbs(poses[ident], part)
        if boxes:
            bottoms.append(float(min(box[0][2] for box in boxes)))
        else:
            bottoms.append(float(poses[ident][2, 3] - radius))
    if len(bottoms) < 2:
        return
    if max(bottoms) - min(bottoms) > WHEEL_PLANE_TOL_IN:
        raise AssemblyError("drive wheels do not share a common floor contact plane")
    floor = min(bottoms)
    skip_tags = WHEEL_TAGS | INTAKE_TAGS | FLY_TAGS | SERVO_TAGS | {"control_hub", "battery", "electronics", "motor", "gearbox"}
    for ident, part in catalog_parts.items():
        if ident not in poses or part_tags(part) & skip_tags:
            continue
        boxes = part_aabbs(poses[ident], part)
        if any(float(box[0][2]) < floor - FLOOR_STRUCTURE_TOL_IN for box in boxes):
            raise AssemblyError(f"structure {ident} extends below drive-wheel floor contact")


def validate_path_constraints(
    poses: dict[str, np.ndarray],
    catalog_parts: dict[str, dict[str, Any]],
    report: InferenceReport,
) -> None:
    intake_ids = report.mechanisms.get("intake") or ()
    if not intake_ids:
        return
    occupied = {ident: part_aabbs(poses[ident], catalog_parts[ident]) for ident in catalog_parts if ident in poses}
    for ident in intake_ids:
        origin = poses[ident][:3, 3]
        volume = (
            origin + np.array([0.25, -3.0, -1.5], dtype=np.float64),
            origin + np.array([5.0, 3.0, 1.5], dtype=np.float64),
        )
        blockers = [
            other
            for other, boxes in occupied.items()
            if other != ident and any(aabbs_overlap(volume, box) for box in boxes)
        ]
        drivetrain_ids = report.mechanisms.get("drivetrain") or ()
        if len(blockers) >= 2 and all(blocker not in drivetrain_ids for blocker in blockers):
            raise AssemblyError(f"intake path at {ident} is blocked by {sorted(blockers)}")


def ftc_soft_warnings(
    catalog_parts: dict[str, dict[str, Any]],
    report: InferenceReport,
) -> tuple[dict[str, Any], ...]:
    warnings: list[dict[str, Any]] = []
    if report.mass is not None:
        span = np.array(report.mass.aabb_max_in) - np.array(report.mass.aabb_min_in)
        if float(np.max(span)) > FTC_STARTING_ENVELOPE_IN + 1e-6:
            warnings.append(
                {
                    "code": "ftc_starting_envelope",
                    "severity": "warning",
                    "message": (
                        f"assembled AABB {span[0]:.2f}x{span[1]:.2f}x{span[2]:.2f} in exceeds the "
                        f"{FTC_STARTING_ENVELOPE_IN:.0f}-inch FTC starting envelope"
                    ),
                }
            )
        if report.mass.mass_kg > FTC_MASS_KG:
            warnings.append(
                {
                    "code": "ftc_mass",
                    "severity": "warning",
                    "message": f"catalog mass {report.mass.mass_kg:.2f} kg exceeds the 42 lb FTC limit",
                }
            )
    if report.topology_fallback:
        warnings.append(
            {
                "code": "ftc_season_parity",
                "severity": "warning",
                "message": (
                    "season mechanism/piece-path legality uses mecanum_biobuzz_4cap parity; "
                    "confirm against the active FTC season"
                ),
            }
        )
    has_hub = any("control_hub" in part_tags(part) for part in catalog_parts.values())
    has_battery = any("battery" in part_tags(part) for part in catalog_parts.values())
    if not has_hub or not has_battery:
        warnings.append(
            {
                "code": "optional_electronics",
                "severity": "warning",
                "message": (
                    "control hub/battery models are optional; electronics function is assumed and "
                    "only geometry/mass would change if they are added"
                ),
            }
        )
    if report.drivetrain and not report.drivetrain.get("complete"):
        warnings.append(
            {
                "code": "incomplete_drivetrain",
                "severity": "info",
                "message": "inferred drivetrain is incomplete; wheel/motor bindings still require confirmation",
            }
        )
    return tuple(warnings)


def validate_inferred_assembly(
    poses: dict[str, np.ndarray],
    instances: dict[str, dict[str, Any]],
    catalog_parts: dict[str, dict[str, Any]],
    connections: dict[str, dict[str, Any]] | list[dict[str, Any]],
    report: InferenceReport,
) -> tuple[dict[str, Any], ...]:
    rows = (
        connections if isinstance(connections, dict) else {str(row["child"]["instanceId"]): row for row in connections}
    )
    validate_coaxiality(poses, catalog_parts, rows)
    validate_travel_envelopes(poses, instances, catalog_parts, rows)
    validate_wheel_floor_contact(poses, catalog_parts, report)
    validate_path_constraints(poses, catalog_parts, report)
    return ftc_soft_warnings(catalog_parts, report)
