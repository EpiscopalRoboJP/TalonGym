"""Rigid transforms and exact mount-to-mount snap solving."""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from talongym.robot.contract import AssemblyError

_AXIS_EPS = 1e-9
_ALIGN_EPS = 1e-8
PATTERN_MATCH_TOL_IN = 0.02  # ~0.5 mm


def _normalize(vector: np.ndarray) -> np.ndarray:
    vector = np.asarray(vector, dtype=np.float64).reshape(3)
    norm = float(np.linalg.norm(vector))
    if norm < _AXIS_EPS:
        raise AssemblyError("mount axis must be a non-zero vector")
    return vector / norm


def _skew(vector: np.ndarray) -> np.ndarray:
    x, y, z = (float(vector[0]), float(vector[1]), float(vector[2]))
    return np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]], dtype=np.float64)


def plane_basis(axis: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    normal = _normalize(axis)
    helper = np.array([0.0, 0.0, 1.0]) if abs(float(normal[2])) < 0.9 else np.array([1.0, 0.0, 0.0])
    u_axis = np.cross(helper, normal)
    u_axis = u_axis / float(np.linalg.norm(u_axis))
    v_axis = np.cross(normal, u_axis)
    return u_axis, v_axis


def euler_matrix(roll_rad: float, pitch_rad: float, yaw_rad: float) -> np.ndarray:
    """FTC Rz(yaw) * Ry(pitch) * Rx(roll), matching mjcf_field._robot_quat."""
    cr, sr = math.cos(roll_rad), math.sin(roll_rad)
    cp, sp = math.cos(pitch_rad), math.sin(pitch_rad)
    cy, sy = math.cos(yaw_rad), math.sin(yaw_rad)
    return np.array(
        [
            [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
            [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
            [-sp, cp * sr, cp * cr],
        ],
        dtype=np.float64,
    )


def matrix_from_pose(pose: dict[str, Any] | None) -> np.ndarray:
    row = pose or {}
    matrix = np.eye(4, dtype=np.float64)
    matrix[:3, :3] = euler_matrix(
        math.radians(float(row.get("rollDeg") or 0.0)),
        math.radians(float(row.get("pitchDeg") or 0.0)),
        math.radians(float(row.get("yawDeg") or 0.0)),
    )
    matrix[0, 3] = float(row.get("x") or 0.0)
    matrix[1, 3] = float(row.get("y") or 0.0)
    matrix[2, 3] = float(row.get("z") or 0.0)
    return matrix


def matrix_to_euler(rotation: np.ndarray) -> tuple[float, float, float]:
    pitch = math.asin(float(np.clip(-rotation[2, 0], -1.0, 1.0)))
    cosine_pitch = math.cos(pitch)
    if abs(cosine_pitch) > 1e-8:
        yaw = math.atan2(float(rotation[1, 0]), float(rotation[0, 0]))
        roll = math.atan2(float(rotation[2, 1]), float(rotation[2, 2]))
    else:
        yaw = math.atan2(float(-rotation[0, 1]), float(rotation[1, 1]))
        roll = 0.0
    return roll, pitch, yaw


def pose_from_matrix(matrix: np.ndarray) -> dict[str, float]:
    roll, pitch, yaw = matrix_to_euler(matrix[:3, :3])
    return {
        "x": float(matrix[0, 3]),
        "y": float(matrix[1, 3]),
        "z": float(matrix[2, 3]),
        "rollDeg": math.degrees(roll),
        "pitchDeg": math.degrees(pitch),
        "yawDeg": math.degrees(yaw),
    }


def invert_transform(matrix: np.ndarray) -> np.ndarray:
    rotation = matrix[:3, :3]
    inverse = np.eye(4, dtype=np.float64)
    inverse[:3, :3] = rotation.T
    inverse[:3, 3] = -rotation.T @ matrix[:3, 3]
    return inverse


def relative_transform(parent: np.ndarray, child: np.ndarray) -> np.ndarray:
    return invert_transform(parent) @ child


def transform_point(matrix: np.ndarray, point: np.ndarray) -> np.ndarray:
    return matrix[:3, :3] @ np.asarray(point, dtype=np.float64) + matrix[:3, 3]


def transform_direction(matrix: np.ndarray, direction: np.ndarray) -> np.ndarray:
    return matrix[:3, :3] @ np.asarray(direction, dtype=np.float64)


def rotation_aligning(source: np.ndarray, destination: np.ndarray) -> np.ndarray:
    source = _normalize(source)
    destination = _normalize(destination)
    cosine = float(np.clip(np.dot(source, destination), -1.0, 1.0))
    if cosine > 1.0 - _ALIGN_EPS:
        return np.eye(3, dtype=np.float64)
    if cosine < -1.0 + _ALIGN_EPS:
        helper = np.array([1.0, 0.0, 0.0]) if abs(float(source[0])) < 0.9 else np.array([0.0, 1.0, 0.0])
        axis = _normalize(np.cross(source, helper))
        return axis_angle(axis, math.pi)
    crossed = np.cross(source, destination)
    skew = _skew(crossed)
    return np.eye(3, dtype=np.float64) + skew + skew @ skew * ((1.0 - cosine) / float(np.dot(crossed, crossed)))


def axis_angle(axis: np.ndarray, theta_rad: float) -> np.ndarray:
    normal = _normalize(axis)
    skew = _skew(normal)
    return np.eye(3, dtype=np.float64) + math.sin(theta_rad) * skew + (skew @ skew) * (1.0 - math.cos(theta_rad))


def signed_angle_around(from_vec: np.ndarray, to_vec: np.ndarray, axis: np.ndarray) -> float:
    normal = _normalize(axis)
    start = from_vec - np.dot(from_vec, normal) * normal
    dest = to_vec - np.dot(to_vec, normal) * normal
    start_norm = float(np.linalg.norm(start))
    dest_norm = float(np.linalg.norm(dest))
    if start_norm < _AXIS_EPS or dest_norm < _AXIS_EPS:
        return 0.0
    start = start / start_norm
    dest = dest / dest_norm
    sine = float(np.dot(normal, np.cross(start, dest)))
    cosine = float(np.clip(np.dot(start, dest), -1.0, 1.0))
    return math.atan2(sine, cosine)


def _project_onto_plane(vector: np.ndarray, origin: np.ndarray, axis: np.ndarray) -> np.ndarray:
    offset = vector - origin
    normal = _normalize(axis)
    return offset - np.dot(offset, normal) * normal


def solve_mount_transform(
    parent_world: np.ndarray,
    parent_mount: dict[str, Any],
    child_mount: dict[str, Any],
    *,
    parent_index: tuple[int, int] | None = None,
    child_index: tuple[int, int] | None = None,
    spin_deg: float = 0.0,
    secondary: dict[str, Any] | None = None,
) -> np.ndarray:
    """Place the child so its mount hole coincides with the parent, axes anti-parallel."""
    from talongym.robot.mounts import hole_in_part, mount_axis

    parent_hole = transform_point(parent_world, hole_in_part(parent_mount, parent_index))
    parent_axis = transform_direction(parent_world, mount_axis(parent_mount))
    child_hole = hole_in_part(child_mount, child_index)
    child_axis = mount_axis(child_mount)
    desired_axis = -_normalize(parent_axis)
    rotation = rotation_aligning(child_axis, desired_axis)
    if secondary:
        spin_deg = _spin_from_secondary(
            parent_world,
            parent_hole,
            desired_axis,
            rotation,
            child_hole,
            child_mount,
            secondary,
        )
    if abs(spin_deg) > 1e-12:
        rotation = axis_angle(desired_axis, math.radians(spin_deg)) @ rotation
    translation = parent_hole - rotation @ child_hole
    matrix = np.eye(4, dtype=np.float64)
    matrix[:3, :3] = rotation
    matrix[:3, 3] = translation
    return matrix


def _spin_from_secondary(
    parent_world: np.ndarray,
    parent_hole: np.ndarray,
    desired_axis: np.ndarray,
    rotation: np.ndarray,
    child_hole: np.ndarray,
    child_mount: dict[str, Any],
    secondary: dict[str, Any],
) -> float:
    from talongym.robot.mounts import hole_in_part

    parent_second = transform_point(parent_world, hole_in_part(secondary["parent_mount"], secondary.get("parent_index")))
    child_second = hole_in_part(secondary["child_mount"], secondary.get("child_index"))
    translation = parent_hole - rotation @ child_hole
    child_second_world = rotation @ child_second + translation
    parent_radial = _project_onto_plane(parent_second, parent_hole, desired_axis)
    child_radial = _project_onto_plane(child_second_world, parent_hole, desired_axis)
    parent_radius = float(np.linalg.norm(parent_radial))
    child_radius = float(np.linalg.norm(child_radial))
    if abs(parent_radius - child_radius) > PATTERN_MATCH_TOL_IN:
        raise AssemblyError(
            f"secondary hole/pattern constraint does not match ({parent_radius:.4f} in vs {child_radius:.4f} in)"
        )
    if parent_radius < PATTERN_MATCH_TOL_IN:
        return 0.0
    return math.degrees(signed_angle_around(child_radial, parent_radial, desired_axis))
