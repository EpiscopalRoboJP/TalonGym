"""Robot-frame intakes and launchers. Poses are inches, +x forward, +y left."""

from __future__ import annotations

import math
from typing import Any

MOVING_EPS_IN_PER_S = 1.0


def _deg_to_rad(deg: float) -> float:
    return deg * math.pi / 180.0


def _wrap_angle(rad: float) -> float:
    return (rad + math.pi) % (2 * math.pi) - math.pi


def robot_to_world(rx: float, ry: float, heading: float, lx: float, ly: float) -> tuple[float, float]:
    c, s = math.cos(heading), math.sin(heading)
    return rx + c * lx - s * ly, ry + s * lx + c * ly


def pose_world(body_x: float, body_y: float, body_heading: float, pose: dict[str, Any]) -> tuple[float, float, float, float, float]:
    """Return field x, y, z, yaw rad, pitch rad for a poseOnRobot dict."""
    lx = float(pose.get("x") or 0.0)
    ly = float(pose.get("y") or 0.0)
    lz = float(pose.get("z") or 0.0)
    lh = _deg_to_rad(float(pose.get("headingDeg") or 0.0))
    pitch = _deg_to_rad(float(pose.get("pitchDeg") or 0.0))
    wx, wy = robot_to_world(body_x, body_y, body_heading, lx, ly)
    yaw = _wrap_angle(body_heading + lh)
    return wx, wy, lz, yaw, pitch


def _clip_deg(value: float, lo_hi: list[Any] | tuple[Any, ...] | None) -> float:
    if not lo_hi or len(lo_hi) < 2:
        return value
    lo, hi = float(lo_hi[0]), float(lo_hi[1])
    if hi < lo:
        lo, hi = hi, lo
    return min(max(value, lo), hi)


def launcher_aim(launcher: dict[str, Any]) -> tuple[float, float]:
    """Chassis-relative yaw/pitch degrees after turret range clip."""
    pose = launcher.get("poseOnRobot") or {}
    yaw = float(pose.get("headingDeg") or 0.0)
    pitch = float(pose.get("pitchDeg") or 0.0)
    if str(launcher.get("aimMode") or "chassis_fixed") == "turret":
        yaw = _clip_deg(yaw, launcher.get("yawRangeDeg"))
        pitch = _clip_deg(pitch, launcher.get("pitchRangeDeg"))
    return yaw, pitch


def muzzle_velocity(speed: float, yaw: float, pitch: float) -> tuple[float, float, float]:
    cp, sp = math.cos(pitch), math.sin(pitch)
    return speed * cp * math.cos(yaw), speed * cp * math.sin(yaw), speed * sp


def piece_in_intake(
    px: float,
    py: float,
    pz: float,
    pr: float,
    body_x: float,
    body_y: float,
    body_heading: float,
    intake: dict[str, Any],
) -> bool:
    pose = intake.get("poseOnRobot") or {}
    wx, wy, wz, yaw, _pitch = pose_world(body_x, body_y, body_heading, pose)
    dx, dy = px - wx, py - wy
    c, s = math.cos(yaw), math.sin(yaw)
    local_x = c * dx + s * dy
    local_y = -s * dx + c * dy
    reach = float(intake.get("reachIn") or 4.0)
    width = float(intake.get("widthIn") or 8.0)
    height = float(intake.get("heightIn") or 6.0)
    if local_x < -pr or local_x > reach + pr:
        return False
    if abs(local_y) > width / 2.0 + pr:
        return False
    return abs(pz - wz) <= height / 2.0 + pr + 2.0


def chassis_moving(vx: float, vy: float) -> bool:
    return math.hypot(vx, vy) > MOVING_EPS_IN_PER_S
