"""Planar drivetrain limits: mecanum IK and motor-clipped body twist."""

from __future__ import annotations

import math
from typing import Any

import numpy as np


def wheel_free_speed_in_per_s(robot: dict[str, Any]) -> float:
    motors = robot.get("motors") or {}
    drive = robot.get("drivetrain") or {}
    rpm = float(motors.get("freeSpeedRpm") or 312.0)
    dia = float(drive.get("wheelDiameterIn") or 4.0)
    # freeSpeedRpm is the output-shaft / wheel speed after any listed gearRatio.
    return rpm * math.pi * dia / 60.0


def max_twist(robot: dict[str, Any]) -> tuple[float, float, float]:
    constraints = robot.get("constraints") or {}
    max_v = float(constraints.get("maxVelInPerS") or 30.0)
    motor_v = wheel_free_speed_in_per_s(robot)
    cap = min(max_v, motor_v) if motor_v > 1e-6 else max_v
    drive = robot.get("drivetrain") or {}
    strafe = float(drive.get("strafeMultiplier") or 1.0)
    max_w = math.radians(float(constraints.get("maxAngVelDegPerS") or 60.0))
    stall = float((robot.get("motors") or {}).get("stallTorqueNm") or 2.1)
    current = float((robot.get("motors") or {}).get("currentLimitA") or 20.0)
    # Crude current/torque derate: 20A vs a 20A limit is 1.0; never drop below 0.35.
    derate = float(np.clip(current / 20.0 * stall / 2.1, 0.35, 1.0))
    return cap * derate, cap * strafe * derate, max_w * derate


def clip_twist(vx: float, vy: float, omega: float, robot: dict[str, Any]) -> tuple[float, float, float]:
    """Clip chassis twist. Tank zeros vy. Swerve uses the same holonomic clip as mecanum (no module IK)."""
    kind = str((robot.get("drivetrain") or {}).get("type") or "mecanum")
    mx, my, mw = max_twist(robot)
    if kind == "tank":
        vy = 0.0
        my = 0.0
    vx = float(np.clip(vx, -mx, mx))
    vy = float(np.clip(vy, -my, my))
    omega = float(np.clip(omega, -mw, mw))
    speed = math.hypot(vx, vy)
    cap = mx
    if speed > cap and speed > 1e-9:
        s = cap / speed
        vx *= s
        vy *= s
    return vx, vy, omega


def mecanum_module_speeds(vx: float, vy: float, omega: float, robot: dict[str, Any]) -> np.ndarray:
    """Four wheel linear speeds (in/s): FL, FR, BL, BR in robot frame."""
    drive = robot.get("drivetrain") or {}
    track = float(drive.get("trackWidthIn") or 15.0)
    wheelbase = float(drive.get("wheelbaseIn") or track)
    L = (track + wheelbase) / 2.0
    return np.array(
        [
            vx - vy - omega * L,
            vx + vy + omega * L,
            vx + vy - omega * L,
            vx - vy + omega * L,
        ],
        dtype=np.float64,
    )
