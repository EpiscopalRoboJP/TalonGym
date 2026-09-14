from __future__ import annotations

import math
from typing import Any

import numpy as np


def scripted_auto(obs: dict[str, np.ndarray], info: dict[str, Any] | None = None) -> dict[str, Any]:
    """LEAVE + PARK baseline for BIOBUZZ AUTO."""
    return scripted_biobuzz(obs, info)


def scripted_biobuzz(obs: dict[str, np.ndarray], info: dict[str, Any] | None = None) -> dict[str, Any]:
    pose = obs["pose_noisy"]
    x, y = float(pose[0]), float(pose[1])
    held = float(obs["held_count"][0])
    pieces = obs["nearest_pieces"]
    t_left = float(obs["time_remaining_s"][0])
    cell_x, cell_y = -9.4, 9.4
    park_x, park_y = -54.0, -48.0
    if t_left < 5:
        return {
            "target_pose": np.array([park_x, park_y, -math.pi / 2], dtype=np.float32),
            "speed_frac": np.array([1.0], dtype=np.float32),
            "mechanism": 0,
        }
    if held >= 1:
        dist = math.hypot(x - cell_x, y - cell_y)
        return {
            "target_pose": np.array([cell_x, cell_y, math.pi / 2], dtype=np.float32),
            "speed_frac": np.array([0.95], dtype=np.float32),
            "mechanism": 2 if dist < 22 else 0,
        }
    valid = pieces[0, 4] > 0.5
    if valid:
        dist = math.hypot(float(pieces[0, 0]), float(pieces[0, 1]))
        if dist < 24:
            tx = x + float(pieces[0, 0])
            ty = y + float(pieces[0, 1])
            return {
                "target_pose": np.array([tx, ty, math.atan2(ty - y, tx - x)], dtype=np.float32),
                "speed_frac": np.array([0.9], dtype=np.float32),
                "mechanism": 1 if dist < 16 else 0,
            }
    return {
        "target_pose": np.array([park_x, park_y, -math.pi / 2], dtype=np.float32),
        "speed_frac": np.array([1.0], dtype=np.float32),
        "mechanism": 0,
    }


def random_policy(obs: dict[str, np.ndarray], info: dict[str, Any] | None = None) -> dict[str, Any]:
    pose = obs["pose_noisy"]
    jitter = np.random.uniform(-20, 20, size=2)
    return {
        "target_pose": np.array([pose[0] + jitter[0], pose[1] + jitter[1], pose[2]], dtype=np.float32),
        "speed_frac": np.array([0.6], dtype=np.float32),
        "mechanism": int(np.random.randint(0, 3)),
    }
