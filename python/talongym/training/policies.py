from __future__ import annotations

import math
from typing import Any

import numpy as np


def _season_slug(info: dict[str, Any] | None) -> str:
    priv = (info or {}).get("privileged") or {}
    return str(priv.get("season") or "decode")


def scripted_auto(obs: dict[str, np.ndarray], info: dict[str, Any] | None = None) -> dict[str, Any]:
    """LEAVE + intake + one score. Baseline for DECODE; also a generic AUTO."""
    slug = _season_slug(info)
    if slug == "into_the_deep":
        return scripted_into_the_deep(obs, info)
    if slug == "centerstage":
        return scripted_centerstage(obs, info)
    if slug == "biobuzz":
        return scripted_biobuzz(obs, info)
    return scripted_decode(obs, info)


def scripted_decode(obs: dict[str, np.ndarray], info: dict[str, Any] | None = None) -> dict[str, Any]:
    pose = obs["pose_noisy"]
    x, y, h = float(pose[0]), float(pose[1]), float(pose[2])
    held = float(obs["held_count"][0])
    pieces = obs["nearest_pieces"]
    mechanism = 0
    speed = 0.95
    tx, ty, th = x, y, h
    if y < -20:
        tx, ty, th = 12.0, 8.0, math.pi / 2
        valid = pieces[0, 4] > 0.5
        if valid and math.hypot(pieces[0, 0], pieces[0, 1]) < 14:
            mechanism = 1
    elif held < 1:
        valid = pieces[0, 4] > 0.5
        if valid:
            tx = x + float(pieces[0, 0])
            ty = y + float(pieces[0, 1])
            th = math.atan2(ty - y, tx - x)
            dist = math.hypot(pieces[0, 0], pieces[0, 1])
            mechanism = 1 if dist < 16 else 0
        else:
            tx, ty, th = 20.0, 12.0, math.pi / 2
    else:
        tx, ty, th = 16.0, 20.0, math.pi / 2
        mechanism = 2
    return {
        "target_pose": np.array([tx, ty, th], dtype=np.float32),
        "speed_frac": np.array([speed], dtype=np.float32),
        "mechanism": mechanism,
    }


def scripted_into_the_deep(obs: dict[str, np.ndarray], info: dict[str, Any] | None = None) -> dict[str, Any]:
    pose = obs["pose_noisy"]
    x, y, h = float(pose[0]), float(pose[1]), float(pose[2])
    held = float(obs["held_count"][0])
    pieces = obs["nearest_pieces"]
    t_left = float(obs["time_remaining_s"][0])
    if t_left < 4:
        return {
            "target_pose": np.array([60.0, -60.0, -math.pi / 2], dtype=np.float32),
            "speed_frac": np.array([1.0], dtype=np.float32),
            "mechanism": 0,
        }
    if held < 1:
        valid = pieces[0, 4] > 0.5
        if valid:
            tx = x + float(pieces[0, 0])
            ty = y + float(pieces[0, 1])
            th = math.atan2(ty - y, tx - x)
            dist = math.hypot(pieces[0, 0], pieces[0, 1])
            mech = 1 if dist < 16 else 0
            return {
                "target_pose": np.array([tx, ty, th], dtype=np.float32),
                "speed_frac": np.array([0.9], dtype=np.float32),
                "mechanism": mech,
            }
        return {
            "target_pose": np.array([24.0, 12.0, math.pi / 2], dtype=np.float32),
            "speed_frac": np.array([0.9], dtype=np.float32),
            "mechanism": 0,
        }
    return {
        "target_pose": np.array([-48.0, 48.0, math.pi / 2], dtype=np.float32),
        "speed_frac": np.array([0.95], dtype=np.float32),
        "mechanism": 2 if math.hypot(x + 48.0, y - 48.0) < 22 else 0,
    }


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


def scripted_centerstage(obs: dict[str, np.ndarray], info: dict[str, Any] | None = None) -> dict[str, Any]:
    pose = obs["pose_noisy"]
    x, y = float(pose[0]), float(pose[1])
    match_obs = obs["match_var_obs"][0]
    unknown = float(match_obs[-1]) > 0.5
    held = float(obs["held_count"][0])
    pieces = obs["nearest_pieces"]
    t_left = float(obs["time_remaining_s"][0])
    if unknown:
        return {
            "target_pose": np.array([0.0, -36.0, math.pi / 2], dtype=np.float32),
            "speed_frac": np.array([0.6], dtype=np.float32),
            "mechanism": 0,
        }
    domain = ["left", "center", "right"]
    idx = int(np.argmax(match_obs[:-1])) if match_obs[:-1].sum() > 0 else 1
    loc = domain[min(idx, 2)]
    spike_x = {"left": -24.0, "center": 0.0, "right": 24.0}[loc]
    if held >= 1 and y < -10:
        return {
            "target_pose": np.array([spike_x, -36.0, math.pi / 2], dtype=np.float32),
            "speed_frac": np.array([0.85], dtype=np.float32),
            "mechanism": 2 if math.hypot(x - spike_x, y + 36.0) < 14 else 0,
        }
    if held < 1 and pieces[0, 4] > 0.5:
        tx = x + float(pieces[0, 0])
        ty = y + float(pieces[0, 1])
        dist = math.hypot(pieces[0, 0], pieces[0, 1])
        return {
            "target_pose": np.array([tx, ty, math.atan2(ty - y, tx - x)], dtype=np.float32),
            "speed_frac": np.array([0.8], dtype=np.float32),
            "mechanism": 1 if dist < 16 else 0,
        }
    if t_left < 8 or y > -20:
        return {
            "target_pose": np.array([36.0, 54.0, math.pi / 2], dtype=np.float32),
            "speed_frac": np.array([1.0], dtype=np.float32),
            "mechanism": 0,
        }
    return {
        "target_pose": np.array([spike_x, -36.0, math.pi / 2], dtype=np.float32),
        "speed_frac": np.array([0.8], dtype=np.float32),
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
