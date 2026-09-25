from __future__ import annotations

import math
from typing import Any

import numpy as np

from talongym.sim.geometry import AABB, detour_waypoints, path_length, wrap_angle
from talongym.sim.world import FOLLOWER_CLEARANCE_IN

# BIOBUZZ AUTO plan for the default 18 in mecanum: launch the 4 preloads into the up CELL
# (3 staged NECTAR + 4 = HIVE TIP, 20), then PARK in the LOADING ZONE (5) while clear of the
# wall (LEAVE, 3). Loose POLLEN scores nothing in AUTO unless it completes another tip, so
# the baseline does not chase it.
#
# Red-side poses per start slot; blue mirrors them. Slot 0's launch pose matches field.json
# red_launch_spot. Slot 1 shoots from the far side of the HIVE so the two never share a spot.
LAUNCH_POSES = ((-12.0, -40.0, math.pi / 2), (-12.0, 40.0, -math.pi / 2))
# The red PARK tape spans x -70.6..-59.6 / y 23.75..46.75. A chassis only needs
# partial overlap, so stopping farther infield leaves room for the asymmetric
# scoring assembly to turn clear of the wall. The two parks remain 26 in apart.
PARK_POSES = ((-52.5, 23.0, 0.0), (-52.5, 49.0, 0.0))
# A teammate this close to a park pose is holding it (a static teammate never leaves its start slot).
PARK_TAKEN_IN = 15.0
# hive_frame_west / hive_frame_east footprints. The waypoint follower routes around these and
# other robots itself; the policy only needs them to estimate how long the park drive takes.
HIVE_FRAMES = (AABB(-24.73, 0.0, 2.0, 19.475), AABB(24.73, 0.0, 2.0, 19.475))
ROBOT_HALF_IN = 9.0
LAUNCH_TOL_IN = 3.0
LAUNCH_HEADING_TOL_RAD = 0.15
# Conservative average speed (braking at route corners included) and slack for the park trip.
PARK_SPEED_IN_PER_S = 18.0
PARK_SLACK_S = 2.0


def scripted_auto(obs: dict[str, np.ndarray], info: dict[str, Any] | None = None) -> dict[str, Any]:
    """HIVE TIP + LEAVE + PARK baseline for BIOBUZZ AUTO."""
    return scripted_biobuzz(obs, info)


def _mirror_for_alliance(x: float, y: float, heading: float, alliance: str | None) -> tuple[float, float, float]:
    """BIOBUZZ AUTO field targets are defined for red; blue's side is the same layout
    rotated 180 degrees, so mirror position and heading rather than hardcode both."""
    if alliance != "blue":
        return x, y, heading
    mirrored = (heading + math.pi + math.pi) % (2 * math.pi) - math.pi
    return -x, -y, mirrored


def _teammate_xy(obs: dict[str, np.ndarray]) -> tuple[float, float] | None:
    mate = obs.get("teammate_pose_noisy")
    # The env reports an all-zero pose when there is no teammate.
    if mate is None or not np.any(np.asarray(mate[:2]) != 0.0):
        return None
    return float(mate[0]), float(mate[1])


def _park_pose(slot: int, alliance: str, mate: tuple[float, float] | None) -> tuple[float, float, float]:
    preferred, other = PARK_POSES[slot], PARK_POSES[1 - slot]
    park = _mirror_for_alliance(*preferred, alliance)
    if mate is not None and math.hypot(mate[0] - park[0], mate[1] - park[1]) < PARK_TAKEN_IN:
        return _mirror_for_alliance(*other, alliance)
    return park


def _drive_to(target: tuple[float, float, float], mechanism: int = 0) -> dict[str, Any]:
    return {
        "target_pose": np.array(target, dtype=np.float32),
        "speed_frac": np.array([1.0], dtype=np.float32),
        "mechanism": mechanism,
    }


def scripted_biobuzz(obs: dict[str, np.ndarray], info: dict[str, Any] | None = None) -> dict[str, Any]:
    pose = obs["pose_noisy"]
    x, y, h = float(pose[0]), float(pose[1]), float(pose[2])
    held = float(obs["held_count"][0])
    t_left = float(obs["time_remaining_s"][0])
    info = info or {}
    alliance = info.get("alliance", "red")
    slot = 1 if str(info.get("robot_id", "")).endswith("_1") else 0
    mate = _teammate_xy(obs)
    launch = _mirror_for_alliance(*LAUNCH_POSES[slot], alliance)
    park = _park_pose(slot, alliance, mate)
    boxes = list(HIVE_FRAMES)
    if mate is not None:
        boxes.append(AABB(mate[0], mate[1], ROBOT_HALF_IN, ROBOT_HALF_IN))
    clearance = ROBOT_HALF_IN + FOLLOWER_CLEARANCE_IN
    park_route = path_length(x, y, detour_waypoints(x, y, park[0], park[1], boxes, clearance))
    park_time_s = park_route / PARK_SPEED_IN_PER_S + PARK_SLACK_S
    if held >= 1 and t_left > park_time_s:
        at_spot = math.hypot(x - launch[0], y - launch[1]) < LAUNCH_TOL_IN
        aimed = abs(wrap_angle(launch[2] - h)) < LAUNCH_HEADING_TOL_RAD
        return _drive_to(launch, 2 if at_spot and aimed else 0)
    return _drive_to(park)


def random_policy(obs: dict[str, np.ndarray], info: dict[str, Any] | None = None) -> dict[str, Any]:
    pose = obs["pose_noisy"]
    jitter = np.random.uniform(-20, 20, size=2)
    return {
        "target_pose": np.array([pose[0] + jitter[0], pose[1] + jitter[1], pose[2]], dtype=np.float32),
        "speed_frac": np.array([0.6], dtype=np.float32),
        "mechanism": int(np.random.randint(0, 3)),
    }
