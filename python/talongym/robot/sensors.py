from __future__ import annotations

import math
from typing import Any

from talongym.sim.geometry import AABB, deg_to_rad, ray_hits_aabb, wrap_angle


def camera_world_pose(robot_x: float, robot_y: float, robot_heading: float, sensor: dict[str, Any]) -> tuple[float, float, float]:
    pose = sensor.get("poseOnRobot") or {}
    lx = float(pose.get("x", 0.0))
    ly = float(pose.get("y", 0.0))
    lh = deg_to_rad(float(pose.get("headingDeg", 0.0)))
    c, s = math.cos(robot_heading), math.sin(robot_heading)
    wx = robot_x + c * lx - s * ly
    wy = robot_y + s * lx + c * ly
    return wx, wy, wrap_angle(robot_heading + lh)


def detect_tags(
    cam_x: float,
    cam_y: float,
    cam_heading: float,
    sensor: dict[str, Any],
    tags: list[dict[str, Any]],
    occluders: list[AABB],
    rng,
) -> list[dict[str, Any]]:
    fov = math.radians(float(sensor.get("fovDeg", 70.0)))
    max_range = float(sensor.get("rangeIn", 96.0))
    fn = float(sensor.get("falseNegativeRate", 0.0))
    hits: list[dict[str, Any]] = []
    for tag in tags:
        pose = tag.get("pose") or {}
        tx, ty = float(pose.get("x", 0.0)), float(pose.get("y", 0.0))
        dx, dy = tx - cam_x, ty - cam_y
        dist = math.hypot(dx, dy)
        if dist < 1e-6 or dist > max_range:
            continue
        bearing = wrap_angle(math.atan2(dy, dx) - cam_heading)
        if abs(bearing) > fov / 2.0:
            continue
        if tag.get("sideFaceObstructed"):
            tag_h = deg_to_rad(float(pose.get("headingDeg", 0.0)))
            nx, ny = math.cos(tag_h), math.sin(tag_h)
            # Tag normal should point roughly toward camera.
            if nx * (-dx) + ny * (-dy) < 0:
                continue
        occluded = False
        ux, uy = dx / dist, dy / dist
        for box in occluders:
            if ray_hits_aabb(cam_x, cam_y, ux, uy, box, dist - 1.0):
                occluded = True
                break
        visible = (not occluded) and (rng.random() >= fn)
        hits.append(
            {
                "tagId": int(tag["tagId"]),
                "bearing": bearing,
                "range": dist,
                "visible": visible,
                "occluded": occluded,
                "role": tag.get("role"),
                "mapsTo": tag.get("mapsTo"),
            }
        )
    return hits
