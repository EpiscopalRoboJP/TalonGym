from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any


def wrap_angle(rad: float) -> float:
    return (rad + math.pi) % (2 * math.pi) - math.pi


def deg_to_rad(deg: float) -> float:
    return deg * math.pi / 180.0


def rad_to_deg(rad: float) -> float:
    return rad * 180.0 / math.pi


@dataclass
class AABB:
    cx: float
    cy: float
    hx: float
    hy: float
    heading: float = 0.0

    @property
    def minx(self) -> float:
        return self.cx - self.hx

    @property
    def maxx(self) -> float:
        return self.cx + self.hx

    @property
    def miny(self) -> float:
        return self.cy - self.hy

    @property
    def maxy(self) -> float:
        return self.cy + self.hy

    def contains(self, x: float, y: float) -> bool:
        return self.minx <= x <= self.maxx and self.miny <= y <= self.maxy

    def overlaps_aabb(self, other: AABB) -> bool:
        return self.minx < other.maxx and self.maxx > other.minx and self.miny < other.maxy and self.maxy > other.miny

    def overlaps_circle(self, x: float, y: float, r: float) -> bool:
        nx = min(max(x, self.minx), self.maxx)
        ny = min(max(y, self.miny), self.maxy)
        dx, dy = x - nx, y - ny
        return dx * dx + dy * dy <= r * r


@dataclass
class Circle:
    x: float
    y: float
    r: float

    def contains(self, px: float, py: float) -> bool:
        dx, dy = px - self.x, py - self.y
        return dx * dx + dy * dy <= self.r * self.r


def shape_from_element(el: dict[str, Any]) -> AABB | Circle | None:
    pose = el.get("pose") or {}
    shape = el.get("shape") or {}
    kind = shape.get("kind", "none")
    x, y = float(pose.get("x", 0.0)), float(pose.get("y", 0.0))
    heading = deg_to_rad(float(pose.get("headingDeg", 0.0)))
    if kind == "aabb":
        w = float(shape.get("width", 0.0))
        d = float(shape.get("depth", 0.0))
        return AABB(x, y, w / 2.0, d / 2.0, heading)
    if kind == "circle":
        return Circle(x, y, float(shape.get("radius", 0.0)))
    return None


def point_in_shape(shape: AABB | Circle | None, x: float, y: float) -> bool:
    if shape is None:
        return False
    if isinstance(shape, AABB):
        return shape.contains(x, y)
    return shape.contains(x, y)


def volume_needs_z(el: dict[str, Any]) -> bool:
    tags = set(el.get("tags") or [])
    kind = str(el.get("type") or "")
    return bool(tags.intersection({"cell", "flower", "goal", "up_cell", "down_cell"})) or kind in {
        "goal",
        "cell",
        "flower",
    }


def point_in_volume(el: dict[str, Any], shape: AABB | Circle | None, x: float, y: float, z: float = 0.0, radius: float = 0.0) -> bool:
    if not point_in_shape(shape, x, y):
        return False
    if not volume_needs_z(el):
        return True
    pose = el.get("pose") or {}
    ez = float(pose.get("z") or 0.0)
    hh = float((el.get("shape") or {}).get("height") or 16.0) / 2.0
    return abs(z - ez) <= hh + radius + 2.0


def ray_hits_aabb(ox: float, oy: float, dx: float, dy: float, box: AABB, max_t: float) -> bool:
    invx = 1.0 / dx if abs(dx) > 1e-9 else 1e9
    invy = 1.0 / dy if abs(dy) > 1e-9 else 1e9
    tx1 = (box.minx - ox) * invx
    tx2 = (box.maxx - ox) * invx
    ty1 = (box.miny - oy) * invy
    ty2 = (box.maxy - oy) * invy
    tmin = max(min(tx1, tx2), min(ty1, ty2))
    tmax = min(max(tx1, tx2), max(ty1, ty2))
    if tmax < 0 or tmin > tmax:
        return False
    t = tmin if tmin >= 0 else tmax
    return 0.0 <= t <= max_t
