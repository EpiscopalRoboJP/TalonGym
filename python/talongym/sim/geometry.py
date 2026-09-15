from __future__ import annotations

import heapq
import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

Point = tuple[float, float]


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


def world_polygon(origin_x: float, origin_y: float, heading: float, local: Sequence[Point]) -> list[Point]:
    """Robot-frame (+x forward, +y left) vertices to world XY."""
    c = math.cos(heading)
    s = math.sin(heading)
    out: list[Point] = []
    for rx, ry in local:
        out.append((origin_x + rx * c - ry * s, origin_y + rx * s + ry * c))
    return out


def aabb_polygon(box: AABB) -> list[Point]:
    return [
        (box.minx, box.miny),
        (box.maxx, box.miny),
        (box.maxx, box.maxy),
        (box.minx, box.maxy),
    ]


def _poly_axes(poly: Sequence[Point]) -> list[Point]:
    axes: list[Point] = []
    n = len(poly)
    for i in range(n):
        x1, y1 = poly[i]
        x2, y2 = poly[(i + 1) % n]
        nx, ny = y1 - y2, x2 - x1
        mag = math.hypot(nx, ny)
        if mag < 1e-12:
            continue
        axes.append((nx / mag, ny / mag))
    return axes


def _project_poly(poly: Sequence[Point], axis: Point) -> tuple[float, float]:
    dots = [p[0] * axis[0] + p[1] * axis[1] for p in poly]
    return min(dots), max(dots)


def polygons_overlap(a: Sequence[Point], b: Sequence[Point]) -> tuple[bool, Point | None]:
    """SAT. MTV pushes polygon a out of b when they overlap."""
    if len(a) < 3 or len(b) < 3:
        return False, None
    min_overlap = float("inf")
    mtv: Point = (0.0, 0.0)
    for axis in _poly_axes(a) + _poly_axes(b):
        amin, amax = _project_poly(a, axis)
        bmin, bmax = _project_poly(b, axis)
        if amax < bmin or bmax < amin:
            return False, None
        overlap = min(amax - bmin, bmax - amin)
        if overlap < min_overlap:
            min_overlap = overlap
            ac = 0.5 * (amin + amax)
            bc = 0.5 * (bmin + bmax)
            sign = -1.0 if ac < bc else 1.0
            mtv = (axis[0] * sign * min_overlap, axis[1] * sign * min_overlap)
    if min_overlap == float("inf"):
        return False, None
    return True, mtv


def point_in_polygon(x: float, y: float, poly: Sequence[Point]) -> bool:
    inside = False
    n = len(poly)
    j = n - 1
    for i in range(n):
        xi, yi = poly[i]
        xj, yj = poly[j]
        if ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / ((yj - yi) or 1e-12) + xi):
            inside = not inside
        j = i
    return inside


def closest_point_on_polygon(x: float, y: float, poly: Sequence[Point]) -> tuple[float, float, float]:
    """Return (qx, qy, dist) for the closest boundary point."""
    best_d = float("inf")
    best = (x, y)
    n = len(poly)
    for i in range(n):
        ax, ay = poly[i]
        bx, by = poly[(i + 1) % n]
        abx, aby = bx - ax, by - ay
        ab2 = abx * abx + aby * aby
        t = 0.0 if ab2 < 1e-12 else max(0.0, min(1.0, ((x - ax) * abx + (y - ay) * aby) / ab2))
        qx, qy = ax + t * abx, ay + t * aby
        d = math.hypot(x - qx, y - qy)
        if d < best_d:
            best_d = d
            best = (qx, qy)
    return best[0], best[1], best_d


def polygon_vs_circle(poly: Sequence[Point], x: float, y: float, r: float) -> tuple[bool, Point | None]:
    """If overlapping, return a vector that pushes the circle out of the polygon."""
    if len(poly) < 3 or r <= 0:
        return False, None
    qx, qy, dist = closest_point_on_polygon(x, y, poly)
    inside = point_in_polygon(x, y, poly)
    if inside:
        if dist < 1e-9:
            return True, (r, 0.0)
        ux, uy = (x - qx) / dist, (y - qy) / dist
        push = r + dist
        return True, (ux * push, uy * push)
    if dist >= r:
        return False, None
    if dist < 1e-9:
        return True, (r, 0.0)
    ux, uy = (x - qx) / dist, (y - qy) / dist
    push = r - dist
    return True, (ux * push, uy * push)


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


def segment_crosses_aabb(ax: float, ay: float, bx: float, by: float, box: AABB, eps: float = 1e-6) -> bool:
    """True when segment a-b passes through the box interior; grazing an edge or corner does not count."""
    dx, dy = bx - ax, by - ay
    t0, t1 = 0.0, 1.0
    for p, q in (
        (-dx, ax - (box.minx + eps)),
        (dx, (box.maxx - eps) - ax),
        (-dy, ay - (box.miny + eps)),
        (dy, (box.maxy - eps) - ay),
    ):
        if abs(p) < 1e-12:
            if q < 0:
                return False
            continue
        t = q / p
        if p < 0:
            t0 = max(t0, t)
        else:
            t1 = min(t1, t)
        if t0 > t1:
            return False
    return True


def _strictly_inside(box: AABB, x: float, y: float, eps: float = 1e-6) -> bool:
    return box.minx + eps < x < box.maxx - eps and box.miny + eps < y < box.maxy - eps


def detour_waypoints(ax: float, ay: float, bx: float, by: float, boxes: Sequence[AABB], inflate: float) -> list[Point]:
    """Shortest route from a to b around every box grown by ``inflate``; ends with b.

    A* over a visibility graph of the grown boxes' corners, so overlapping boxes close the gap
    between them. A box holding the start or the goal is ignored on the leg leaving or entering
    it (the chassis is already there or must go there). With no route, heads straight for b.
    """
    goal = (bx, by)
    grown = [AABB(b.cx, b.cy, b.hx + inflate, b.hy + inflate) for b in boxes]
    start_in = {i for i, g in enumerate(grown) if _strictly_inside(g, ax, ay)}
    goal_in = {i for i, g in enumerate(grown) if _strictly_inside(g, bx, by)}

    def blocked(p: Point, q: Point, skip: set[int]) -> bool:
        return any(i not in skip and segment_crosses_aabb(p[0], p[1], q[0], q[1], g) for i, g in enumerate(grown))

    if not blocked((ax, ay), goal, start_in | goal_in):
        return [goal]
    nodes: list[Point] = [(ax, ay), goal]
    for i, g in enumerate(grown):
        for c in aabb_polygon(g):
            if not any(j != i and _strictly_inside(h, *c) for j, h in enumerate(grown)):
                nodes.append(c)
    best = {0: 0.0}
    prev: dict[int, int] = {}
    done: set[int] = set()
    heap = [(math.hypot(bx - ax, by - ay), 0.0, 0)]
    while heap:
        _, cost, u = heapq.heappop(heap)
        if u in done:
            continue
        done.add(u)
        if u == 1:
            break
        for v in range(1, len(nodes)):
            if v in done:
                continue
            step = math.hypot(nodes[v][0] - nodes[u][0], nodes[v][1] - nodes[u][1])
            if cost + step >= best.get(v, math.inf):
                continue
            skip = (start_in if u == 0 else set()) | (goal_in if v == 1 else set())
            if blocked(nodes[u], nodes[v], skip):
                continue
            best[v] = cost + step
            prev[v] = u
            heapq.heappush(heap, (best[v] + math.hypot(bx - nodes[v][0], by - nodes[v][1]), best[v], v))
    if 1 not in prev:
        return [goal]
    route: list[Point] = []
    v = 1
    while v != 0:
        route.append(nodes[v])
        v = prev[v]
    return route[::-1]


def path_length(ax: float, ay: float, waypoints: Sequence[Point]) -> float:
    total = 0.0
    for wx, wy in waypoints:
        total += math.hypot(wx - ax, wy - ay)
        ax, ay = wx, wy
    return total


def push_out_of_boxes(x: float, y: float, boxes: Sequence[AABB], inflate: float) -> Point:
    """Move (x, y) to the nearest point outside every box grown by ``inflate`` (checked in order)."""
    for box in boxes:
        grown = AABB(box.cx, box.cy, box.hx + inflate, box.hy + inflate)
        if not (grown.minx < x < grown.maxx and grown.miny < y < grown.maxy):
            continue
        exits = [
            (x - grown.minx, grown.minx, y),
            (grown.maxx - x, grown.maxx, y),
            (y - grown.miny, x, grown.miny),
            (grown.maxy - y, x, grown.maxy),
        ]
        _, x, y = min(exits)
    return x, y
