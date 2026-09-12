from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np

from talongym.sim.geometry import AABB, wrap_angle


@dataclass
class Body:
    id: str
    x: float
    y: float
    heading: float
    vx: float = 0.0
    vy: float = 0.0
    omega: float = 0.0
    hx: float = 9.0
    hy: float = 9.0
    dynamic: bool = True
    alliance: str = "red"
    kind: str = "aabb"
    radius: float = 0.0
    restitution: float = 0.2
    mass: float = 1.0

    def aabb(self) -> AABB:
        if self.kind == "circle" and self.radius > 0:
            return AABB(self.x, self.y, self.radius, self.radius, self.heading)
        return AABB(self.x, self.y, self.hx, self.hy, self.heading)


@dataclass
class ContactSet:
    wall: bool = False
    robot: bool = False
    piece: bool = False


@dataclass
class WorldStep:
    robots: list[Body]
    pieces: list[Body]
    obstacles: list[AABB]
    walls: list[AABB]
    max_vel: float = 30.0
    max_accel: float = 30.0
    max_omega: float = 1.0
    max_ang_accel: float = 1.0


class PhysicsBackend(Protocol):
    name: str

    def reset_batch(self, n: int) -> None: ...

    def step_batch(self, states: list[WorldStep], dt: float) -> list[ContactSet]: ...

    def contacts(self, index: int = 0) -> ContactSet: ...


def perimeter_walls(half_w: float, half_d: float, thickness: float = 2.0) -> list[AABB]:
    t = thickness
    return [
        AABB(0.0, half_d + t * 0.5, half_w + t, t * 0.5),
        AABB(0.0, -half_d - t * 0.5, half_w + t, t * 0.5),
        AABB(-half_w - t * 0.5, 0.0, t * 0.5, half_d + t),
        AABB(half_w + t * 0.5, 0.0, t * 0.5, half_d + t),
    ]


def math_hypot(x: float, y: float) -> float:
    return float(np.hypot(x, y))


class KinematicBackend:
    """CI-only chassis integrator. No piece contacts."""

    name = "kinematic"

    def __init__(self, field_half_w: float, field_half_d: float) -> None:
        self.hw = field_half_w
        self.hd = field_half_d
        self._contacts: list[ContactSet] = [ContactSet()]

    def reset_batch(self, n: int) -> None:
        self._contacts = [ContactSet() for _ in range(max(1, n))]

    def step_batch(self, states: list[WorldStep], dt: float) -> list[ContactSet]:
        out: list[ContactSet] = []
        for state in states:
            flags = ContactSet()
            others = [b for b in state.robots]
            for body in state.robots:
                if not body.dynamic:
                    continue
                _clip_chassis(body, state.max_vel, state.max_omega)
                body.x += body.vx * dt
                body.y += body.vy * dt
                body.heading = wrap_angle(body.heading + body.omega * dt)
                w, r = _resolve_chassis(body, state.walls + state.obstacles, others, self.hw, self.hd)
                flags.wall = flags.wall or w
                flags.robot = flags.robot or r
            out.append(flags)
        self._contacts = out
        return out

    def contacts(self, index: int = 0) -> ContactSet:
        if not self._contacts:
            return ContactSet()
        return self._contacts[min(index, len(self._contacts) - 1)]

    def step_world(self, state: WorldStep, dt: float) -> ContactSet:
        return self.step_batch([state], dt)[0]


class Planar2DBackend:
    """Phase 0 production fallback: planar contacts for chassis, walls, and floor pieces.

    Launch / classify remain FSM + time-of-flight, not rigid-body flight.
    """

    name = "planar2d"

    def __init__(self, field_half_w: float, field_half_d: float) -> None:
        self.hw = field_half_w
        self.hd = field_half_d
        self._contacts: list[ContactSet] = [ContactSet()]

    def reset_batch(self, n: int) -> None:
        self._contacts = [ContactSet() for _ in range(max(1, n))]

    def step_batch(self, states: list[WorldStep], dt: float) -> list[ContactSet]:
        out = [self._step_one(state, dt) for state in states]
        self._contacts = out
        return out

    def contacts(self, index: int = 0) -> ContactSet:
        if not self._contacts:
            return ContactSet()
        return self._contacts[min(index, len(self._contacts) - 1)]

    def step_world(self, state: WorldStep, dt: float) -> ContactSet:
        return self.step_batch([state], dt)[0]

    def _step_one(self, state: WorldStep, dt: float) -> ContactSet:
        flags = ContactSet()
        robots = state.robots
        pieces = [p for p in state.pieces if p.dynamic]
        walls = list(state.walls)
        obstacles = list(state.obstacles)

        for body in robots:
            if not body.dynamic:
                continue
            _clip_chassis(body, state.max_vel, state.max_omega)
            body.x += body.vx * dt
            body.y += body.vy * dt
            body.heading = wrap_angle(body.heading + body.omega * dt)
            w, r = _resolve_chassis(body, walls + obstacles, robots, self.hw, self.hd)
            flags.wall = flags.wall or w
            flags.robot = flags.robot or r

        for piece in pieces:
            piece.vx *= 0.92
            piece.vy *= 0.92
            if abs(piece.vx) < 0.05:
                piece.vx = 0.0
            if abs(piece.vy) < 0.05:
                piece.vy = 0.0
            piece.x += piece.vx * dt
            piece.y += piece.vy * dt
            if _circle_vs_boxes(piece, walls, bounce=True):
                flags.wall = True
            _circle_vs_boxes(piece, obstacles, bounce=True)

        for i, a in enumerate(pieces):
            for b in pieces[i + 1 :]:
                if _separate_circles(a, b):
                    flags.piece = True

        for robot in robots:
            for piece in pieces:
                if _chassis_vs_circle(robot, piece):
                    flags.piece = True
        return flags


def _clip_chassis(body: Body, max_vel: float, max_omega: float) -> None:
    speed = math_hypot(body.vx, body.vy)
    if speed > max_vel and speed > 1e-9:
        s = max_vel / speed
        body.vx *= s
        body.vy *= s
    body.omega = float(np.clip(body.omega, -max_omega, max_omega))


def _resolve_chassis(
    body: Body,
    boxes: list[AABB],
    others: list[Body],
    hw: float,
    hd: float,
) -> tuple[bool, bool]:
    wall_hit = False
    robot_hit = False
    margin_x = hw - body.hx
    margin_y = hd - body.hy
    if body.x < -margin_x:
        body.x = -margin_x
        body.vx = 0.0
        wall_hit = True
    if body.x > margin_x:
        body.x = margin_x
        body.vx = 0.0
        wall_hit = True
    if body.y < -margin_y:
        body.y = -margin_y
        body.vy = 0.0
        wall_hit = True
    if body.y > margin_y:
        body.y = margin_y
        body.vy = 0.0
        wall_hit = True
    me = body.aabb()
    for box in boxes:
        if me.overlaps_aabb(box):
            wall_hit = True
            _separate_aabb(body, box)
            me = body.aabb()
    for ob in others:
        if ob.id == body.id:
            continue
        if me.overlaps_aabb(ob.aabb()):
            robot_hit = True
            _separate_bodies(body, ob)
            me = body.aabb()
    return wall_hit, robot_hit


def _separate_aabb(body: Body, box: AABB) -> None:
    dx_left = body.aabb().maxx - box.minx
    dx_right = box.maxx - body.aabb().minx
    dy_down = body.aabb().maxy - box.miny
    dy_up = box.maxy - body.aabb().miny
    min_x = min(dx_left, dx_right)
    min_y = min(dy_down, dy_up)
    if min_x < min_y:
        body.x -= dx_left if dx_left < dx_right else -dx_right
        body.vx = 0.0
    else:
        body.y -= dy_down if dy_down < dy_up else -dy_up
        body.vy = 0.0


def _separate_bodies(a: Body, b: Body) -> None:
    dx = a.x - b.x
    dy = a.y - b.y
    overlap_x = a.hx + b.hx - abs(dx)
    overlap_y = a.hy + b.hy - abs(dy)
    if overlap_x <= 0 or overlap_y <= 0:
        return
    if overlap_x < overlap_y:
        push = overlap_x / 2.0 if b.dynamic else overlap_x
        sign = 1.0 if dx >= 0 else -1.0
        a.x += sign * push
        if b.dynamic:
            b.x -= sign * push
        a.vx = 0.0
    else:
        push = overlap_y / 2.0 if b.dynamic else overlap_y
        sign = 1.0 if dy >= 0 else -1.0
        a.y += sign * push
        if b.dynamic:
            b.y -= sign * push
        a.vy = 0.0


def _circle_vs_boxes(piece: Body, boxes: list[AABB], bounce: bool) -> bool:
    hit = False
    r = piece.radius or max(piece.hx, piece.hy)
    for box in boxes:
        if not box.overlaps_circle(piece.x, piece.y, r):
            continue
        hit = True
        nx = min(max(piece.x, box.minx), box.maxx)
        ny = min(max(piece.y, box.miny), box.maxy)
        dx, dy = piece.x - nx, piece.y - ny
        dist = math_hypot(dx, dy)
        if dist < 1e-9:
            # Center inside: push out on min axis.
            left = piece.x - box.minx
            right = box.maxx - piece.x
            down = piece.y - box.miny
            up = box.maxy - piece.y
            m = min(left, right, down, up)
            if m == left:
                piece.x = box.minx - r
                piece.vx = -abs(piece.vx) * (piece.restitution if bounce else 0.0)
            elif m == right:
                piece.x = box.maxx + r
                piece.vx = abs(piece.vx) * (piece.restitution if bounce else 0.0)
            elif m == down:
                piece.y = box.miny - r
                piece.vy = -abs(piece.vy) * (piece.restitution if bounce else 0.0)
            else:
                piece.y = box.maxy + r
                piece.vy = abs(piece.vy) * (piece.restitution if bounce else 0.0)
        else:
            overlap = r - dist
            ux, uy = dx / dist, dy / dist
            piece.x += ux * overlap
            piece.y += uy * overlap
            vn = piece.vx * ux + piece.vy * uy
            if bounce and vn < 0:
                piece.vx -= (1.0 + piece.restitution) * vn * ux
                piece.vy -= (1.0 + piece.restitution) * vn * uy
            else:
                piece.vx *= 0.4
                piece.vy *= 0.4
    return hit


def _separate_circles(a: Body, b: Body) -> bool:
    ra = a.radius or max(a.hx, a.hy)
    rb = b.radius or max(b.hx, b.hy)
    dx, dy = a.x - b.x, a.y - b.y
    dist = math_hypot(dx, dy)
    min_d = ra + rb
    if dist >= min_d:
        return False
    if dist < 1e-9:
        dx, dy, dist = 1.0, 0.0, 1.0
    overlap = min_d - dist
    ux, uy = dx / dist, dy / dist
    ma = max(a.mass, 1e-3)
    mb = max(b.mass, 1e-3)
    total = ma + mb
    a.x += ux * overlap * (mb / total)
    a.y += uy * overlap * (mb / total)
    b.x -= ux * overlap * (ma / total)
    b.y -= uy * overlap * (ma / total)
    rvx, rvy = a.vx - b.vx, a.vy - b.vy
    vn = rvx * ux + rvy * uy
    if vn < 0:
        e = min(a.restitution, b.restitution)
        impulse = -(1.0 + e) * vn / (1.0 / ma + 1.0 / mb)
        a.vx += impulse / ma * ux
        a.vy += impulse / ma * uy
        b.vx -= impulse / mb * ux
        b.vy -= impulse / mb * uy
    return True


def _chassis_vs_circle(robot: Body, piece: Body) -> bool:
    box = robot.aabb()
    r = piece.radius or max(piece.hx, piece.hy)
    if not box.overlaps_circle(piece.x, piece.y, r):
        return False
    nx = min(max(piece.x, box.minx), box.maxx)
    ny = min(max(piece.y, box.miny), box.maxy)
    dx, dy = piece.x - nx, piece.y - ny
    dist = math_hypot(dx, dy)
    if dist < 1e-9:
        left = piece.x - box.minx
        right = box.maxx - piece.x
        down = piece.y - box.miny
        up = box.maxy - piece.y
        m = min(left, right, down, up)
        if m == left:
            piece.x = box.minx - r
        elif m == right:
            piece.x = box.maxx + r
        elif m == down:
            piece.y = box.miny - r
        else:
            piece.y = box.maxy + r
        piece.vx += robot.vx * 0.35
        piece.vy += robot.vy * 0.35
        return True
    overlap = r - dist
    if overlap <= 0:
        return False
    ux, uy = dx / dist, dy / dist
    piece.x += ux * overlap
    piece.y += uy * overlap
    piece.vx += robot.vx * 0.45 + ux * 4.0
    piece.vy += robot.vy * 0.45 + uy * 4.0
    return True


def default_backend(field_half_w: float, field_half_d: float) -> PhysicsBackend:
    try:
        import talongym_engine  # noqa: F401

        name = getattr(talongym_engine, "engine_name", lambda: "")()
        if name and "stub" not in str(name):
            from talongym.sim.rapier import Rapier2DBackend

            return Rapier2DBackend(field_half_w, field_half_d)
    except Exception:
        pass
    return Planar2DBackend(field_half_w, field_half_d)
