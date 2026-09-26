from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Protocol

import numpy as np

from talongym.sim.geometry import (
    AABB,
    aabb_polygon,
    polygon_vs_circle,
    polygons_overlap,
    world_polygon,
    wrap_angle,
)

INCHES_PER_METER = 39.37007874015748
IN_G = 386.0886
AIR_DENSITY_KG_PER_IN3 = 1.225 / (INCHES_PER_METER ** 3)
NM_TO_INCH_TORQUE = INCHES_PER_METER ** 2
RPM_TO_RAD_S = 2.0 * math.pi / 60.0


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
    z: float = 5.0
    vz: float = 0.0
    kick: bool = False
    footprint: tuple[tuple[float, float], ...] | None = None
    type_id: str = ""
    qw: float = 1.0
    qx: float = 0.0
    qy: float = 0.0
    qz: float = 0.0
    wx: float = 0.0
    wy: float = 0.0
    wz: float = 0.0

    def aabb(self) -> AABB:
        if self.kind == "circle" and self.radius > 0:
            return AABB(self.x, self.y, self.radius, self.radius, self.heading)
        return AABB(self.x, self.y, self.hx, self.hy, self.heading)

    def world_footprint(self) -> list[tuple[float, float]] | None:
        if self.kind != "mesh" or not self.footprint or len(self.footprint) < 3:
            return None
        return world_polygon(self.x, self.y, self.heading, self.footprint)


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
    field_mechanism_targets: dict[str, float] = field(default_factory=dict)
    robot_mechanism_states: dict[str, dict[str, Any]] = field(default_factory=dict)
    piece_owners: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class PieceInteraction:
    """Robot-frame wrench on a game piece from configured mechanism state."""

    fx: float = 0.0
    fy: float = 0.0
    fz: float = 0.0
    tx: float = 0.0
    ty: float = 0.0
    tz: float = 0.0
    flywheel_load_inch: float = 0.0


def _smoothstep(value: float, lo: float, hi: float) -> float:
    if value <= lo:
        return 0.0
    if value >= hi:
        return 1.0
    t = (value - lo) / (hi - lo)
    return t * t * (3.0 - 2.0 * t)


def _clip_force(fx: float, fy: float, fz: float, mass: float) -> tuple[float, float, float]:
    max_accel = 2200.0
    mag = math.sqrt(fx * fx + fy * fy + fz * fz)
    limit = max(mass, 1e-4) * max_accel
    if mag <= limit or mag < 1e-12:
        return fx, fy, fz
    scale = limit / mag
    return fx * scale, fy * scale, fz * scale


def world_to_robot_xy(px: float, py: float, rx: float, ry: float, heading: float) -> tuple[float, float]:
    dx, dy = px - rx, py - ry
    cos_h, sin_h = math.cos(heading), math.sin(heading)
    return cos_h * dx + sin_h * dy, -sin_h * dx + cos_h * dy


def rotate_robot_to_world(lx: float, ly: float, heading: float) -> tuple[float, float]:
    cos_h, sin_h = math.cos(heading), math.sin(heading)
    return cos_h * lx - sin_h * ly, sin_h * lx + cos_h * ly


def rotate_world_vel_to_robot(vx: float, vy: float, heading: float) -> tuple[float, float]:
    cos_h, sin_h = math.cos(heading), math.sin(heading)
    return cos_h * vx + sin_h * vy, -sin_h * vx + cos_h * vy


def relative_velocity_robot(
    vx: float,
    vy: float,
    vz: float,
    robot_vx: float,
    robot_vy: float,
    omega: float,
    heading: float,
    rel_world_x: float,
    rel_world_y: float,
) -> tuple[float, float, float]:
    """Piece velocity relative to the chassis, expressed in the robot frame."""
    rvx = robot_vx - omega * rel_world_y
    rvy = robot_vy + omega * rel_world_x
    return (
        *rotate_world_vel_to_robot(vx - rvx, vy - rvy, heading),
        vz,
    )


def aerodynamic_wrench(
    vx: float,
    vy: float,
    vz: float,
    wx: float,
    wy: float,
    wz: float,
    radius_in: float,
    *,
    drag_coefficient: float,
    magnus_coefficient: float,
    air_density: float = AIR_DENSITY_KG_PER_IN3,
) -> tuple[float, float, float, float, float, float]:
    """Linear drag, Magnus lift, and spin damping in FTC inches.

    Forces are kg*in/s^2; torques are kg*in^2/s^2.
    """
    speed = math.sqrt(vx * vx + vy * vy + vz * vz)
    area = math.pi * radius_in * radius_in
    fx = fy = fz = 0.0
    if speed > 1e-6 and drag_coefficient > 0:
        drag = 0.5 * drag_coefficient * air_density * area * speed
        fx -= drag * vx
        fy -= drag * vy
        fz -= drag * vz
    if magnus_coefficient > 0:
        mx = wy * vz - wz * vy
        my = wz * vx - wx * vz
        mz = wx * vy - wy * vx
        magnus = magnus_coefficient * air_density * area * radius_in
        fx += magnus * mx
        fy += magnus * my
        fz += magnus * mz
    spin = math.sqrt(wx * wx + wy * wy + wz * wz)
    tx = ty = tz = 0.0
    if spin > 1e-6 and drag_coefficient > 0:
        ang_drag = 0.5 * drag_coefficient * air_density * math.pi * (radius_in ** 5) * spin
        tx -= ang_drag * wx
        ty -= ang_drag * wy
        tz -= ang_drag * wz
    return fx, fy, fz, tx, ty, tz


def _actuator(mechanism: dict[str, Any], ident: Any) -> dict[str, Any]:
    if not ident:
        return {}
    row = (mechanism.get("actuators") or {}).get(str(ident))
    return row if isinstance(row, dict) else {}


def gate_open_fraction(actuator: dict[str, Any]) -> float:
    travel = actuator.get("travelLimit") or [0.0, 1.0]
    if not isinstance(travel, (list, tuple)) or len(travel) != 2:
        command = float(actuator.get("command") or -1.0)
        return _smoothstep(0.5 * (command + 1.0), 0.0, 1.0)
    lo, hi = float(travel[0]), float(travel[1])
    position = float(actuator.get("position") or lo)
    return max(0.0, min(1.0, (position - lo) / max(hi - lo, 1e-6)))


def _launch_direction(path: dict[str, Any], actuators: dict[str, Any]) -> tuple[float, float, float, float]:
    muzzle = dict(path.get("muzzlePose") or {})
    pitch_deg = float(muzzle.get("pitchDeg") or 0.0)
    yaw_deg = float(muzzle.get("yawDeg") or muzzle.get("headingDeg") or 0.0)
    hood = _actuator({"actuators": actuators}, path.get("hoodActuatorId"))
    if hood:
        pitch_deg = float(hood.get("position") or pitch_deg)
    turret = _actuator({"actuators": actuators}, path.get("turretActuatorId"))
    if turret:
        yaw_deg = float(turret.get("position") or yaw_deg)
    pitch = math.radians(pitch_deg)
    yaw = math.radians(yaw_deg)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    return cp * cy, cp * sy, sp, pitch_deg


def mechanism_piece_force(
    *,
    piece_local: tuple[float, float, float],
    piece_vel_local: tuple[float, float, float],
    piece_radius: float,
    piece_mass: float,
    mechanism: dict[str, Any],
    owned: bool = False,
) -> PieceInteraction:
    """Contact-like spatial forces from intake, conveyor, gate, and flywheel state.

    Never assigns a launch velocity. Gate fraction, wheel RPM, and hood/turret
    pose scale and aim the force. A closed gate produces no muzzle force.
    """
    path = mechanism.get("piecePath") or {}
    if not path:
        return PieceInteraction()
    chassis = mechanism.get("chassis") or {}
    half_x = 0.5 * float(chassis.get("lengthIn") or 18.0)
    half_y = 0.5 * float(chassis.get("widthIn") or 18.0)
    height = float(chassis.get("heightIn") or 14.0)
    px, py, pz = piece_local
    vx, vy, vz = piece_vel_local
    mass = max(float(piece_mass), 1e-4)
    radius = max(float(piece_radius), 0.1)
    if not owned and (
        abs(px) > half_x + 9.0 + radius
        or abs(py) > half_y + 8.0 + radius
        or pz > height + radius + 4.0
        or pz < -1.0
    ):
        return PieceInteraction()
    fx = fy = fz = 0.0
    tx = ty = tz = 0.0
    flywheel_load = 0.0

    intake = _actuator(mechanism, path.get("intakeActuatorId"))
    conveyor = _actuator(mechanism, path.get("conveyorActuatorId"))
    flywheel = _actuator(mechanism, path.get("flywheelActuatorId"))
    gate = _actuator(mechanism, path.get("gateActuatorId"))
    intake_rpm = abs(float(intake.get("rpm") or 0.0))
    conveyor_rpm = abs(float(conveyor.get("rpm") or 0.0))
    flywheel_rpm = float(flywheel.get("rpm") or 0.0)
    flywheel_target = max(abs(float(flywheel.get("targetRpm") or 4500.0)), 1.0)
    intake_target = max(abs(float(intake.get("targetRpm") or 300.0)), 1.0)
    conveyor_target = max(abs(float(conveyor.get("targetRpm") or 250.0)), 1.0)
    open_frac = gate_open_fraction(gate) if gate else 0.0
    launch_scale = _smoothstep(open_frac, 0.35, 0.85)
    rpm_frac = min(1.0, abs(flywheel_rpm) / flywheel_target)

    intake_pose = dict(path.get("intakePose") or {"x": half_x, "y": 0.0, "z": 2.0})
    ix = float(intake_pose.get("x") or half_x)
    iy = float(intake_pose.get("y") or 0.0)
    iz = float(intake_pose.get("z") or 2.0)
    slots = list(path.get("storageSlots") or [])
    if slots:
        slot_xs = [float(slot.get("x") or 0.0) for slot in slots]
        slot_z = float(slots[0].get("z") or 4.0)
        magazine_x = sum(slot_xs) / len(slot_xs)
    else:
        magazine_x = 0.0
        slot_z = 4.0

    dx_in, dy_in, dz_in = px - ix, py - iy, pz - iz
    intake_dist = math.sqrt(dx_in * dx_in + dy_in * dy_in + dz_in * dz_in)
    intake_reach = 7.0 + radius
    intake_on = intake_rpm > 8.0
    if intake_on and intake_dist < intake_reach and pz < height + radius:
        pull = (intake_rpm / intake_target) * _smoothstep(intake_reach - intake_dist, 0.0, intake_reach)
        tau = 0.06
        into_x = (ix - 2.5) - px
        desired_vx = 28.0 * math.copysign(1.0, into_x if abs(into_x) > 0.2 else -1.0) * pull
        desired_vy = -py * 8.0 * pull
        desired_vz = (iz - pz) * 6.0 * pull
        fx += mass * (desired_vx - vx) / tau
        fy += mass * (desired_vy - vy) / tau
        fz += mass * (desired_vz - vz) / tau

    inside = (
        abs(px) <= half_x + radius + 1.0
        and abs(py) <= half_y + radius
        and -0.5 <= pz <= height + radius
    )
    conveyor_on = conveyor_rpm > 8.0 or owned
    feed_out = launch_scale > 0.2 and rpm_frac > 0.2
    if (inside or owned) and (conveyor_on or feed_out):
        tau = 0.08
        belt = min(1.0, conveyor_rpm / conveyor_target) if conveyor_rpm > 8.0 else 0.35
        if feed_out:
            muzzle = dict(path.get("muzzlePose") or {})
            target_x = float(muzzle.get("x") or ix)
            target_z = float(muzzle.get("z") or slot_z)
            speed = 36.0 * max(belt, 0.45)
            desired_vx = max(speed, (target_x - px) * 12.0)
            if px < target_x - 2.5:
                desired_vz = (slot_z + 1.0 - pz) * 8.0
            else:
                desired_vz = (target_z - pz) * 12.0
        else:
            target_x = magazine_x
            target_z = slot_z
            speed = 18.0 * belt
            desired_vx = (target_x - px) * 4.0
            desired_vx = max(-speed, min(speed, desired_vx))
            desired_vz = (target_z - pz) * 8.0 * belt
        fy += mass * (-vy - py * 10.0) / tau * max(belt, 0.25)
        fx += mass * (desired_vx - vx) / tau
        fz += mass * (desired_vz - vz) / tau

    dx_l, dy_l, dz_l, _pitch_deg = _launch_direction(path, mechanism.get("actuators") or {})
    muzzle = dict(path.get("muzzlePose") or {})
    mx = float(muzzle.get("x") or half_x)
    my = float(muzzle.get("y") or 0.0)
    mz = float(muzzle.get("z") or 10.0)
    wheel_r = float(path.get("wheelRadiusIn") or 2.0)
    compression = float(path.get("compressionIn") or 0.25)
    efficiency = float(path.get("launchEfficiency") or 0.2)
    # Assembly presets provide the wheel's actual chassis-relative world-height
    # center. Older physical presets retain the template-derived fallback.
    wheel_pose = path.get("flywheelPose")
    if isinstance(wheel_pose, dict):
        flywheel_x = float(wheel_pose["x"])
        flywheel_y = float(wheel_pose["y"])
        flywheel_z = float(wheel_pose["z"])
    else:
        flywheel_x = mx - dx_l * (wheel_r + 1.0)
        flywheel_y = my - dy_l * (wheel_r + 1.0)
        flywheel_z = mz - dz_l * (wheel_r + 1.0)
    rel_x, rel_y, rel_z = px - flywheel_x, py - flywheel_y, pz - flywheel_z
    contact_r = wheel_r + radius + compression + 1.5
    dist = math.sqrt(rel_x * rel_x + rel_y * rel_y + rel_z * rel_z)
    contact = _smoothstep(contact_r - dist, 0.0, contact_r)
    surface = abs(flywheel_rpm) * RPM_TO_RAD_S * wheel_r
    if contact > 0.0 and abs(flywheel_rpm) > 1.0:
        if launch_scale > 0.0:
            desired = efficiency * surface * math.copysign(1.0, flywheel_rpm if flywheel_rpm != 0 else 1.0)
            v_along = vx * dx_l + vy * dy_l + vz * dz_l
            tau = 0.05
            gain = contact * launch_scale * rpm_frac
            mag = mass * (desired - v_along) / tau * gain
            fx += mag * dx_l
            fy += mag * dy_l
            fz += mag * dz_l
            flywheel_load = -wheel_r * mag
            spin = -math.copysign(surface / max(radius, 1e-3), desired)
            ty += 0.04 * mass * radius * radius * spin * gain
        elif dist < contact_r:
            back = contact * (1.0 - launch_scale)
            fx -= mass * 18.0 * back * dx_l
            fy += mass * (-vy) / 0.08 * back
            fz += mass * ((slot_z - pz) * 6.0 - vz) / 0.08 * back

    fx, fy, fz = _clip_force(fx, fy, fz, mass)
    return PieceInteraction(
        fx=fx,
        fy=fy,
        fz=fz,
        tx=tx,
        ty=ty,
        tz=tz,
        flywheel_load_inch=flywheel_load,
    )


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
    """Phase 0 production fallback: planar contacts for chassis, walls, and floor pieces."""

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
    poly = body.world_footprint()
    me = body.aabb()
    for box in boxes:
        if poly is not None:
            hit, mtv = polygons_overlap(poly, aabb_polygon(box))
            if hit and mtv is not None:
                wall_hit = True
                body.x += mtv[0]
                body.y += mtv[1]
                body.vx = 0.0
                body.vy = 0.0
                poly = body.world_footprint()
            continue
        if me.overlaps_aabb(box):
            wall_hit = True
            _separate_aabb(body, box)
            me = body.aabb()
    for ob in others:
        if ob.id == body.id:
            continue
        other_poly = ob.world_footprint()
        if poly is not None and other_poly is not None:
            hit, mtv = polygons_overlap(poly, other_poly)
            if hit and mtv is not None:
                robot_hit = True
                body.x += mtv[0]
                body.y += mtv[1]
                body.vx = 0.0
                poly = body.world_footprint()
            continue
        if poly is not None:
            hit, mtv = polygons_overlap(poly, aabb_polygon(ob.aabb()))
            if hit and mtv is not None:
                robot_hit = True
                body.x += mtv[0]
                body.y += mtv[1]
                body.vx = 0.0
                poly = body.world_footprint()
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
    r = piece.radius or max(piece.hx, piece.hy)
    poly = robot.world_footprint()
    if poly is not None:
        hit, push = polygon_vs_circle(poly, piece.x, piece.y, r)
        if not hit or push is None:
            return False
        piece.x += push[0]
        piece.y += push[1]
        piece.vx += robot.vx * 0.45
        piece.vy += robot.vy * 0.45
        return True
    box = robot.aabb()
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
