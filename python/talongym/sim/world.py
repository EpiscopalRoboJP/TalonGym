from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Any

import numpy as np

from talongym.presets.loader import LoadedPresets
from talongym.robot.drivetrain import clip_twist
from talongym.robot.sensors import camera_world_pose, detect_tags
from talongym.rules.engine import MECHANISM_VERBS, RuleContext, RuleEngine, TickEvent
from talongym.sim.geometry import AABB, deg_to_rad, point_in_shape, rad_to_deg, shape_from_element, wrap_angle
from talongym.sim.physics import Body, WorldStep, default_backend, perimeter_walls


@dataclass
class Piece:
    id: str
    type_id: str
    x: float
    y: float
    radius: float
    attrs: dict[str, Any]
    held_by: str | None = None
    in_flight: bool = False
    flight: list[tuple[str, float, float]] = field(default_factory=list)
    flight_i: int = 0
    scored: bool = False
    vx: float = 0.0
    vy: float = 0.0
    restitution: float = 0.3
    mass: float = 0.1


@dataclass
class RobotState:
    body: Body
    held: list[str]
    intake_timer: float = 0.0
    score_timer: float = 0.0
    last_verb: str = "idle"
    collision_time_s: float = 0.0
    first_contact_s: float | None = None
    entered_restricted: bool = False
    vision_hits: list[dict[str, Any]] = field(default_factory=list)


class World:
    def __init__(self, bundle: LoadedPresets, seed: int = 0, control_hz: int = 25, substeps: int = 2) -> None:
        self.bundle = bundle
        self.field = bundle.field
        self.robot = bundle.robot
        self.scoring = bundle.scoring
        self.rng = np.random.default_rng(seed)
        self.control_hz = control_hz
        self.dt = 1.0 / control_hz / substeps
        self.substeps = substeps
        self.engine = RuleEngine(self.scoring)
        fw = float(self.field["fieldSizeIn"]["width"])
        fd = float(self.field["fieldSizeIn"]["depth"])
        self.backend = default_backend(fw / 2.0, fd / 2.0)
        self.walls = perimeter_walls(fw / 2.0, fd / 2.0)
        self.elements = list(self.field.get("elements") or [])
        self.element_shapes = {el["id"]: shape_from_element(el) for el in self.elements}
        self.triggers = {
            (el.get("triggerId") or el["id"]): el
            for el in self.elements
            if el.get("isTrigger")
        }
        self.obstacles: list[AABB] = []
        for el in self.elements:
            if not el.get("isCollider"):
                continue
            if "perimeter" in (el.get("tags") or []):
                continue
            sh = self.element_shapes.get(el["id"])
            if isinstance(sh, AABB) and sh.hx < 70 and sh.hy < 70:
                self.obstacles.append(sh)
        self.occluders: list[AABB] = []
        for el in self.elements:
            if el.get("isOccluder"):
                sh = self.element_shapes.get(el["id"])
                if isinstance(sh, AABB):
                    self.occluders.append(sh)
        self.piece_types = {p["typeId"]: p for p in self.field.get("gamePieces") or []}
        self.tags = list(self.field.get("aprilTags") or [])
        self.constraints = self.robot.get("constraints") or {}
        self.max_vel = float(self.constraints.get("maxVelInPerS", 30))
        self.max_accel = float(self.constraints.get("maxAccelInPerS2", 30))
        self.max_ang_vel = math.radians(float(self.constraints.get("maxAngVelDegPerS", 60)))
        self.max_ang_accel = math.radians(float(self.constraints.get("maxAngAccelDegPerS2", 60)))
        chassis = self.robot.get("chassis") or {}
        self.robot_hx = float(chassis.get("widthIn", 18)) / 2.0
        self.robot_hy = float(chassis.get("lengthIn", 18)) / 2.0
        mech = self.robot.get("mechanisms") or {}
        self.capacity = int(mech.get("capacity", 3))
        self.intake_time = float(mech.get("intakeCycleTimeS", 0.4))
        self.score_time = float(mech.get("scoreCycleTimeS", 0.6))
        odo = self.robot.get("odometry") or {}
        self.pos_noise = float(odo.get("positionNoiseStdIn", 0.3))
        self.heading_noise = math.radians(float(odo.get("headingNoiseStdDeg", 1.0)))
        self.cameras = [s for s in self.robot.get("sensors") or [] if s.get("kind") == "apriltag_camera"]
        self.gate_ids = [
            el["id"]
            for el in self.elements
            if el.get("type") == "gate" or "gate" in (el.get("tags") or [])
        ]
        self.seq_accs = [
            a["id"] for a in (self.scoring.get("accumulators") or []) if a.get("valueType") == "sequence"
        ]
        self.queue_caps = {
            a["id"]: int(a.get("maxLen") or 9)
            for a in (self.scoring.get("accumulators") or [])
            if a.get("valueType") == "sequence"
        }
        self.time_s = 0.0
        self.phase = "AUTO"
        self.auto_s = 30.0
        for ph in self.scoring.get("phases") or []:
            if ph.get("id") == "AUTO":
                self.auto_s = float(ph.get("durationS", 30))
        self.robots: dict[str, RobotState] = {}
        self.pieces: dict[str, Piece] = {}
        self.accumulators: dict[str, Any] = {}
        self.match_vars: dict[str, Any] = {}
        self.observed_vars: dict[str, Any] = {}
        self.gate_state: dict[str, str] = {gid: "closed" for gid in self.gate_ids}
        self.queues: dict[str, list[str]] = {sid: [] for sid in self.seq_accs}
        self.fire_counts: dict[str, int] = {}
        self.prev_occupancy: dict[str, set[str]] = {}
        self.explains: list[dict[str, Any]] = []
        self.true_score = 0.0
        self.wall_hit = False
        self.robot_hit = False
        self.piece_hit = False
        self.vision_hits: list[dict[str, Any]] = []
        self.last_events: list[TickEvent] = []
        self.pending_piece_ops: list[tuple[str, str | None, str | None]] = []

    def reset(
        self,
        seed: int | None = None,
        static_teammate: bool = True,
        opponent_mode: str = "none",
        live_teammate: bool = False,
    ) -> None:
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.time_s = 0.0
        self.phase = "AUTO"
        self.true_score = 0.0
        self.explains = []
        self.fire_counts = {}
        self.gate_state = {gid: "closed" for gid in self.gate_ids}
        self.queues = {sid: [] for sid in self.seq_accs}
        self.accumulators = self.engine.init_accumulators()
        match_vals, acc_seed = self.engine.init_match_vars(self.rng)
        self.match_vars = match_vals
        self.accumulators.update(acc_seed)
        self.observed_vars = {k: None for k in self.match_vars}
        self.pieces = {}
        pid = 0
        for spawn in self.field.get("spawns") or []:
            ptype = spawn["pieceTypeId"]
            spec = self.piece_types[ptype]
            sh = spec.get("shape") or {}
            if sh.get("kind") == "circle":
                rad = float(sh.get("radius") or 2.5)
            else:
                rad = max(float(sh.get("width") or 3.0), float(sh.get("depth") or 3.0)) / 2.0
            color = (spec.get("attributes") or {}).get("color")
            for pose in spawn.get("poses") or []:
                jx = float(self.rng.normal(0, 0.2))
                jy = float(self.rng.normal(0, 0.2))
                name = f"p{pid}"
                pid += 1
                piece = Piece(
                    id=name,
                    type_id=ptype,
                    x=float(pose["x"]) + jx,
                    y=float(pose["y"]) + jy,
                    radius=rad,
                    attrs={"color": color, "passed_goal_top": False, "passed_archway": False},
                    restitution=float(spec.get("restitution") or 0.3),
                    mass=float(spec.get("massKg") or 0.1),
                )
                self.pieces[name] = piece
        self.robots = {}
        self._spawn_robots(static_teammate=static_teammate, opponent_mode=opponent_mode, live_teammate=live_teammate)
        self.prev_occupancy = self._occupancy()
        self.vision_hits = []
        self.pending_piece_ops = []
        self.backend.reset_batch(1)
        self._sense()

    def _start_slot(self, alliance: str, slot: int) -> dict[str, Any] | None:
        found = [
            s
            for s in (self.field.get("startSlots") or [])
            if s.get("alliance") == alliance and int(s.get("slot", 0)) == slot
        ]
        if found:
            return found[0]
        alts = [s for s in (self.field.get("startSlots") or []) if s.get("alliance") == alliance]
        alts = sorted(alts, key=lambda s: int(s.get("slot", 0)))
        if slot < len(alts):
            return alts[slot]
        return None

    def _make_robot(self, rid: str, alliance: str, slot: dict[str, Any], dynamic: bool) -> RobotState:
        pose = slot["pose"]
        return RobotState(
            body=Body(
                rid,
                float(pose["x"]),
                float(pose["y"]),
                deg_to_rad(float(pose["headingDeg"])),
                hx=self.robot_hx,
                hy=self.robot_hy,
                dynamic=dynamic,
                alliance=alliance,
            ),
            held=[],
        )

    def _spawn_robots(self, static_teammate: bool, opponent_mode: str, live_teammate: bool) -> None:
        red0 = self._start_slot("red", 0) or {
            "id": "red_0",
            "alliance": "red",
            "slot": 0,
            "pose": {"x": 0, "y": -48, "headingDeg": 90},
        }
        self.robots["red_0"] = self._make_robot("red_0", "red", red0, True)
        red1 = self._start_slot("red", 1)
        if red1 is not None and (live_teammate or static_teammate):
            self.robots["red_1"] = self._make_robot("red_1", "red", red1, bool(live_teammate))
        mode = opponent_mode or "none"
        if mode != "none":
            dynamic_opp = mode != "static"
            for i, key in enumerate(("blue_0", "blue_1")):
                sl = self._start_slot("blue", i)
                if sl is None:
                    continue
                self.robots[key] = self._make_robot(key, "blue", sl, dynamic_opp)

    def actor(self) -> RobotState:
        return self.robots["red_0"]

    def _occupancy(self) -> dict[str, set[str]]:
        occ: dict[str, set[str]] = {tid: set() for tid in self.triggers}
        for tid, el in self.triggers.items():
            sh = self.element_shapes[el["id"]]
            for rid, rs in self.robots.items():
                if point_in_shape(sh, rs.body.x, rs.body.y):
                    occ[tid].add(rid)
            for p in self.pieces.values():
                if p.held_by or p.scored or p.in_flight:
                    continue
                if point_in_shape(sh, p.x, p.y):
                    occ[tid].add(p.id)
        return occ

    def _events_from_occupancy(self, occ: dict[str, set[str]]) -> list[TickEvent]:
        events: list[TickEvent] = []
        for tid, now in occ.items():
            prev = self.prev_occupancy.get(tid, set())
            for ident in now - prev:
                if ident in self.robots:
                    events.append(TickEvent("volumeEnter", volume_id=tid, robot_id=ident))
                    events.append(TickEvent("contactStart", volume_id=tid, robot_id=ident))
                elif ident in self.pieces:
                    p = self.pieces[ident]
                    events.append(
                        TickEvent(
                            "volumeEnter",
                            volume_id=tid,
                            piece_id=p.id,
                            piece_type=p.type_id,
                            piece_attrs=dict(p.attrs),
                        )
                    )
            for ident in prev - now:
                if ident in self.robots:
                    events.append(TickEvent("volumeExit", volume_id=tid, robot_id=ident))
        return events

    def _apply_follower(self, rs: RobotState, target: np.ndarray, speed_frac: float, dt: float) -> None:
        if not rs.body.dynamic:
            rs.body.vx = rs.body.vy = rs.body.omega = 0.0
            return
        tx, ty, th = float(target[0]), float(target[1]), float(target[2])
        ex, ey = tx - rs.body.x, ty - rs.body.y
        dist = math.hypot(ex, ey)
        speed = self.max_vel * float(np.clip(speed_frac, 0.2, 1.0))
        if dist > 1e-3:
            des_vx = speed * ex / dist
            des_vy = speed * ey / dist
        else:
            des_vx = des_vy = 0.0
        heading_err = wrap_angle(th - rs.body.heading)
        des_w = float(np.clip(2.5 * heading_err, -self.max_ang_vel, self.max_ang_vel))
        dvx = float(np.clip(des_vx - rs.body.vx, -self.max_accel * dt, self.max_accel * dt))
        dvy = float(np.clip(des_vy - rs.body.vy, -self.max_accel * dt, self.max_accel * dt))
        dw = float(np.clip(des_w - rs.body.omega, -self.max_ang_accel * dt, self.max_ang_accel * dt))
        rs.body.vx += dvx
        rs.body.vy += dvy
        rs.body.omega += dw
        vx, vy, om = clip_twist(rs.body.vx, rs.body.vy, rs.body.omega, self.robot)
        rs.body.vx, rs.body.vy, rs.body.omega = vx, vy, om

    def _apply_velocity(self, rs: RobotState, vx: float, vy: float, omega: float, dt: float) -> None:
        if not rs.body.dynamic:
            rs.body.vx = rs.body.vy = rs.body.omega = 0.0
            return
        des_vx, des_vy, des_w = clip_twist(vx, vy, omega, self.robot)
        dvx = float(np.clip(des_vx - rs.body.vx, -self.max_accel * dt, self.max_accel * dt))
        dvy = float(np.clip(des_vy - rs.body.vy, -self.max_accel * dt, self.max_accel * dt))
        dw = float(np.clip(des_w - rs.body.omega, -self.max_ang_accel * dt, self.max_ang_accel * dt))
        rs.body.vx += dvx
        rs.body.vy += dvy
        rs.body.omega += dw
        rs.body.vx, rs.body.vy, rs.body.omega = clip_twist(rs.body.vx, rs.body.vy, rs.body.omega, self.robot)

    def _apply_command(self, rs: RobotState, action: dict[str, Any], dt: float) -> None:
        vel = action.get("velocity")
        if vel is not None:
            arr = np.asarray(vel, dtype=np.float64).reshape(-1)
            self._apply_velocity(
                rs,
                float(arr[0]),
                float(arr[1]) if arr.size > 1 else 0.0,
                float(arr[2]) if arr.size > 2 else 0.0,
                dt,
            )
            return
        target = np.asarray(
            action.get("target_pose", [rs.body.x, rs.body.y, rs.body.heading]),
            dtype=np.float64,
        ).reshape(3)
        speed = float(action.get("speed_frac", 0.8))
        self._apply_follower(rs, target, speed, dt)

    def _mechanisms(self, rs: RobotState, verb: str, dt: float, events: list[TickEvent]) -> None:
        rs.last_verb = verb
        if verb == "intake" and len(rs.held) < self.capacity:
            grabbed = False
            for p in self.pieces.values():
                if p.held_by or p.in_flight or p.scored:
                    continue
                if math.hypot(p.x - rs.body.x, p.y - rs.body.y) <= (self.robot_hx + p.radius + 4.0):
                    rs.intake_timer += dt
                    if rs.intake_timer >= self.intake_time:
                        p.held_by = rs.body.id
                        p.vx = p.vy = 0.0
                        p.x, p.y = rs.body.x, rs.body.y
                        rs.held.append(p.id)
                        rs.intake_timer = 0.0
                    grabbed = True
                    break
            if not grabbed:
                rs.intake_timer = 0.0
        else:
            rs.intake_timer = 0.0
        if verb == "score" and rs.held and rs.score_timer <= 0:
            pid = rs.held.pop(0)
            p = self.pieces[pid]
            p.held_by = None
            p.vx = p.vy = 0.0
            p.attrs["passed_goal_top"] = False
            p.attrs["passed_archway"] = False
            waypoints: list[tuple[str, float, float]] = []
            for tag in ("open_top", "archway", "square"):
                el = next(
                    (
                        e
                        for e in self.elements
                        if tag in (e.get("tags") or []) and e.get("alliance") == rs.body.alliance
                    ),
                    None,
                )
                if el:
                    pose = el["pose"]
                    waypoints.append((el.get("triggerId") or el["id"], float(pose["x"]), float(pose["y"])))
            if waypoints:
                p.in_flight = True
                p.flight = waypoints
                p.flight_i = 0
            else:
                target = self._nearest_score_volume(rs)
                if target:
                    pose = target["pose"]
                    p.x, p.y = float(pose["x"]), float(pose["y"])
                    p.in_flight = False
                    p.scored = False
                    events.append(
                        TickEvent(
                            "volumeEnter",
                            volume_id=target.get("triggerId") or target["id"],
                            piece_id=p.id,
                            piece_type=p.type_id,
                            piece_attrs=dict(p.attrs),
                        )
                    )
                else:
                    p.x, p.y = rs.body.x, rs.body.y
            rs.score_timer = self.score_time
        if verb == "open_gate":
            gate_el = next(
                (e for e in self.elements if e.get("type") == "gate" or "gate" in (e.get("tags") or [])),
                None,
            )
            if gate_el:
                sh = self.element_shapes[gate_el["id"]]
                if point_in_shape(sh, rs.body.x, rs.body.y):
                    self.gate_state[gate_el["id"]] = "open"
                    events.append(
                        TickEvent(
                            "contactStart",
                            volume_id=gate_el.get("triggerId") or gate_el["id"],
                            robot_id=rs.body.id,
                            fsm_id=gate_el["id"],
                        )
                    )
                    self.pending_piece_ops.append(("release_queue", None, None))
        rs.score_timer = max(0.0, rs.score_timer - dt)

    def _nearest_score_volume(self, rs: RobotState) -> dict[str, Any] | None:
        tags = {"basket", "chamber", "net", "backdrop", "backstage", "goal", "high", "low", "park"}
        best = None
        best_d = 1e9
        for el in self.elements:
            el_tags = set(el.get("tags") or [])
            if not el_tags.intersection(tags):
                continue
            if el.get("alliance") not in {rs.body.alliance, "neutral", None}:
                continue
            pose = el.get("pose") or {}
            d = math.hypot(float(pose.get("x", 0)) - rs.body.x, float(pose.get("y", 0)) - rs.body.y)
            if d < best_d:
                best_d = d
                best = el
        return best

    def _advance_flights(self, events: list[TickEvent], dt: float) -> None:
        for p in self.pieces.values():
            if not p.in_flight or p.scored:
                continue
            if p.flight_i >= len(p.flight):
                p.in_flight = False
                p.scored = True
                continue
            vid, gx, gy = p.flight[p.flight_i]
            dx, dy = gx - p.x, gy - p.y
            dist = math.hypot(dx, dy)
            step = 80.0 * dt
            if dist <= step:
                p.x, p.y = gx, gy
                if "goal" in vid or "top" in vid:
                    p.attrs["passed_goal_top"] = True
                if "arch" in vid:
                    p.attrs["passed_archway"] = True
                events.append(
                    TickEvent(
                        "volumeEnter",
                        volume_id=vid,
                        piece_id=p.id,
                        piece_type=p.type_id,
                        piece_attrs=dict(p.attrs),
                    )
                )
                p.flight_i += 1
                if p.flight_i >= len(p.flight):
                    p.in_flight = False
                    p.scored = True
                    p.x, p.y = gx, gy
            else:
                p.x += step * dx / dist
                p.y += step * dy / dist

    def _sense_robot(self, rs: RobotState) -> list[dict[str, Any]]:
        hits_all: list[dict[str, Any]] = []
        for cam in self.cameras:
            cx, cy, ch = camera_world_pose(rs.body.x, rs.body.y, rs.body.heading, cam)
            hits = detect_tags(cx, cy, ch, cam, self.tags, self.occluders, self.rng)
            hits_all.extend(hits)
            for hit in hits:
                if not hit.get("visible") or not hit.get("mapsTo"):
                    continue
                for var in self.scoring.get("matchVariables") or []:
                    role = (var.get("observeVia") or {}).get("tagRole")
                    if role and hit.get("role") == role:
                        self.observed_vars[var["id"]] = hit["mapsTo"]
        rs.vision_hits = hits_all
        return hits_all

    def _sense(self) -> None:
        self.vision_hits = []
        actor_id = self.actor().body.id if self.robots else "red_0"
        for rs in self.robots.values():
            hits = self._sense_robot(rs)
            if rs.body.id == actor_id:
                self.vision_hits = hits

    def _chassis_hits_other(self, rs: RobotState, slack: float = 0.6) -> bool:
        me = rs.body.aabb()
        for other in self.robots.values():
            if other.body.id == rs.body.id:
                continue
            ob = other.body.aabb()
            if (
                me.minx < ob.maxx + slack
                and me.maxx + slack > ob.minx
                and me.miny < ob.maxy + slack
                and me.maxy + slack > ob.miny
            ):
                return True
        return False

    def _update_contacts_and_restricted(self, dt: float, occ: dict[str, set[str]]) -> None:
        for rs in self.robots.values():
            if self._chassis_hits_other(rs):
                rs.collision_time_s += dt
                if rs.first_contact_s is None:
                    rs.first_contact_s = self.time_s
            tag = f"restricted_for_{rs.body.alliance}"
            for tid, el in self.triggers.items():
                tags = el.get("tags") or []
                if tag in tags or el.get("id") == tag or el.get("triggerId") == tag:
                    if rs.body.id in occ.get(tid, set()):
                        rs.entered_restricted = True

    def _run_rules(self, events: list[TickEvent]) -> float:
        occ = self._occupancy()
        events.extend(self._events_from_occupancy(occ))
        rs = self.actor()
        actor_vols = {tid for tid, ids in occ.items() if rs.body.id in ids}
        ctx = RuleContext(
            phase=self.phase,
            events=events,
            volume_occupancy=occ,
            prev_occupancy=self.prev_occupancy,
            accumulators=self.accumulators,
            match_vars=self.match_vars,
            gate_state=self.gate_state,
            queues=self.queues,
            queue_caps=self.queue_caps,
            actor_id=rs.body.id,
            actor_volumes=actor_vols,
            fire_counts=self.fire_counts,
            ident_types={p.id: p.type_id for p in self.pieces.values()},
            piece_ops=self.pending_piece_ops,
        )
        explains, delta = self.engine.evaluate(ctx)
        self.explains.extend(explains)
        self.true_score = float(self.accumulators.get(self.engine.true_score_id) or self.true_score)
        self.prev_occupancy = occ
        self.last_events = events
        self._apply_piece_ops()
        return float(delta)

    def _apply_piece_ops(self) -> None:
        ops = list(self.pending_piece_ops)
        self.pending_piece_ops = []
        for kind, piece_id, extra in ops:
            if kind == "despawn" and piece_id and piece_id in self.pieces:
                p = self.pieces[piece_id]
                p.scored = True
                p.in_flight = False
                p.held_by = None
                p.vx = p.vy = 0.0
                for rs in self.robots.values():
                    if piece_id in rs.held:
                        rs.held.remove(piece_id)
            elif kind == "transfer" and piece_id and piece_id in self.pieces:
                p = self.pieces[piece_id]
                p.scored = True
                p.in_flight = False
                p.held_by = None
                if extra:
                    el = next((e for e in self.elements if e["id"] == extra or e.get("triggerId") == extra), None)
                    if el:
                        pose = el.get("pose") or {}
                        p.x, p.y = float(pose.get("x", p.x)), float(pose.get("y", p.y))
            elif kind == "release_queue":
                for sid in list(self.queues):
                    self.queues[sid] = []
                    if sid in self.accumulators and isinstance(self.accumulators[sid], list):
                        self.accumulators[sid] = []
                for gid in self.gate_state:
                    self.gate_state[gid] = "open"

    def step(
        self,
        target_pose: np.ndarray | None = None,
        speed_frac: float = 1.0,
        mechanism: int = 0,
        end_phase: bool = False,
        actions: dict[str, dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        if actions is None:
            if target_pose is None:
                rs0 = self.actor()
                target_pose = np.array([rs0.body.x, rs0.body.y, rs0.body.heading], dtype=np.float64)
            actions = {
                "red_0": {
                    "target_pose": np.asarray(target_pose, dtype=np.float64).reshape(3),
                    "speed_frac": float(speed_frac),
                    "mechanism": int(mechanism),
                }
            }
        events: list[TickEvent] = []
        self.wall_hit = False
        self.robot_hit = False
        self.piece_hit = False
        for _ in range(self.substeps):
            for rid, rs in self.robots.items():
                act = actions.get(rid) or {"target_pose": [rs.body.x, rs.body.y, rs.body.heading], "speed_frac": 0.2, "mechanism": 0}
                self._apply_command(rs, act, self.dt)
                verb = MECHANISM_VERBS[int(act.get("mechanism", 0)) % len(MECHANISM_VERBS)]
                self._mechanisms(rs, verb, self.dt, events)
            floor_bodies: list[Body] = []
            floor_index: dict[str, Piece] = {}
            for p in self.pieces.values():
                if p.held_by or p.in_flight or p.scored:
                    continue
                body = Body(
                    p.id,
                    p.x,
                    p.y,
                    0.0,
                    vx=p.vx,
                    vy=p.vy,
                    hx=p.radius,
                    hy=p.radius,
                    kind="circle",
                    radius=p.radius,
                    restitution=p.restitution,
                    mass=p.mass,
                )
                floor_bodies.append(body)
                floor_index[p.id] = p
            flags = self.backend.step_world(
                WorldStep(
                    robots=[o.body for o in self.robots.values()],
                    pieces=floor_bodies,
                    obstacles=self.obstacles,
                    walls=self.walls,
                    max_vel=self.max_vel,
                    max_accel=self.max_accel,
                    max_omega=self.max_ang_vel,
                    max_ang_accel=self.max_ang_accel,
                ),
                self.dt,
            )
            self.wall_hit = self.wall_hit or flags.wall
            self.robot_hit = self.robot_hit or flags.robot
            self.piece_hit = self.piece_hit or flags.piece
            for body in floor_bodies:
                p = floor_index[body.id]
                p.x, p.y, p.vx, p.vy = body.x, body.y, body.vx, body.vy
            self._advance_flights(events, self.dt)
            for p in self.pieces.values():
                if p.held_by and p.held_by in self.robots:
                    holder = self.robots[p.held_by]
                    p.x, p.y = holder.body.x, holder.body.y
                    p.vx = p.vy = 0.0
            self.time_s += self.dt
            occ_mid = self._occupancy()
            self._update_contacts_and_restricted(self.dt, occ_mid)
        self._sense()
        rs = self.actor()
        if end_phase or self.time_s >= self.auto_s - 1e-9:
            events.append(TickEvent("phaseEnd", phase="AUTO", robot_id=rs.body.id))
        verb = MECHANISM_VERBS[int((actions.get("red_0") or {}).get("mechanism", 0)) % len(MECHANISM_VERBS)]
        delta = self._run_rules(events)
        occ = self.prev_occupancy
        self._update_contacts_and_restricted(0.0, occ)
        return {"true_score_delta": delta, "verb": verb}

    def snapshot(self) -> dict[str, Any]:
        rs = self.actor()
        return {
            "t": self.time_s,
            "phase": self.phase,
            "trueScore": self.true_score,
            "robots": [
                {
                    "id": r.body.id,
                    "x": r.body.x,
                    "y": r.body.y,
                    "headingDeg": rad_to_deg(r.body.heading),
                    "held": list(r.held),
                    "dynamic": r.body.dynamic,
                    "alliance": r.body.alliance,
                    "collisionTimeS": r.collision_time_s,
                    "firstContactS": r.first_contact_s,
                    "enteredRestricted": r.entered_restricted,
                }
                for r in self.robots.values()
            ],
            "pieces": [
                {
                    "id": p.id,
                    "typeId": p.type_id,
                    "x": p.x,
                    "y": p.y,
                    "color": p.attrs.get("color"),
                    "heldBy": p.held_by,
                    "inFlight": p.in_flight,
                    "scored": p.scored,
                }
                for p in self.pieces.values()
            ],
            "queues": dict(self.queues),
            "gate": dict(self.gate_state),
            "matchVarsPrivileged": dict(self.match_vars),
            "observedMatchVars": dict(self.observed_vars),
            "vision": self.vision_hits,
            "events": [e.explain if hasattr(e, "explain") else e.kind for e in self.last_events[-8:]],
            "explains": list(self.explains[-6:]),
            "collision": {
                "wall": self.wall_hit,
                "robot": self.robot_hit,
                "piece": self.piece_hit,
                "collisionTimeS": rs.collision_time_s,
                "firstContactS": rs.first_contact_s,
                "enteredRestricted": rs.entered_restricted,
            },
            "fieldSizeIn": self.field["fieldSizeIn"],
            "elements": [
                {
                    "id": el["id"],
                    "type": el.get("type"),
                    "alliance": el.get("alliance"),
                    "pose": el.get("pose") or {"x": 0, "y": 0, "headingDeg": 0},
                    "shape": el.get("shape"),
                    "tags": el.get("tags") or [],
                    "isOccluder": bool(el.get("isOccluder")),
                }
                for el in self.elements
            ],
            "aprilTags": [
                {
                    "tagId": t.get("tagId"),
                    "role": t.get("role"),
                    "pose": t.get("pose") or {"x": 0, "y": 0, "z": 12, "headingDeg": 0},
                    "sizeIn": t.get("sizeIn") or 8,
                    "mapsTo": t.get("mapsTo"),
                }
                for t in (self.field.get("aprilTags") or [])
            ],
            "noisyPose": [
                rs.body.x + float(self.rng.normal(0, self.pos_noise)),
                rs.body.y + float(self.rng.normal(0, self.pos_noise)),
                wrap_angle(rs.body.heading + float(self.rng.normal(0, self.heading_noise))),
            ],
        }
